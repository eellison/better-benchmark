import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512]", primals_2: "i64[1, 512]", primals_3: "i64[1, 512]", primals_4: "f32[32000, 768]", primals_7: "f32[768]", primals_9: "f32[768, 768]", primals_11: "f32[768]", primals_13: "f32[3072, 768]", primals_15: "f32[768, 3072]", primals_17: "f32[768]", primals_19: "f32[768]", primals_21: "f32[3072, 768]", primals_23: "f32[768, 3072]", primals_25: "f32[768]", primals_27: "f32[768]", primals_29: "f32[3072, 768]", primals_31: "f32[768, 3072]", primals_33: "f32[768]", primals_35: "f32[768]", primals_37: "f32[3072, 768]", primals_39: "f32[768, 3072]", primals_41: "f32[768]", primals_43: "f32[768]", primals_45: "f32[3072, 768]", primals_47: "f32[768, 3072]", primals_49: "f32[768]", primals_51: "f32[768]", primals_53: "f32[3072, 768]", primals_55: "f32[768, 3072]", primals_57: "f32[768]", primals_59: "f32[768]", primals_61: "f32[3072, 768]", primals_63: "f32[768, 3072]", primals_65: "f32[768]", primals_67: "f32[768]", primals_69: "f32[3072, 768]", primals_71: "f32[768, 3072]", primals_73: "f32[768]", primals_75: "f32[768]", primals_77: "f32[3072, 768]", primals_79: "f32[768, 3072]", primals_81: "f32[768]", primals_83: "f32[768]", primals_85: "f32[3072, 768]", primals_87: "f32[768, 3072]", primals_89: "f32[768]", primals_91: "f32[768]", primals_93: "f32[3072, 768]", primals_95: "f32[768, 3072]", primals_97: "f32[768]", primals_99: "f32[768]", primals_101: "f32[3072, 768]", primals_103: "f32[768, 3072]", primals_105: "f32[768]", primals_109: "f32[768, 768]", primals_111: "f32[768]", primals_114: "i64[32, 512]", mul: "f32[32, 512, 768]", view: "f32[16384, 768]", gt: "b8[32, 512, 768]", mul_4: "f32[32, 512, 768]", view_2: "f32[16384, 768]", addmm_1: "f32[16384, 3072]", view_4: "f32[16384, 3072]", gt_1: "b8[32, 512, 768]", mul_12: "f32[32, 512, 768]", mul_14: "f32[32, 512, 768]", view_6: "f32[16384, 768]", addmm_3: "f32[16384, 3072]", view_8: "f32[16384, 3072]", gt_2: "b8[32, 512, 768]", mul_22: "f32[32, 512, 768]", mul_24: "f32[32, 512, 768]", view_10: "f32[16384, 768]", addmm_5: "f32[16384, 3072]", view_12: "f32[16384, 3072]", gt_3: "b8[32, 512, 768]", mul_32: "f32[32, 512, 768]", mul_34: "f32[32, 512, 768]", view_14: "f32[16384, 768]", addmm_7: "f32[16384, 3072]", view_16: "f32[16384, 3072]", gt_4: "b8[32, 512, 768]", mul_42: "f32[32, 512, 768]", mul_44: "f32[32, 512, 768]", view_18: "f32[16384, 768]", addmm_9: "f32[16384, 3072]", view_20: "f32[16384, 3072]", gt_5: "b8[32, 512, 768]", mul_52: "f32[32, 512, 768]", mul_54: "f32[32, 512, 768]", view_22: "f32[16384, 768]", addmm_11: "f32[16384, 3072]", view_24: "f32[16384, 3072]", gt_6: "b8[32, 512, 768]", mul_62: "f32[32, 512, 768]", mul_64: "f32[32, 512, 768]", view_26: "f32[16384, 768]", addmm_13: "f32[16384, 3072]", view_28: "f32[16384, 3072]", gt_7: "b8[32, 512, 768]", mul_72: "f32[32, 512, 768]", mul_74: "f32[32, 512, 768]", view_30: "f32[16384, 768]", addmm_15: "f32[16384, 3072]", view_32: "f32[16384, 3072]", gt_8: "b8[32, 512, 768]", mul_82: "f32[32, 512, 768]", mul_84: "f32[32, 512, 768]", view_34: "f32[16384, 768]", addmm_17: "f32[16384, 3072]", view_36: "f32[16384, 3072]", gt_9: "b8[32, 512, 768]", mul_92: "f32[32, 512, 768]", mul_94: "f32[32, 512, 768]", view_38: "f32[16384, 768]", addmm_19: "f32[16384, 3072]", view_40: "f32[16384, 3072]", gt_10: "b8[32, 512, 768]", mul_102: "f32[32, 512, 768]", mul_104: "f32[32, 512, 768]", view_42: "f32[16384, 768]", addmm_21: "f32[16384, 3072]", view_44: "f32[16384, 3072]", gt_11: "b8[32, 512, 768]", mul_112: "f32[32, 512, 768]", mul_114: "f32[32, 512, 768]", view_46: "f32[16384, 768]", addmm_23: "f32[16384, 3072]", view_48: "f32[16384, 3072]", gt_12: "b8[32, 512, 768]", mul_122: "f32[32, 512, 768]", view_50: "f32[16384, 768]", addmm_26: "f32[16384, 768]", getitem_51: "f32[32, 512, 1]", rsqrt_25: "f32[32, 512, 1]", view_52: "f32[16384, 768]", view_53: "f32[32, 512, 32000]", amax: "f32[16384, 1]", log: "f32[16384, 1]", convert_element_type_12: "f32[]", div_3: "f32[32, 512, 1]", div_4: "f32[32, 512, 1]", div_5: "f32[32, 512, 1]", div_6: "f32[32, 512, 1]", div_7: "f32[32, 512, 1]", div_8: "f32[32, 512, 1]", div_9: "f32[32, 512, 1]", div_10: "f32[32, 512, 1]", div_11: "f32[32, 512, 1]", div_12: "f32[32, 512, 1]", div_13: "f32[32, 512, 1]", div_14: "f32[32, 512, 1]", div_15: "f32[32, 512, 1]", div_16: "f32[32, 512, 1]", div_17: "f32[32, 512, 1]", div_18: "f32[32, 512, 1]", div_19: "f32[32, 512, 1]", div_20: "f32[32, 512, 1]", div_21: "f32[32, 512, 1]", div_22: "f32[32, 512, 1]", div_23: "f32[32, 512, 1]", div_24: "f32[32, 512, 1]", div_25: "f32[32, 512, 1]", div_26: "f32[32, 512, 1]", div_27: "f32[32, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[32, 512, 32000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:666 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_12);  tangents_1 = convert_element_type_12 = None
        view_55: "i64[16384]" = torch.ops.aten.reshape.default(primals_114, [-1]);  primals_114 = None
        unsqueeze_1: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(view_55, 1);  view_55 = None
        ne_3: "b8[16384, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[16384, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[32000]" = torch.ops.prims.iota.default(32000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 32000]" = torch.ops.aten.reshape.default(iota_default, [1, 32000]);  iota_default = None
        expand_default: "i64[16384, 32000]" = torch.ops.aten.expand.default(where_2, [16384, 32000]);  where_2 = None
        eq_tensor: "b8[16384, 32000]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:666 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[16384, 32000]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[16384, 1]" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = None
        mul_130: "f32[16384, 32000]" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None
        view_54: "f32[16384, 32000]" = torch.ops.aten.reshape.default(view_53, [-1, 32000]);  view_53 = None
        sub_26: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(view_54, amax);  view_54 = amax = None
        sub_27: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        exp_1: "f32[16384, 32000]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_4: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_130, [1], True)
        mul_131: "f32[16384, 32000]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_28: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        view_56: "f32[32, 512, 32000]" = torch.ops.aten.reshape.default(sub_28, [32, 512, 32000]);  sub_28 = None
        add_104: "f32[32, 512, 32000]" = torch.ops.aten.add.Tensor(tangents_2, view_56);  tangents_2 = view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:331 in forward, code: hidden_states = self.decoder(hidden_states)
        view_57: "f32[16384, 32000]" = torch.ops.aten.reshape.default(add_104, [16384, 32000]);  add_104 = None
        permute_27: "f32[768, 32000]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_28: "f32[32000, 768]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        mm: "f32[16384, 768]" = torch.ops.aten.mm.default(view_57, permute_28);  permute_28 = None
        permute_29: "f32[32000, 16384]" = torch.ops.aten.permute.default(view_57, [1, 0])
        mm_1: "f32[32000, 768]" = torch.ops.aten.mm.default(permute_29, view_52);  permute_29 = view_52 = None
        sum_5: "f32[1, 32000]" = torch.ops.aten.sum.dim_IntList(view_57, [0], True);  view_57 = None
        view_58: "f32[32000]" = torch.ops.aten.reshape.default(sum_5, [32000]);  sum_5 = None
        view_59: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm, [32, 512, 768]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:318 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_133: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_59, primals_111);  primals_111 = None
        mul_134: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_133, 768)
        sum_6: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_133, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:316 in forward, code: hidden_states = self.dense(hidden_states)
        view_51: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [32, 512, 768]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_124: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
        pow_13: "f32[32, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_51, 3.0)
        mul_125: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_100: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_51, mul_125);  mul_125 = None
        mul_126: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_13: "f32[32, 512, 768]" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
        add_101: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_127: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_124, add_101)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:318 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_25: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_127, getitem_51);  mul_127 = getitem_51 = None
        mul_128: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        mul_135: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_133, mul_128);  mul_133 = None
        sum_7: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [2], True);  mul_135 = None
        mul_136: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_128, sum_7);  sum_7 = None
        sub_30: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_134, sum_6);  mul_134 = sum_6 = None
        sub_31: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_30, mul_136);  sub_30 = mul_136 = None
        div_2: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
        mul_137: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_31);  div_2 = sub_31 = None
        mul_138: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_59, mul_128);  mul_128 = None
        sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_138, [0, 1]);  mul_138 = None
        sum_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_59, [0, 1]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_139: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, mul_124);  mul_124 = None
        mul_140: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, add_101);  mul_137 = add_101 = None
        mul_141: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_32: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(1, mul_141);  mul_141 = None
        mul_142: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_139, sub_32);  mul_139 = sub_32 = None
        mul_143: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_142, 0.7978845608028654);  mul_142 = None
        mul_144: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_143, 0.044715)
        pow_14: "f32[32, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_51, 2.0);  view_51 = None
        mul_145: "f32[32, 512, 768]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_146: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        add_105: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_143, mul_146);  mul_143 = mul_146 = None
        mul_147: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_140, 0.5);  mul_140 = None
        add_106: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_105, mul_147);  add_105 = mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:316 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_106, [16384, 768]);  add_106 = None
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_32: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_2: "f32[16384, 768]" = torch.ops.aten.mm.default(view_60, permute_32);  permute_32 = None
        permute_33: "f32[768, 16384]" = torch.ops.aten.permute.default(view_60, [1, 0])
        mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_33, view_50);  permute_33 = view_50 = None
        sum_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_60, [0], True);  view_60 = None
        view_61: "f32[768]" = torch.ops.aten.reshape.default(sum_10, [768]);  sum_10 = None
        view_62: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_2, [32, 512, 768]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_149: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_62, primals_105);  primals_105 = None
        mul_150: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_149, 768)
        sum_11: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [2], True)
        mul_151: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_149, mul_122);  mul_149 = None
        sum_12: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [2], True);  mul_151 = None
        mul_152: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, sum_12);  sum_12 = None
        sub_34: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_150, sum_11);  mul_150 = sum_11 = None
        sub_35: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_34, mul_152);  sub_34 = mul_152 = None
        mul_153: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_35);  div_3 = sub_35 = None
        mul_154: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_62, mul_122);  mul_122 = None
        sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_154, [0, 1]);  mul_154 = None
        sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_62, [0, 1]);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_13: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_155: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.0);  convert_element_type_13 = None
        mul_156: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_153, mul_155);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_156, [16384, 768]);  mul_156 = None
        permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        permute_36: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_4: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_63, permute_36);  permute_36 = None
        permute_37: "f32[768, 16384]" = torch.ops.aten.permute.default(view_63, [1, 0])
        mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_37, view_48);  permute_37 = view_48 = None
        sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_63, [0], True);  view_63 = None
        view_64: "f32[768]" = torch.ops.aten.reshape.default(sum_15, [768]);  sum_15 = None
        view_65: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_4, [32, 512, 3072]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_47: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 3072]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_47, 0.5)
        mul_157: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_65, mul_116);  mul_116 = None
        pow_12: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_47, 3.0)
        mul_117: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_47, mul_117);  mul_117 = None
        mul_118: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_96: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_158: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_65, add_96);  view_65 = add_96 = None
        mul_159: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_36: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_159);  mul_159 = None
        mul_160: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_157, sub_36);  mul_157 = sub_36 = None
        mul_161: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_160, 0.7978845608028654);  mul_160 = None
        mul_162: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_161, 0.044715)
        pow_15: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_47, 2.0);  view_47 = None
        mul_163: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
        mul_164: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        add_107: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_161, mul_164);  mul_161 = mul_164 = None
        mul_165: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_158, 0.5);  mul_158 = None
        add_108: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_107, mul_165);  add_107 = mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_66: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_108, [16384, 3072]);  add_108 = None
        permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        permute_40: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_6: "f32[16384, 768]" = torch.ops.aten.mm.default(view_66, permute_40);  permute_40 = None
        permute_41: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_66, [1, 0])
        mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_41, view_46);  permute_41 = view_46 = None
        sum_16: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_66, [0], True);  view_66 = None
        view_67: "f32[3072]" = torch.ops.aten.reshape.default(sum_16, [3072]);  sum_16 = None
        view_68: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_6, [32, 512, 768]);  mm_6 = None
        add_109: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_153, view_68);  mul_153 = view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_167: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_109, primals_99);  primals_99 = None
        mul_168: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_167, 768)
        sum_17: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True)
        mul_169: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_167, mul_114);  mul_167 = None
        sum_18: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True);  mul_169 = None
        mul_170: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_114, sum_18);  sum_18 = None
        sub_38: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_168, sum_17);  mul_168 = sum_17 = None
        sub_39: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_170);  sub_38 = mul_170 = None
        mul_171: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_39);  div_4 = sub_39 = None
        mul_172: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_109, mul_114);  mul_114 = None
        sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1]);  mul_172 = None
        sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_109, [0, 1]);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        full_default_5: "f32[32, 512, 768, 2]" = torch.ops.aten.full.default([32, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_171, 3, 0)
        view_as_complex: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter);  select_scatter = None
        _fft_c2c_12: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex, [1, 2], 0, False);  view_as_complex = None
        view_as_real_12: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_12);  _fft_c2c_12 = None
        select_13: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_12, 3, 0);  view_as_real_12 = None
        add_110: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_171, select_13);  mul_171 = select_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_174: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, primals_97);  primals_97 = None
        mul_175: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_174, 768)
        sum_21: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_174, [2], True)
        mul_176: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_174, mul_112);  mul_174 = None
        sum_22: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True);  mul_176 = None
        mul_177: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_112, sum_22);  sum_22 = None
        sub_41: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_175, sum_21);  mul_175 = sum_21 = None
        sub_42: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_177);  sub_41 = mul_177 = None
        mul_178: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_42);  div_5 = sub_42 = None
        mul_179: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, mul_112);  mul_112 = None
        sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_179, [0, 1]);  mul_179 = None
        sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_180: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.0);  convert_element_type_14 = None
        mul_181: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_178, mul_180);  mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_69: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_181, [16384, 768]);  mul_181 = None
        permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        permute_44: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_8: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_69, permute_44);  permute_44 = None
        permute_45: "f32[768, 16384]" = torch.ops.aten.permute.default(view_69, [1, 0])
        mm_9: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_45, view_44);  permute_45 = view_44 = None
        sum_25: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_69, [0], True);  view_69 = None
        view_70: "f32[768]" = torch.ops.aten.reshape.default(sum_25, [768]);  sum_25 = None
        view_71: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_8, [32, 512, 3072]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_43: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_21, [32, 512, 3072]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_106: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
        mul_182: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_71, mul_106);  mul_106 = None
        pow_11: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
        mul_107: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_43, mul_107);  mul_107 = None
        mul_108: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_88: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_183: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_71, add_88);  view_71 = add_88 = None
        mul_184: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_43: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_184);  mul_184 = None
        mul_185: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_182, sub_43);  mul_182 = sub_43 = None
        mul_186: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_185, 0.7978845608028654);  mul_185 = None
        mul_187: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_186, 0.044715)
        pow_16: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 2.0);  view_43 = None
        mul_188: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_189: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
        add_111: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_186, mul_189);  mul_186 = mul_189 = None
        mul_190: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_183, 0.5);  mul_183 = None
        add_112: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_111, mul_190);  add_111 = mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_72: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_112, [16384, 3072]);  add_112 = None
        permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_48: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_10: "f32[16384, 768]" = torch.ops.aten.mm.default(view_72, permute_48);  permute_48 = None
        permute_49: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_72, [1, 0])
        mm_11: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_49, view_42);  permute_49 = view_42 = None
        sum_26: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_72, [0], True);  view_72 = None
        view_73: "f32[3072]" = torch.ops.aten.reshape.default(sum_26, [3072]);  sum_26 = None
        view_74: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_10, [32, 512, 768]);  mm_10 = None
        add_113: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_178, view_74);  mul_178 = view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_192: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_91);  primals_91 = None
        mul_193: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_192, 768)
        sum_27: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
        mul_194: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_104);  mul_192 = None
        sum_28: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
        mul_195: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_104, sum_28);  sum_28 = None
        sub_45: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_193, sum_27);  mul_193 = sum_27 = None
        sub_46: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_195);  sub_45 = mul_195 = None
        mul_196: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_46);  div_6 = sub_46 = None
        mul_197: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, mul_104);  mul_104 = None
        sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
        sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_1: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_196, 3, 0)
        view_as_complex_1: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_1);  select_scatter_1 = None
        _fft_c2c_13: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_1, [1, 2], 0, False);  view_as_complex_1 = None
        view_as_real_13: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_13);  _fft_c2c_13 = None
        select_14: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_13, 3, 0);  view_as_real_13 = None
        add_114: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_196, select_14);  mul_196 = select_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_199: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, primals_89);  primals_89 = None
        mul_200: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_199, 768)
        sum_31: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True)
        mul_201: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_199, mul_102);  mul_199 = None
        sum_32: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True);  mul_201 = None
        mul_202: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, sum_32);  sum_32 = None
        sub_48: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_200, sum_31);  mul_200 = sum_31 = None
        sub_49: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_202);  sub_48 = mul_202 = None
        mul_203: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_49);  div_7 = sub_49 = None
        mul_204: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, mul_102);  mul_102 = None
        sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 1]);  mul_204 = None
        sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_15: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_205: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.0);  convert_element_type_15 = None
        mul_206: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_203, mul_205);  mul_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_75: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_206, [16384, 768]);  mul_206 = None
        permute_20: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_52: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_12: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_75, permute_52);  permute_52 = None
        permute_53: "f32[768, 16384]" = torch.ops.aten.permute.default(view_75, [1, 0])
        mm_13: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_53, view_40);  permute_53 = view_40 = None
        sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True);  view_75 = None
        view_76: "f32[768]" = torch.ops.aten.reshape.default(sum_35, [768]);  sum_35 = None
        view_77: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_12, [32, 512, 3072]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_39: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 3072]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_96: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_39, 0.5)
        mul_207: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_77, mul_96);  mul_96 = None
        pow_10: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_39, 3.0)
        mul_97: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_39, mul_97);  mul_97 = None
        mul_98: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_80: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_208: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_77, add_80);  view_77 = add_80 = None
        mul_209: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_50: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_209);  mul_209 = None
        mul_210: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_207, sub_50);  mul_207 = sub_50 = None
        mul_211: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_210, 0.7978845608028654);  mul_210 = None
        mul_212: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_211, 0.044715)
        pow_17: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_39, 2.0);  view_39 = None
        mul_213: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
        mul_214: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
        add_115: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_211, mul_214);  mul_211 = mul_214 = None
        mul_215: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_208, 0.5);  mul_208 = None
        add_116: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_115, mul_215);  add_115 = mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_78: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_116, [16384, 3072]);  add_116 = None
        permute_19: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_14: "f32[16384, 768]" = torch.ops.aten.mm.default(view_78, permute_56);  permute_56 = None
        permute_57: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_78, [1, 0])
        mm_15: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_57, view_38);  permute_57 = view_38 = None
        sum_36: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_78, [0], True);  view_78 = None
        view_79: "f32[3072]" = torch.ops.aten.reshape.default(sum_36, [3072]);  sum_36 = None
        view_80: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_14, [32, 512, 768]);  mm_14 = None
        add_117: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_203, view_80);  mul_203 = view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_217: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, primals_83);  primals_83 = None
        mul_218: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_217, 768)
        sum_37: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [2], True)
        mul_219: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_217, mul_94);  mul_217 = None
        sum_38: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [2], True);  mul_219 = None
        mul_220: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_94, sum_38);  sum_38 = None
        sub_52: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_218, sum_37);  mul_218 = sum_37 = None
        sub_53: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_220);  sub_52 = mul_220 = None
        mul_221: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_53);  div_8 = sub_53 = None
        mul_222: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, mul_94);  mul_94 = None
        sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_222, [0, 1]);  mul_222 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_2: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_221, 3, 0)
        view_as_complex_2: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_2);  select_scatter_2 = None
        _fft_c2c_14: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_2, [1, 2], 0, False);  view_as_complex_2 = None
        view_as_real_14: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_14);  _fft_c2c_14 = None
        select_15: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_14, 3, 0);  view_as_real_14 = None
        add_118: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_221, select_15);  mul_221 = select_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_224: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_118, primals_81);  primals_81 = None
        mul_225: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_224, 768)
        sum_41: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
        mul_226: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_224, mul_92);  mul_224 = None
        sum_42: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
        mul_227: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, sum_42);  sum_42 = None
        sub_55: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_225, sum_41);  mul_225 = sum_41 = None
        sub_56: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_227);  sub_55 = mul_227 = None
        mul_228: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_56);  div_9 = sub_56 = None
        mul_229: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_118, mul_92);  mul_92 = None
        sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
        sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_118, [0, 1]);  add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_16: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_230: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.0);  convert_element_type_16 = None
        mul_231: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_228, mul_230);  mul_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_81: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_231, [16384, 768]);  mul_231 = None
        permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        permute_60: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        mm_16: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_81, permute_60);  permute_60 = None
        permute_61: "f32[768, 16384]" = torch.ops.aten.permute.default(view_81, [1, 0])
        mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_61, view_36);  permute_61 = view_36 = None
        sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_81, [0], True);  view_81 = None
        view_82: "f32[768]" = torch.ops.aten.reshape.default(sum_45, [768]);  sum_45 = None
        view_83: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_16, [32, 512, 3072]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_35: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_17, [32, 512, 3072]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_86: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
        mul_232: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_83, mul_86);  mul_86 = None
        pow_9: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
        mul_87: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_87);  mul_87 = None
        mul_88: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_72: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_233: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_83, add_72);  view_83 = add_72 = None
        mul_234: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_57: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_234);  mul_234 = None
        mul_235: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_232, sub_57);  mul_232 = sub_57 = None
        mul_236: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_235, 0.7978845608028654);  mul_235 = None
        mul_237: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_236, 0.044715)
        pow_18: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 2.0);  view_35 = None
        mul_238: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_239: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
        add_119: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_236, mul_239);  mul_236 = mul_239 = None
        mul_240: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_233, 0.5);  mul_233 = None
        add_120: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_119, mul_240);  add_119 = mul_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_120, [16384, 3072]);  add_120 = None
        permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_64: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        mm_18: "f32[16384, 768]" = torch.ops.aten.mm.default(view_84, permute_64);  permute_64 = None
        permute_65: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_84, [1, 0])
        mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_65, view_34);  permute_65 = view_34 = None
        sum_46: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_84, [0], True);  view_84 = None
        view_85: "f32[3072]" = torch.ops.aten.reshape.default(sum_46, [3072]);  sum_46 = None
        view_86: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_18, [32, 512, 768]);  mm_18 = None
        add_121: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_228, view_86);  mul_228 = view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_242: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_121, primals_75);  primals_75 = None
        mul_243: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_242, 768)
        sum_47: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_242, [2], True)
        mul_244: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_242, mul_84);  mul_242 = None
        sum_48: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [2], True);  mul_244 = None
        mul_245: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, sum_48);  sum_48 = None
        sub_59: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_243, sum_47);  mul_243 = sum_47 = None
        sub_60: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_245);  sub_59 = mul_245 = None
        mul_246: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_60);  div_10 = sub_60 = None
        mul_247: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_121, mul_84);  mul_84 = None
        sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_247, [0, 1]);  mul_247 = None
        sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_121, [0, 1]);  add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_3: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_246, 3, 0)
        view_as_complex_3: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_3);  select_scatter_3 = None
        _fft_c2c_15: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_3, [1, 2], 0, False);  view_as_complex_3 = None
        view_as_real_15: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_15);  _fft_c2c_15 = None
        select_16: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_15, 3, 0);  view_as_real_15 = None
        add_122: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_246, select_16);  mul_246 = select_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_249: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, primals_73);  primals_73 = None
        mul_250: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_249, 768)
        sum_51: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2], True)
        mul_251: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_249, mul_82);  mul_249 = None
        sum_52: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True);  mul_251 = None
        mul_252: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_82, sum_52);  sum_52 = None
        sub_62: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_250, sum_51);  mul_250 = sum_51 = None
        sub_63: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_252);  sub_62 = mul_252 = None
        mul_253: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_63);  div_11 = sub_63 = None
        mul_254: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, mul_82);  mul_82 = None
        sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 1]);  mul_254 = None
        sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_17: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_255: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.0);  convert_element_type_17 = None
        mul_256: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, mul_255);  mul_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_87: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_256, [16384, 768]);  mul_256 = None
        permute_16: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        permute_68: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_20: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_87, permute_68);  permute_68 = None
        permute_69: "f32[768, 16384]" = torch.ops.aten.permute.default(view_87, [1, 0])
        mm_21: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_69, view_32);  permute_69 = view_32 = None
        sum_55: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_87, [0], True);  view_87 = None
        view_88: "f32[768]" = torch.ops.aten.reshape.default(sum_55, [768]);  sum_55 = None
        view_89: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_20, [32, 512, 3072]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_31: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_15, [32, 512, 3072]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
        mul_257: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_89, mul_76);  mul_76 = None
        pow_8: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_31, 3.0)
        mul_77: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_31, mul_77);  mul_77 = None
        mul_78: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_64: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_258: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_89, add_64);  view_89 = add_64 = None
        mul_259: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_64: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_259);  mul_259 = None
        mul_260: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_257, sub_64);  mul_257 = sub_64 = None
        mul_261: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_260, 0.7978845608028654);  mul_260 = None
        mul_262: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_261, 0.044715)
        pow_19: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_31, 2.0);  view_31 = None
        mul_263: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
        mul_264: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
        add_123: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_261, mul_264);  mul_261 = mul_264 = None
        mul_265: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_258, 0.5);  mul_258 = None
        add_124: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_123, mul_265);  add_123 = mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_90: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_124, [16384, 3072]);  add_124 = None
        permute_15: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        permute_72: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_22: "f32[16384, 768]" = torch.ops.aten.mm.default(view_90, permute_72);  permute_72 = None
        permute_73: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_90, [1, 0])
        mm_23: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_73, view_30);  permute_73 = view_30 = None
        sum_56: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_90, [0], True);  view_90 = None
        view_91: "f32[3072]" = torch.ops.aten.reshape.default(sum_56, [3072]);  sum_56 = None
        view_92: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_22, [32, 512, 768]);  mm_22 = None
        add_125: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_253, view_92);  mul_253 = view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_267: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_67);  primals_67 = None
        mul_268: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_267, 768)
        sum_57: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
        mul_269: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_267, mul_74);  mul_267 = None
        sum_58: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
        mul_270: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, sum_58);  sum_58 = None
        sub_66: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_268, sum_57);  mul_268 = sum_57 = None
        sub_67: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_270);  sub_66 = mul_270 = None
        mul_271: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_67);  div_12 = sub_67 = None
        mul_272: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_74);  mul_74 = None
        sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
        sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_4: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_271, 3, 0)
        view_as_complex_4: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_4);  select_scatter_4 = None
        _fft_c2c_16: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_4, [1, 2], 0, False);  view_as_complex_4 = None
        view_as_real_16: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_16);  _fft_c2c_16 = None
        select_17: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_16, 3, 0);  view_as_real_16 = None
        add_126: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_271, select_17);  mul_271 = select_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_274: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, primals_65);  primals_65 = None
        mul_275: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_274, 768)
        sum_61: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True)
        mul_276: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_274, mul_72);  mul_274 = None
        sum_62: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True);  mul_276 = None
        mul_277: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, sum_62);  sum_62 = None
        sub_69: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_275, sum_61);  mul_275 = sum_61 = None
        sub_70: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_277);  sub_69 = mul_277 = None
        mul_278: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_70);  div_13 = sub_70 = None
        mul_279: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, mul_72);  mul_72 = None
        sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 1]);  mul_279 = None
        sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_18: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_280: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.0);  convert_element_type_18 = None
        mul_281: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_278, mul_280);  mul_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_93: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_281, [16384, 768]);  mul_281 = None
        permute_14: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        mm_24: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_93, permute_76);  permute_76 = None
        permute_77: "f32[768, 16384]" = torch.ops.aten.permute.default(view_93, [1, 0])
        mm_25: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_77, view_28);  permute_77 = view_28 = None
        sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_93, [0], True);  view_93 = None
        view_94: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
        view_95: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_24, [32, 512, 3072]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_27: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_13, [32, 512, 3072]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_66: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_27, 0.5)
        mul_282: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_95, mul_66);  mul_66 = None
        pow_7: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 3.0)
        mul_67: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_27, mul_67);  mul_67 = None
        mul_68: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_56: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_283: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_95, add_56);  view_95 = add_56 = None
        mul_284: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_71: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_284);  mul_284 = None
        mul_285: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_282, sub_71);  mul_282 = sub_71 = None
        mul_286: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_285, 0.7978845608028654);  mul_285 = None
        mul_287: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_286, 0.044715)
        pow_20: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 2.0);  view_27 = None
        mul_288: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_289: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
        add_127: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_286, mul_289);  mul_286 = mul_289 = None
        mul_290: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_283, 0.5);  mul_283 = None
        add_128: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_127, mul_290);  add_127 = mul_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_96: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_128, [16384, 3072]);  add_128 = None
        permute_13: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_26: "f32[16384, 768]" = torch.ops.aten.mm.default(view_96, permute_80);  permute_80 = None
        permute_81: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_96, [1, 0])
        mm_27: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_81, view_26);  permute_81 = view_26 = None
        sum_66: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_96, [0], True);  view_96 = None
        view_97: "f32[3072]" = torch.ops.aten.reshape.default(sum_66, [3072]);  sum_66 = None
        view_98: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_26, [32, 512, 768]);  mm_26 = None
        add_129: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_278, view_98);  mul_278 = view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_292: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, primals_59);  primals_59 = None
        mul_293: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_292, 768)
        sum_67: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True)
        mul_294: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_292, mul_64);  mul_292 = None
        sum_68: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True);  mul_294 = None
        mul_295: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, sum_68);  sum_68 = None
        sub_73: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_293, sum_67);  mul_293 = sum_67 = None
        sub_74: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_73, mul_295);  sub_73 = mul_295 = None
        mul_296: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_74);  div_14 = sub_74 = None
        mul_297: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, mul_64);  mul_64 = None
        sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_5: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_296, 3, 0)
        view_as_complex_5: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_5);  select_scatter_5 = None
        _fft_c2c_17: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_5, [1, 2], 0, False);  view_as_complex_5 = None
        view_as_real_17: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_17);  _fft_c2c_17 = None
        select_18: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_17, 3, 0);  view_as_real_17 = None
        add_130: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_296, select_18);  mul_296 = select_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_299: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, primals_57);  primals_57 = None
        mul_300: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_299, 768)
        sum_71: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True)
        mul_301: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_299, mul_62);  mul_299 = None
        sum_72: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [2], True);  mul_301 = None
        mul_302: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_62, sum_72);  sum_72 = None
        sub_76: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_300, sum_71);  mul_300 = sum_71 = None
        sub_77: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_302);  sub_76 = mul_302 = None
        mul_303: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_77);  div_15 = sub_77 = None
        mul_304: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, mul_62);  mul_62 = None
        sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1]);  mul_304 = None
        sum_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_130, [0, 1]);  add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_19: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_305: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.0);  convert_element_type_19 = None
        mul_306: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_303, mul_305);  mul_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_99: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_306, [16384, 768]);  mul_306 = None
        permute_12: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_84: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_28: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_99, permute_84);  permute_84 = None
        permute_85: "f32[768, 16384]" = torch.ops.aten.permute.default(view_99, [1, 0])
        mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_85, view_24);  permute_85 = view_24 = None
        sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_99, [0], True);  view_99 = None
        view_100: "f32[768]" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None
        view_101: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_28, [32, 512, 3072]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_23: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_11, [32, 512, 3072]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
        mul_307: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_101, mul_56);  mul_56 = None
        pow_6: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 3.0)
        mul_57: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_23, mul_57);  mul_57 = None
        mul_58: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_48: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_308: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_101, add_48);  view_101 = add_48 = None
        mul_309: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_78: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_309);  mul_309 = None
        mul_310: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_307, sub_78);  mul_307 = sub_78 = None
        mul_311: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_310, 0.7978845608028654);  mul_310 = None
        mul_312: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_311, 0.044715)
        pow_21: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 2.0);  view_23 = None
        mul_313: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
        mul_314: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        add_131: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_311, mul_314);  mul_311 = mul_314 = None
        mul_315: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_308, 0.5);  mul_308 = None
        add_132: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_131, mul_315);  add_131 = mul_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_102: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_132, [16384, 3072]);  add_132 = None
        permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_30: "f32[16384, 768]" = torch.ops.aten.mm.default(view_102, permute_88);  permute_88 = None
        permute_89: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_102, [1, 0])
        mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_89, view_22);  permute_89 = view_22 = None
        sum_76: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_102, [0], True);  view_102 = None
        view_103: "f32[3072]" = torch.ops.aten.reshape.default(sum_76, [3072]);  sum_76 = None
        view_104: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_30, [32, 512, 768]);  mm_30 = None
        add_133: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_303, view_104);  mul_303 = view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_317: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, primals_51);  primals_51 = None
        mul_318: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_317, 768)
        sum_77: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True)
        mul_319: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_317, mul_54);  mul_317 = None
        sum_78: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True);  mul_319 = None
        mul_320: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_54, sum_78);  sum_78 = None
        sub_80: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_318, sum_77);  mul_318 = sum_77 = None
        sub_81: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_320);  sub_80 = mul_320 = None
        mul_321: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_81);  div_16 = sub_81 = None
        mul_322: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, mul_54);  mul_54 = None
        sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_322, [0, 1]);  mul_322 = None
        sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1]);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_6: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_321, 3, 0)
        view_as_complex_6: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_6);  select_scatter_6 = None
        _fft_c2c_18: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_6, [1, 2], 0, False);  view_as_complex_6 = None
        view_as_real_18: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_18);  _fft_c2c_18 = None
        select_19: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_18, 3, 0);  view_as_real_18 = None
        add_134: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_321, select_19);  mul_321 = select_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_324: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, primals_49);  primals_49 = None
        mul_325: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_324, 768)
        sum_81: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [2], True)
        mul_326: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_324, mul_52);  mul_324 = None
        sum_82: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True);  mul_326 = None
        mul_327: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, sum_82);  sum_82 = None
        sub_83: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_325, sum_81);  mul_325 = sum_81 = None
        sub_84: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_327);  sub_83 = mul_327 = None
        mul_328: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_84);  div_17 = sub_84 = None
        mul_329: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_52);  mul_52 = None
        sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 1]);  mul_329 = None
        sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_330: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.0);  convert_element_type_20 = None
        mul_331: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_328, mul_330);  mul_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_105: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_331, [16384, 768]);  mul_331 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
        permute_92: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_32: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_105, permute_92);  permute_92 = None
        permute_93: "f32[768, 16384]" = torch.ops.aten.permute.default(view_105, [1, 0])
        mm_33: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_93, view_20);  permute_93 = view_20 = None
        sum_85: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True);  view_105 = None
        view_106: "f32[768]" = torch.ops.aten.reshape.default(sum_85, [768]);  sum_85 = None
        view_107: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_32, [32, 512, 3072]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 3072]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_332: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_46);  mul_46 = None
        pow_5: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 3.0)
        mul_47: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_19, mul_47);  mul_47 = None
        mul_48: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_40: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_333: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, add_40);  view_107 = add_40 = None
        mul_334: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_85: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_334);  mul_334 = None
        mul_335: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_332, sub_85);  mul_332 = sub_85 = None
        mul_336: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_335, 0.7978845608028654);  mul_335 = None
        mul_337: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_336, 0.044715)
        pow_22: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 2.0);  view_19 = None
        mul_338: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_339: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        add_135: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_336, mul_339);  mul_336 = mul_339 = None
        mul_340: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_333, 0.5);  mul_333 = None
        add_136: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_135, mul_340);  add_135 = mul_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_136, [16384, 3072]);  add_136 = None
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_96: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_34: "f32[16384, 768]" = torch.ops.aten.mm.default(view_108, permute_96);  permute_96 = None
        permute_97: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_108, [1, 0])
        mm_35: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_97, view_18);  permute_97 = view_18 = None
        sum_86: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_108, [0], True);  view_108 = None
        view_109: "f32[3072]" = torch.ops.aten.reshape.default(sum_86, [3072]);  sum_86 = None
        view_110: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_34, [32, 512, 768]);  mm_34 = None
        add_137: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_328, view_110);  mul_328 = view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_342: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, primals_43);  primals_43 = None
        mul_343: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_342, 768)
        sum_87: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
        mul_344: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_342, mul_44);  mul_342 = None
        sum_88: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
        mul_345: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, sum_88);  sum_88 = None
        sub_87: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_343, sum_87);  mul_343 = sum_87 = None
        sub_88: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_345);  sub_87 = mul_345 = None
        mul_346: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_88);  div_18 = sub_88 = None
        mul_347: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, mul_44);  mul_44 = None
        sum_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
        sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_7: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_346, 3, 0)
        view_as_complex_7: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_7);  select_scatter_7 = None
        _fft_c2c_19: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_7, [1, 2], 0, False);  view_as_complex_7 = None
        view_as_real_19: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_19);  _fft_c2c_19 = None
        select_20: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_19, 3, 0);  view_as_real_19 = None
        add_138: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_346, select_20);  mul_346 = select_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_349: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, primals_41);  primals_41 = None
        mul_350: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_349, 768)
        sum_91: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True)
        mul_351: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_349, mul_42);  mul_349 = None
        sum_92: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [2], True);  mul_351 = None
        mul_352: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_92);  sum_92 = None
        sub_90: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_350, sum_91);  mul_350 = sum_91 = None
        sub_91: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_352);  sub_90 = mul_352 = None
        mul_353: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_91);  div_19 = sub_91 = None
        mul_354: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_42);  mul_42 = None
        sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_354, [0, 1]);  mul_354 = None
        sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_21: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_355: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.0);  convert_element_type_21 = None
        mul_356: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_353, mul_355);  mul_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_111: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_356, [16384, 768]);  mul_356 = None
        permute_8: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_100: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_36: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_111, permute_100);  permute_100 = None
        permute_101: "f32[768, 16384]" = torch.ops.aten.permute.default(view_111, [1, 0])
        mm_37: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_101, view_16);  permute_101 = view_16 = None
        sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_111, [0], True);  view_111 = None
        view_112: "f32[768]" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
        view_113: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_36, [32, 512, 3072]);  mm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_15: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_7, [32, 512, 3072]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
        mul_357: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_113, mul_36);  mul_36 = None
        pow_4: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_15, 3.0)
        mul_37: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_15, mul_37);  mul_37 = None
        mul_38: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_32: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_358: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_113, add_32);  view_113 = add_32 = None
        mul_359: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_92: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_359);  mul_359 = None
        mul_360: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_357, sub_92);  mul_357 = sub_92 = None
        mul_361: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_360, 0.7978845608028654);  mul_360 = None
        mul_362: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_361, 0.044715)
        pow_23: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_15, 2.0);  view_15 = None
        mul_363: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
        mul_364: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_362, mul_363);  mul_362 = mul_363 = None
        add_139: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_361, mul_364);  mul_361 = mul_364 = None
        mul_365: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_358, 0.5);  mul_358 = None
        add_140: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_139, mul_365);  add_139 = mul_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_114: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_140, [16384, 3072]);  add_140 = None
        permute_7: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_104: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_38: "f32[16384, 768]" = torch.ops.aten.mm.default(view_114, permute_104);  permute_104 = None
        permute_105: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_114, [1, 0])
        mm_39: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_105, view_14);  permute_105 = view_14 = None
        sum_96: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True);  view_114 = None
        view_115: "f32[3072]" = torch.ops.aten.reshape.default(sum_96, [3072]);  sum_96 = None
        view_116: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_38, [32, 512, 768]);  mm_38 = None
        add_141: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_353, view_116);  mul_353 = view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_367: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, primals_35);  primals_35 = None
        mul_368: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_367, 768)
        sum_97: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
        mul_369: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_367, mul_34);  mul_367 = None
        sum_98: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
        mul_370: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, sum_98);  sum_98 = None
        sub_94: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_368, sum_97);  mul_368 = sum_97 = None
        sub_95: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_94, mul_370);  sub_94 = mul_370 = None
        mul_371: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_95);  div_20 = sub_95 = None
        mul_372: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, mul_34);  mul_34 = None
        sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
        sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_8: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_371, 3, 0)
        view_as_complex_8: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_8);  select_scatter_8 = None
        _fft_c2c_20: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_8, [1, 2], 0, False);  view_as_complex_8 = None
        view_as_real_20: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_20);  _fft_c2c_20 = None
        select_21: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_20, 3, 0);  view_as_real_20 = None
        add_142: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_371, select_21);  mul_371 = select_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_374: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, primals_33);  primals_33 = None
        mul_375: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_374, 768)
        sum_101: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True)
        mul_376: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_374, mul_32);  mul_374 = None
        sum_102: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [2], True);  mul_376 = None
        mul_377: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, sum_102);  sum_102 = None
        sub_97: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_375, sum_101);  mul_375 = sum_101 = None
        sub_98: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_97, mul_377);  sub_97 = mul_377 = None
        mul_378: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_98);  div_21 = sub_98 = None
        mul_379: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, mul_32);  mul_32 = None
        sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 1]);  mul_379 = None
        sum_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1]);  add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_22: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_380: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.0);  convert_element_type_22 = None
        mul_381: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_378, mul_380);  mul_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_117: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_381, [16384, 768]);  mul_381 = None
        permute_6: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        mm_40: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_117, permute_108);  permute_108 = None
        permute_109: "f32[768, 16384]" = torch.ops.aten.permute.default(view_117, [1, 0])
        mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_109, view_12);  permute_109 = view_12 = None
        sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True);  view_117 = None
        view_118: "f32[768]" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None
        view_119: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_40, [32, 512, 3072]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_11: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 3072]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_26: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_11, 0.5)
        mul_382: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_119, mul_26);  mul_26 = None
        pow_3: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_11, 3.0)
        mul_27: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_11, mul_27);  mul_27 = None
        mul_28: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_24: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_383: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_119, add_24);  view_119 = add_24 = None
        mul_384: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_99: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_384);  mul_384 = None
        mul_385: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_382, sub_99);  mul_382 = sub_99 = None
        mul_386: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_385, 0.7978845608028654);  mul_385 = None
        mul_387: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_386, 0.044715)
        pow_24: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_11, 2.0);  view_11 = None
        mul_388: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_389: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_387, mul_388);  mul_387 = mul_388 = None
        add_143: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_386, mul_389);  mul_386 = mul_389 = None
        mul_390: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_383, 0.5);  mul_383 = None
        add_144: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_143, mul_390);  add_143 = mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_120: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_144, [16384, 3072]);  add_144 = None
        permute_5: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_112: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_42: "f32[16384, 768]" = torch.ops.aten.mm.default(view_120, permute_112);  permute_112 = None
        permute_113: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_120, [1, 0])
        mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_113, view_10);  permute_113 = view_10 = None
        sum_106: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True);  view_120 = None
        view_121: "f32[3072]" = torch.ops.aten.reshape.default(sum_106, [3072]);  sum_106 = None
        view_122: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_42, [32, 512, 768]);  mm_42 = None
        add_145: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_378, view_122);  mul_378 = view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_392: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, primals_27);  primals_27 = None
        mul_393: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_392, 768)
        sum_107: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [2], True)
        mul_394: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_392, mul_24);  mul_392 = None
        sum_108: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [2], True);  mul_394 = None
        mul_395: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_108);  sum_108 = None
        sub_101: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_393, sum_107);  mul_393 = sum_107 = None
        sub_102: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_395);  sub_101 = mul_395 = None
        mul_396: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_102);  div_22 = sub_102 = None
        mul_397: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, mul_24);  mul_24 = None
        sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 1]);  mul_397 = None
        sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_145, [0, 1]);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_9: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_396, 3, 0)
        view_as_complex_9: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_9);  select_scatter_9 = None
        _fft_c2c_21: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_9, [1, 2], 0, False);  view_as_complex_9 = None
        view_as_real_21: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_21);  _fft_c2c_21 = None
        select_22: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_21, 3, 0);  view_as_real_21 = None
        add_146: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_396, select_22);  mul_396 = select_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_399: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_25);  primals_25 = None
        mul_400: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_399, 768)
        sum_111: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True)
        mul_401: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_399, mul_22);  mul_399 = None
        sum_112: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [2], True);  mul_401 = None
        mul_402: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, sum_112);  sum_112 = None
        sub_104: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_400, sum_111);  mul_400 = sum_111 = None
        sub_105: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_402);  sub_104 = mul_402 = None
        mul_403: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_105);  div_23 = sub_105 = None
        mul_404: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_22);  mul_22 = None
        sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 1]);  mul_404 = None
        sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_23: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_405: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.0);  convert_element_type_23 = None
        mul_406: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_403, mul_405);  mul_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_123: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_406, [16384, 768]);  mul_406 = None
        permute_4: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_44: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_123, permute_116);  permute_116 = None
        permute_117: "f32[768, 16384]" = torch.ops.aten.permute.default(view_123, [1, 0])
        mm_45: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_117, view_8);  permute_117 = view_8 = None
        sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_123, [0], True);  view_123 = None
        view_124: "f32[768]" = torch.ops.aten.reshape.default(sum_115, [768]);  sum_115 = None
        view_125: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_44, [32, 512, 3072]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_7: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_3, [32, 512, 3072]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_16: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_7, 0.5)
        mul_407: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_125, mul_16);  mul_16 = None
        pow_2: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_7, 3.0)
        mul_17: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_7, mul_17);  mul_17 = None
        mul_18: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_16: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_408: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_125, add_16);  view_125 = add_16 = None
        mul_409: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_106: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_409);  mul_409 = None
        mul_410: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_407, sub_106);  mul_407 = sub_106 = None
        mul_411: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_410, 0.7978845608028654);  mul_410 = None
        mul_412: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_411, 0.044715)
        pow_25: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_7, 2.0);  view_7 = None
        mul_413: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_25, 3.0);  pow_25 = None
        mul_414: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_412, mul_413);  mul_412 = mul_413 = None
        add_147: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_411, mul_414);  mul_411 = mul_414 = None
        mul_415: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_408, 0.5);  mul_408 = None
        add_148: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_147, mul_415);  add_147 = mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_148, [16384, 3072]);  add_148 = None
        permute_3: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_46: "f32[16384, 768]" = torch.ops.aten.mm.default(view_126, permute_120);  permute_120 = None
        permute_121: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_126, [1, 0])
        mm_47: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_121, view_6);  permute_121 = view_6 = None
        sum_116: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
        view_127: "f32[3072]" = torch.ops.aten.reshape.default(sum_116, [3072]);  sum_116 = None
        view_128: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_46, [32, 512, 768]);  mm_46 = None
        add_149: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_403, view_128);  mul_403 = view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_417: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, primals_19);  primals_19 = None
        mul_418: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_417, 768)
        sum_117: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [2], True)
        mul_419: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_417, mul_14);  mul_417 = None
        sum_118: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [2], True);  mul_419 = None
        mul_420: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_14, sum_118);  sum_118 = None
        sub_108: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_418, sum_117);  mul_418 = sum_117 = None
        sub_109: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_420);  sub_108 = mul_420 = None
        mul_421: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_109);  div_24 = sub_109 = None
        mul_422: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, mul_14);  mul_14 = None
        sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_422, [0, 1]);  mul_422 = None
        sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_10: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_421, 3, 0)
        view_as_complex_10: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_10);  select_scatter_10 = None
        _fft_c2c_22: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_10, [1, 2], 0, False);  view_as_complex_10 = None
        view_as_real_22: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_22);  _fft_c2c_22 = None
        select_23: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_22, 3, 0);  view_as_real_22 = None
        add_150: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_421, select_23);  mul_421 = select_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_424: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, primals_17);  primals_17 = None
        mul_425: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_424, 768)
        sum_121: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_424, [2], True)
        mul_426: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_424, mul_12);  mul_424 = None
        sum_122: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [2], True);  mul_426 = None
        mul_427: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, sum_122);  sum_122 = None
        sub_111: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_425, sum_121);  mul_425 = sum_121 = None
        sub_112: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_427);  sub_111 = mul_427 = None
        mul_428: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_112);  div_25 = sub_112 = None
        mul_429: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, mul_12);  mul_12 = None
        sum_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 1]);  mul_429 = None
        sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_24: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_430: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.0);  convert_element_type_24 = None
        mul_431: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_428, mul_430);  mul_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_431, [16384, 768]);  mul_431 = None
        permute_2: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_124: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_48: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_129, permute_124);  permute_124 = None
        permute_125: "f32[768, 16384]" = torch.ops.aten.permute.default(view_129, [1, 0])
        mm_49: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_125, view_4);  permute_125 = view_4 = None
        sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
        view_130: "f32[768]" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
        view_131: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_48, [32, 512, 3072]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_3: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_1, [32, 512, 3072]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_3, 0.5)
        mul_432: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_131, mul_6);  mul_6 = None
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_3, 3.0)
        mul_7: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_3, mul_7);  mul_7 = None
        mul_8: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_8: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_433: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_131, add_8);  view_131 = add_8 = None
        mul_434: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_113: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_434);  mul_434 = None
        mul_435: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_432, sub_113);  mul_432 = sub_113 = None
        mul_436: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_435, 0.7978845608028654);  mul_435 = None
        mul_437: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_436, 0.044715)
        pow_26: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_3, 2.0);  view_3 = None
        mul_438: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
        mul_439: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
        add_151: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_436, mul_439);  mul_436 = mul_439 = None
        mul_440: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_433, 0.5);  mul_433 = None
        add_152: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_151, mul_440);  add_151 = mul_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_132: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_152, [16384, 3072]);  add_152 = None
        permute_1: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_128: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_50: "f32[16384, 768]" = torch.ops.aten.mm.default(view_132, permute_128);  permute_128 = None
        permute_129: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_132, [1, 0])
        mm_51: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_129, view_2);  permute_129 = view_2 = None
        sum_126: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_132, [0], True);  view_132 = None
        view_133: "f32[3072]" = torch.ops.aten.reshape.default(sum_126, [3072]);  sum_126 = None
        view_134: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_50, [32, 512, 768]);  mm_50 = None
        add_153: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_428, view_134);  mul_428 = view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_442: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, primals_11);  primals_11 = None
        mul_443: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_442, 768)
        sum_127: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True)
        mul_444: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_442, mul_4);  mul_442 = None
        sum_128: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [2], True);  mul_444 = None
        mul_445: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_128);  sum_128 = None
        sub_115: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_443, sum_127);  mul_443 = sum_127 = None
        sub_116: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_445);  sub_115 = mul_445 = None
        mul_446: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_116);  div_26 = sub_116 = None
        mul_447: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, mul_4);  mul_4 = None
        sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 1]);  mul_447 = None
        sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_11: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_446, 3, 0);  full_default_5 = None
        view_as_complex_11: "c64[32, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_11);  select_scatter_11 = None
        _fft_c2c_23: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_11, [1, 2], 0, False);  view_as_complex_11 = None
        view_as_real_23: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_23);  _fft_c2c_23 = None
        select_24: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_23, 3, 0);  view_as_real_23 = None
        add_154: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_446, select_24);  mul_446 = select_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:138 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_25: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_448: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.0);  convert_element_type_25 = None
        mul_449: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_154, mul_448);  add_154 = mul_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:137 in forward, code: embeddings = self.projection(embeddings)
        view_135: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_449, [16384, 768]);  mul_449 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_132: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_52: "f32[16384, 768]" = torch.ops.aten.mm.default(view_135, permute_132);  permute_132 = None
        permute_133: "f32[768, 16384]" = torch.ops.aten.permute.default(view_135, [1, 0])
        mm_53: "f32[768, 768]" = torch.ops.aten.mm.default(permute_133, view);  permute_133 = view = None
        sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
        view_136: "f32[768]" = torch.ops.aten.reshape.default(sum_131, [768]);  sum_131 = None
        view_137: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_52, [32, 512, 768]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:136 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_451: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_137, primals_7);  primals_7 = None
        mul_452: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_451, 768)
        sum_132: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True)
        mul_453: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_451, mul);  mul_451 = None
        sum_133: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True);  mul_453 = None
        mul_454: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, sum_133);  sum_133 = None
        sub_118: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_452, sum_132);  mul_452 = sum_132 = None
        sub_119: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_118, mul_454);  sub_118 = mul_454 = None
        mul_455: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_119);  div_27 = sub_119 = None
        mul_456: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_137, mul);  mul = None
        sum_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_456, [0, 1]);  mul_456 = None
        sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_137, [0, 1]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:135 in forward, code: embeddings += position_embeddings
        sum_136: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_455, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:134 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_3, -1)
        unsqueeze_2: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        where_4: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_2, full_default_1, sum_136);  unsqueeze_2 = sum_136 = None
        full_default_18: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default_18, [primals_3], where_4, True);  full_default_18 = primals_3 = where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:479 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(primals_2, [32, 512]);  primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:130 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_1: "b8[32, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
        unsqueeze_3: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_5: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_3, full_default_1, mul_455);  unsqueeze_3 = None
        full_default_20: "f32[4, 768]" = torch.ops.aten.full.default([4, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[4, 768]" = torch.ops.aten.index_put.default(full_default_20, [expand], where_5, True);  full_default_20 = expand = where_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:129 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_2: "b8[32, 512]" = torch.ops.aten.eq.Scalar(primals_1, 3)
        unsqueeze_4: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_6: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_4, full_default_1, mul_455);  unsqueeze_4 = full_default_1 = mul_455 = None
        full_default_22: "f32[32000, 768]" = torch.ops.aten.full.default([32000, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "f32[32000, 768]" = torch.ops.aten.index_put.default(full_default_22, [primals_1], where_6, True);  full_default_22 = primals_1 = where_6 = None
        add_155: "f32[32000, 768]" = torch.ops.aten.add.Tensor(mm_1, index_put_2);  mm_1 = index_put_2 = None
        return (None, None, None, add_155, index_put_1, index_put, sum_134, sum_135, mm_53, view_136, sum_129, sum_130, mm_51, view_133, mm_49, view_130, sum_123, sum_124, sum_119, sum_120, mm_47, view_127, mm_45, view_124, sum_113, sum_114, sum_109, sum_110, mm_43, view_121, mm_41, view_118, sum_103, sum_104, sum_99, sum_100, mm_39, view_115, mm_37, view_112, sum_93, sum_94, sum_89, sum_90, mm_35, view_109, mm_33, view_106, sum_83, sum_84, sum_79, sum_80, mm_31, view_103, mm_29, view_100, sum_73, sum_74, sum_69, sum_70, mm_27, view_97, mm_25, view_94, sum_63, sum_64, sum_59, sum_60, mm_23, view_91, mm_21, view_88, sum_53, sum_54, sum_49, sum_50, mm_19, view_85, mm_17, view_82, sum_43, sum_44, sum_39, sum_40, mm_15, view_79, mm_13, view_76, sum_33, sum_34, sum_29, sum_30, mm_11, view_73, mm_9, view_70, sum_23, sum_24, sum_19, sum_20, mm_7, view_67, mm_5, view_64, sum_13, sum_14, None, None, mm_3, view_61, sum_8, sum_9, view_58, None)
