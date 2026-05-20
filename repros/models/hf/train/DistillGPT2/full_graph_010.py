class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512]", primals_2: "f32[50257, 768]", primals_4: "f32[768]", primals_7: "f32[768, 2304]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_13: "f32[768, 3072]", primals_15: "f32[3072, 768]", primals_16: "f32[768]", primals_19: "f32[768, 2304]", primals_21: "f32[768, 768]", primals_22: "f32[768]", primals_25: "f32[768, 3072]", primals_27: "f32[3072, 768]", primals_28: "f32[768]", primals_31: "f32[768, 2304]", primals_33: "f32[768, 768]", primals_34: "f32[768]", primals_37: "f32[768, 3072]", primals_39: "f32[3072, 768]", primals_40: "f32[768]", primals_43: "f32[768, 2304]", primals_45: "f32[768, 768]", primals_46: "f32[768]", primals_49: "f32[768, 3072]", primals_51: "f32[3072, 768]", primals_52: "f32[768]", primals_55: "f32[768, 2304]", primals_57: "f32[768, 768]", primals_58: "f32[768]", primals_61: "f32[768, 3072]", primals_63: "f32[3072, 768]", primals_64: "f32[768]", primals_67: "f32[768, 2304]", primals_69: "f32[768, 768]", primals_70: "f32[768]", primals_73: "f32[768, 3072]", primals_75: "f32[3072, 768]", primals_76: "f32[768]", embedding: "f32[32, 512, 768]", unsqueeze: "i64[1, 512]", embedding_1: "f32[1, 512, 768]", gt: "b8[32, 512, 768]", getitem_1: "f32[32, 512, 1]", rsqrt: "f32[32, 512, 1]", permute: "f32[32, 12, 512, 64]", permute_1: "f32[32, 12, 512, 64]", permute_2: "f32[32, 12, 512, 64]", where: "f32[32, 1, 512, 512]", getitem_5: "f32[32, 12, 512, 64]", getitem_6: "f32[32, 12, 512]", getitem_7: "i64[]", getitem_8: "i64[]", gt_1: "b8[32, 512, 768]", mul_6: "f32[32, 512, 768]", addmm_2: "f32[16384, 3072]", gt_2: "b8[32, 512, 768]", mul_14: "f32[32, 512, 768]", permute_4: "f32[32, 12, 512, 64]", permute_5: "f32[32, 12, 512, 64]", permute_6: "f32[32, 12, 512, 64]", getitem_16: "f32[32, 12, 512, 64]", getitem_17: "f32[32, 12, 512]", getitem_18: "i64[]", getitem_19: "i64[]", gt_3: "b8[32, 512, 768]", mul_18: "f32[32, 512, 768]", addmm_6: "f32[16384, 3072]", gt_4: "b8[32, 512, 768]", mul_26: "f32[32, 512, 768]", permute_8: "f32[32, 12, 512, 64]", permute_9: "f32[32, 12, 512, 64]", permute_10: "f32[32, 12, 512, 64]", getitem_27: "f32[32, 12, 512, 64]", getitem_28: "f32[32, 12, 512]", getitem_29: "i64[]", getitem_30: "i64[]", gt_5: "b8[32, 512, 768]", mul_30: "f32[32, 512, 768]", addmm_10: "f32[16384, 3072]", gt_6: "b8[32, 512, 768]", mul_38: "f32[32, 512, 768]", permute_12: "f32[32, 12, 512, 64]", permute_13: "f32[32, 12, 512, 64]", permute_14: "f32[32, 12, 512, 64]", getitem_38: "f32[32, 12, 512, 64]", getitem_39: "f32[32, 12, 512]", getitem_40: "i64[]", getitem_41: "i64[]", gt_7: "b8[32, 512, 768]", mul_42: "f32[32, 512, 768]", addmm_14: "f32[16384, 3072]", gt_8: "b8[32, 512, 768]", mul_50: "f32[32, 512, 768]", permute_16: "f32[32, 12, 512, 64]", permute_17: "f32[32, 12, 512, 64]", permute_18: "f32[32, 12, 512, 64]", getitem_49: "f32[32, 12, 512, 64]", getitem_50: "f32[32, 12, 512]", getitem_51: "i64[]", getitem_52: "i64[]", gt_9: "b8[32, 512, 768]", mul_54: "f32[32, 512, 768]", addmm_18: "f32[16384, 3072]", gt_10: "b8[32, 512, 768]", mul_62: "f32[32, 512, 768]", permute_20: "f32[32, 12, 512, 64]", permute_21: "f32[32, 12, 512, 64]", permute_22: "f32[32, 12, 512, 64]", getitem_60: "f32[32, 12, 512, 64]", getitem_61: "f32[32, 12, 512]", getitem_62: "i64[]", getitem_63: "i64[]", gt_11: "b8[32, 512, 768]", mul_66: "f32[32, 512, 768]", addmm_22: "f32[16384, 3072]", gt_12: "b8[32, 512, 768]", mul_74: "f32[32, 512, 768]", view_74: "f32[16384, 768]", div: "f32[32, 512, 1]", permute_30: "f32[3072, 16384]", permute_32: "f32[768, 16384]", div_1: "f32[32, 512, 1]", permute_40: "f32[768, 16384]", div_2: "f32[32, 512, 1]", permute_42: "f32[3072, 16384]", permute_44: "f32[768, 16384]", div_3: "f32[32, 512, 1]", permute_52: "f32[768, 16384]", div_4: "f32[32, 512, 1]", permute_54: "f32[3072, 16384]", permute_56: "f32[768, 16384]", div_5: "f32[32, 512, 1]", permute_64: "f32[768, 16384]", div_6: "f32[32, 512, 1]", permute_66: "f32[3072, 16384]", permute_68: "f32[768, 16384]", div_7: "f32[32, 512, 1]", permute_76: "f32[768, 16384]", div_8: "f32[32, 512, 1]", permute_78: "f32[3072, 16384]", permute_80: "f32[768, 16384]", div_9: "f32[32, 512, 1]", permute_88: "f32[768, 16384]", div_10: "f32[32, 512, 1]", permute_90: "f32[3072, 16384]", permute_92: "f32[768, 16384]", div_11: "f32[32, 512, 1]", permute_100: "f32[768, 16384]", tangents_1: "f32[32, 512, 50257]", tangents_2: "f32[32, 512, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:706 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_76: "f32[16384, 50257]" = torch.ops.aten.reshape.default(tangents_1, [16384, 50257]);  tangents_1 = None
        permute_25: "f32[50257, 16384]" = torch.ops.aten.permute.default(view_76, [1, 0])
        constant_pad_nd_default_2: "f32[50260, 16384]" = torch.ops.aten.constant_pad_nd.default(permute_25, [0, 0, 0, 3]);  permute_25 = None
        mm_default_1: "f32[50260, 768]" = torch.ops.aten.mm.default(constant_pad_nd_default_2, view_74);  constant_pad_nd_default_2 = view_74 = None
        slice_tensor: "f32[50257, 768]" = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -3);  mm_default_1 = None
        permute_24: "f32[768, 50257]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_27: "f32[50257, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        constant_pad_nd_default: "f32[16384, 50260]" = torch.ops.aten.constant_pad_nd.default(view_76, [0, 3, 0, 0]);  view_76 = None
        constant_pad_nd_default_1: "f32[50260, 768]" = torch.ops.aten.constant_pad_nd.default(permute_27, [0, 0, 0, 3]);  permute_27 = None
        mm_default: "f32[16384, 768]" = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        view_77: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_default, [32, 512, 768]);  mm_default = None
        add_54: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(tangents_2, view_77);  tangents_2 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_77: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_54, primals_76);  primals_76 = None
        mul_78: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_77, 768)
        sum_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_77, [2], True)
        mul_79: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_77, mul_74);  mul_77 = None
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_79, [2], True);  mul_79 = None
        mul_80: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, sum_2);  sum_2 = None
        sub_16: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_78, sum_1);  mul_78 = sum_1 = None
        sub_17: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_16, mul_80);  sub_16 = mul_80 = None
        mul_81: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div, sub_17);  div = sub_17 = None
        mul_82: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_54, mul_74);  mul_74 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_82, [0, 1]);  mul_82 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_54, [0, 1]);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_83: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.0);  convert_element_type = None
        mul_84: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, mul_83);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_79: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_84, [16384, 768]);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_29: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        mm_3: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_79, permute_29);  permute_29 = None
        mm_4: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_30, view_79);  permute_30 = None
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_79, [0], True);  view_79 = None
        view_80: "f32[768]" = torch.ops.aten.reshape.default(sum_5, [768]);  sum_5 = None
        view_81: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_3, [32, 512, 3072]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_70: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        mul_85: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_81, mul_68);  mul_68 = None
        pow_6: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_69: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_49: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_70, mul_69);  mul_69 = None
        mul_70: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_5: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_50: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_86: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_81, add_50);  view_81 = add_50 = None
        mul_87: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_18: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_87);  mul_87 = None
        mul_88: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_85, sub_18);  mul_85 = sub_18 = None
        mul_89: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_88, 0.7978845608028654);  mul_88 = None
        mul_90: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_89, 0.044715)
        pow_7: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 2.0);  view_70 = None
        mul_91: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_7, 3.0);  pow_7 = None
        mul_92: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
        add_55: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_89, mul_92);  mul_89 = mul_92 = None
        mul_93: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_86, 0.5);  mul_86 = None
        add_56: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_55, mul_93);  add_55 = mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_82: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_56, [16384, 3072]);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_31: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        mm_5: "f32[16384, 768]" = torch.ops.aten.mm.default(view_82, permute_31);  permute_31 = None
        mm_6: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_32, view_82);  permute_32 = None
        sum_6: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_82, [0], True);  view_82 = None
        view_83: "f32[3072]" = torch.ops.aten.reshape.default(sum_6, [3072]);  sum_6 = None
        view_84: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_5, [32, 512, 768]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_95: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_84, primals_70);  primals_70 = None
        mul_96: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_95, 768)
        sum_7: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True)
        mul_97: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_95, mul_66);  mul_95 = None
        sum_8: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_97, [2], True);  mul_97 = None
        mul_98: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_8);  sum_8 = None
        sub_20: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_96, sum_7);  mul_96 = sum_7 = None
        sub_21: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_20, mul_98);  sub_20 = mul_98 = None
        mul_99: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_21);  div_1 = sub_21 = None
        mul_100: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_84, mul_66);  mul_66 = None
        sum_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_100, [0, 1]);  mul_100 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_84, [0, 1]);  view_84 = None
        add_57: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, mul_99);  mul_81 = mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_1: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_101: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.0);  convert_element_type_1 = None
        mul_102: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_57, mul_101);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_85: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_102, [16384, 768]);  mul_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        mm_7: "f32[16384, 768]" = torch.ops.aten.mm.default(view_85, permute_33);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_23, [32, 512, -1]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_66, [-1, 768]);  view_66 = None
        permute_34: "f32[768, 16384]" = torch.ops.aten.permute.default(view_67, [1, 0]);  view_67 = None
        mm_8: "f32[768, 768]" = torch.ops.aten.mm.default(permute_34, view_85);  permute_34 = None
        sum_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_85, [0], True);  view_85 = None
        view_86: "f32[768]" = torch.ops.aten.reshape.default(sum_11, [768]);  sum_11 = None
        view_87: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_7, [32, 512, 768]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_88: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_87, [32, 512, 12, 64]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where, [32, 12, 512, 512]);  where = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_35, permute_22, permute_20, permute_21, expand_2, getitem_60, getitem_61, getitem_62, getitem_63, 1e-30, [True, True, True, False], scale = 0.125);  permute_35 = permute_22 = permute_20 = permute_21 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = None
        getitem_68: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_69: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_70: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_36: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_68, [0, 2, 1, 3]);  getitem_68 = None
        view_89: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_36, [32, 512, 768]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_37: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_70, [0, 2, 1, 3]);  getitem_70 = None
        view_90: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_37, [32, 512, 768]);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_38: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3]);  getitem_69 = None
        view_91: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_38, [32, 512, 768]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_1: "f32[32, 512, 2304]" = torch.ops.aten.cat.default([view_89, view_91, view_90], 2);  view_89 = view_91 = view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_92: "f32[16384, 2304]" = torch.ops.aten.reshape.default(cat_1, [16384, 2304]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_39: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        mm_9: "f32[16384, 768]" = torch.ops.aten.mm.default(view_92, permute_39);  permute_39 = None
        mm_10: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_40, view_92);  permute_40 = None
        sum_12: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_92, [0], True);  view_92 = None
        view_93: "f32[2304]" = torch.ops.aten.reshape.default(sum_12, [2304]);  sum_12 = None
        view_94: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_9, [32, 512, 768]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_104: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_94, primals_64);  primals_64 = None
        mul_105: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_104, 768)
        sum_13: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [2], True)
        mul_106: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_104, mul_62);  mul_104 = None
        sum_14: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_106, [2], True);  mul_106 = None
        mul_107: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_62, sum_14);  sum_14 = None
        sub_23: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_105, sum_13);  mul_105 = sum_13 = None
        sub_24: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_23, mul_107);  sub_23 = mul_107 = None
        mul_108: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_24);  div_2 = sub_24 = None
        mul_109: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_94, mul_62);  mul_62 = None
        sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_109, [0, 1]);  mul_109 = None
        sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_94, [0, 1]);  view_94 = None
        add_58: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_57, mul_108);  add_57 = mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_110: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.0);  convert_element_type_2 = None
        mul_111: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_58, mul_110);  mul_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_95: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_111, [16384, 768]);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_41: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        mm_11: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_95, permute_41);  permute_41 = None
        mm_12: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_42, view_95);  permute_42 = None
        sum_17: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_95, [0], True);  view_95 = None
        view_96: "f32[768]" = torch.ops.aten.reshape.default(sum_17, [768]);  sum_17 = None
        view_97: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_11, [32, 512, 3072]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_58: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_18, [32, 512, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        mul_112: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_97, mul_56);  mul_56 = None
        pow_5: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_57: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_58, mul_57);  mul_57 = None
        mul_58: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_42: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_113: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_97, add_42);  view_97 = add_42 = None
        mul_114: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_25: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_114);  mul_114 = None
        mul_115: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_112, sub_25);  mul_112 = sub_25 = None
        mul_116: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_115, 0.7978845608028654);  mul_115 = None
        mul_117: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_116, 0.044715)
        pow_8: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 2.0);  view_58 = None
        mul_118: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_8, 3.0);  pow_8 = None
        mul_119: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
        add_59: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_116, mul_119);  mul_116 = mul_119 = None
        mul_120: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_113, 0.5);  mul_113 = None
        add_60: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_59, mul_120);  add_59 = mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_98: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_60, [16384, 3072]);  add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        mm_13: "f32[16384, 768]" = torch.ops.aten.mm.default(view_98, permute_43);  permute_43 = None
        mm_14: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_44, view_98);  permute_44 = None
        sum_18: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True);  view_98 = None
        view_99: "f32[3072]" = torch.ops.aten.reshape.default(sum_18, [3072]);  sum_18 = None
        view_100: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_13, [32, 512, 768]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_122: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_100, primals_58);  primals_58 = None
        mul_123: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, 768)
        sum_19: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
        mul_124: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, mul_54);  mul_122 = None
        sum_20: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
        mul_125: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_54, sum_20);  sum_20 = None
        sub_27: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_123, sum_19);  mul_123 = sum_19 = None
        sub_28: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_27, mul_125);  sub_27 = mul_125 = None
        mul_126: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_28);  div_3 = sub_28 = None
        mul_127: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_100, mul_54);  mul_54 = None
        sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
        sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_100, [0, 1]);  view_100 = None
        add_61: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_58, mul_126);  add_58 = mul_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_3: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_128: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.0);  convert_element_type_3 = None
        mul_129: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_61, mul_128);  mul_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_101: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_129, [16384, 768]);  mul_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        mm_15: "f32[16384, 768]" = torch.ops.aten.mm.default(view_101, permute_45);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_19, [32, 512, -1]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_54, [-1, 768]);  view_54 = None
        permute_46: "f32[768, 16384]" = torch.ops.aten.permute.default(view_55, [1, 0]);  view_55 = None
        mm_16: "f32[768, 768]" = torch.ops.aten.mm.default(permute_46, view_101);  permute_46 = None
        sum_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_101, [0], True);  view_101 = None
        view_102: "f32[768]" = torch.ops.aten.reshape.default(sum_23, [768]);  sum_23 = None
        view_103: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_15, [32, 512, 768]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_104: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_103, [32, 512, 12, 64]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_47, permute_18, permute_16, permute_17, expand_2, getitem_49, getitem_50, getitem_51, getitem_52, 1e-30, [True, True, True, False], scale = 0.125);  permute_47 = permute_18 = permute_16 = permute_17 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = None
        getitem_72: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_1[0]
        getitem_73: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_1[1]
        getitem_74: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_48: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None
        view_105: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_48, [32, 512, 768]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_49: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        view_106: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_49, [32, 512, 768]);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_50: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_73, [0, 2, 1, 3]);  getitem_73 = None
        view_107: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_50, [32, 512, 768]);  permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_2: "f32[32, 512, 2304]" = torch.ops.aten.cat.default([view_105, view_107, view_106], 2);  view_105 = view_107 = view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_108: "f32[16384, 2304]" = torch.ops.aten.reshape.default(cat_2, [16384, 2304]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_51: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        mm_17: "f32[16384, 768]" = torch.ops.aten.mm.default(view_108, permute_51);  permute_51 = None
        mm_18: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_52, view_108);  permute_52 = None
        sum_24: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_108, [0], True);  view_108 = None
        view_109: "f32[2304]" = torch.ops.aten.reshape.default(sum_24, [2304]);  sum_24 = None
        view_110: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_17, [32, 512, 768]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_131: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_110, primals_52);  primals_52 = None
        mul_132: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_131, 768)
        sum_25: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True)
        mul_133: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_131, mul_50);  mul_131 = None
        sum_26: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_133, [2], True);  mul_133 = None
        mul_134: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_26);  sum_26 = None
        sub_30: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_132, sum_25);  mul_132 = sum_25 = None
        sub_31: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_30, mul_134);  sub_30 = mul_134 = None
        mul_135: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_31);  div_4 = sub_31 = None
        mul_136: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_110, mul_50);  mul_50 = None
        sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_136, [0, 1]);  mul_136 = None
        sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_110, [0, 1]);  view_110 = None
        add_62: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_61, mul_135);  add_61 = mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_4: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_137: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.0);  convert_element_type_4 = None
        mul_138: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_62, mul_137);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_111: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_138, [16384, 768]);  mul_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        mm_19: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_111, permute_53);  permute_53 = None
        mm_20: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_54, view_111);  permute_54 = None
        sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_111, [0], True);  view_111 = None
        view_112: "f32[768]" = torch.ops.aten.reshape.default(sum_29, [768]);  sum_29 = None
        view_113: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_19, [32, 512, 3072]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_46: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_14, [32, 512, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        mul_139: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_113, mul_44);  mul_44 = None
        pow_4: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_45: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_33: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_46, mul_45);  mul_45 = None
        mul_46: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_3: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_34: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_140: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_113, add_34);  view_113 = add_34 = None
        mul_141: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_32: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_141);  mul_141 = None
        mul_142: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_139, sub_32);  mul_139 = sub_32 = None
        mul_143: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_142, 0.7978845608028654);  mul_142 = None
        mul_144: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_143, 0.044715)
        pow_9: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 2.0);  view_46 = None
        mul_145: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_9, 3.0);  pow_9 = None
        mul_146: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        add_63: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_143, mul_146);  mul_143 = mul_146 = None
        mul_147: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_140, 0.5);  mul_140 = None
        add_64: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_63, mul_147);  add_63 = mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_114: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_64, [16384, 3072]);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        mm_21: "f32[16384, 768]" = torch.ops.aten.mm.default(view_114, permute_55);  permute_55 = None
        mm_22: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_56, view_114);  permute_56 = None
        sum_30: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True);  view_114 = None
        view_115: "f32[3072]" = torch.ops.aten.reshape.default(sum_30, [3072]);  sum_30 = None
        view_116: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_21, [32, 512, 768]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_149: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_116, primals_46);  primals_46 = None
        mul_150: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_149, 768)
        sum_31: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [2], True)
        mul_151: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_149, mul_42);  mul_149 = None
        sum_32: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [2], True);  mul_151 = None
        mul_152: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_32);  sum_32 = None
        sub_34: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_150, sum_31);  mul_150 = sum_31 = None
        sub_35: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_34, mul_152);  sub_34 = mul_152 = None
        mul_153: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_35);  div_5 = sub_35 = None
        mul_154: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_116, mul_42);  mul_42 = None
        sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_154, [0, 1]);  mul_154 = None
        sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_116, [0, 1]);  view_116 = None
        add_65: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_62, mul_153);  add_62 = mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_5: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_155: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.0);  convert_element_type_5 = None
        mul_156: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_65, mul_155);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_117: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_156, [16384, 768]);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        mm_23: "f32[16384, 768]" = torch.ops.aten.mm.default(view_117, permute_57);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_15, [32, 512, -1]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_42, [-1, 768]);  view_42 = None
        permute_58: "f32[768, 16384]" = torch.ops.aten.permute.default(view_43, [1, 0]);  view_43 = None
        mm_24: "f32[768, 768]" = torch.ops.aten.mm.default(permute_58, view_117);  permute_58 = None
        sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True);  view_117 = None
        view_118: "f32[768]" = torch.ops.aten.reshape.default(sum_35, [768]);  sum_35 = None
        view_119: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_23, [32, 512, 768]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_120: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_119, [32, 512, 12, 64]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_59: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_59, permute_14, permute_12, permute_13, expand_2, getitem_38, getitem_39, getitem_40, getitem_41, 1e-30, [True, True, True, False], scale = 0.125);  permute_59 = permute_14 = permute_12 = permute_13 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = None
        getitem_76: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_2[0]
        getitem_77: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_2[1]
        getitem_78: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_60: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_76, [0, 2, 1, 3]);  getitem_76 = None
        view_121: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_60, [32, 512, 768]);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_61: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_78, [0, 2, 1, 3]);  getitem_78 = None
        view_122: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_61, [32, 512, 768]);  permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_62: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_77, [0, 2, 1, 3]);  getitem_77 = None
        view_123: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_62, [32, 512, 768]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_3: "f32[32, 512, 2304]" = torch.ops.aten.cat.default([view_121, view_123, view_122], 2);  view_121 = view_123 = view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_124: "f32[16384, 2304]" = torch.ops.aten.reshape.default(cat_3, [16384, 2304]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_63: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        mm_25: "f32[16384, 768]" = torch.ops.aten.mm.default(view_124, permute_63);  permute_63 = None
        mm_26: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_64, view_124);  permute_64 = None
        sum_36: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_124, [0], True);  view_124 = None
        view_125: "f32[2304]" = torch.ops.aten.reshape.default(sum_36, [2304]);  sum_36 = None
        view_126: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_25, [32, 512, 768]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_158: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_126, primals_40);  primals_40 = None
        mul_159: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_158, 768)
        sum_37: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_158, [2], True)
        mul_160: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_158, mul_38);  mul_158 = None
        sum_38: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_160, [2], True);  mul_160 = None
        mul_161: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, sum_38);  sum_38 = None
        sub_37: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_159, sum_37);  mul_159 = sum_37 = None
        sub_38: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_37, mul_161);  sub_37 = mul_161 = None
        mul_162: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_38);  div_6 = sub_38 = None
        mul_163: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_126, mul_38);  mul_38 = None
        sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_163, [0, 1]);  mul_163 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_126, [0, 1]);  view_126 = None
        add_66: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_65, mul_162);  add_65 = mul_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_6: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_164: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.0);  convert_element_type_6 = None
        mul_165: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_66, mul_164);  mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_127: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_165, [16384, 768]);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        mm_27: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_127, permute_65);  permute_65 = None
        mm_28: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_66, view_127);  permute_66 = None
        sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_127, [0], True);  view_127 = None
        view_128: "f32[768]" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None
        view_129: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_27, [32, 512, 3072]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_34: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_32: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        mul_166: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_32);  mul_32 = None
        pow_3: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_33: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_34, mul_33);  mul_33 = None
        mul_34: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_34);  mul_34 = None
        add_26: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_167: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, add_26);  view_129 = add_26 = None
        mul_168: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_39: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_168);  mul_168 = None
        mul_169: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_166, sub_39);  mul_166 = sub_39 = None
        mul_170: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_169, 0.7978845608028654);  mul_169 = None
        mul_171: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_170, 0.044715)
        pow_10: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 2.0);  view_34 = None
        mul_172: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_10, 3.0);  pow_10 = None
        mul_173: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
        add_67: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_170, mul_173);  mul_170 = mul_173 = None
        mul_174: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_167, 0.5);  mul_167 = None
        add_68: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_67, mul_174);  add_67 = mul_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_130: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_68, [16384, 3072]);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_67: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        mm_29: "f32[16384, 768]" = torch.ops.aten.mm.default(view_130, permute_67);  permute_67 = None
        mm_30: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_68, view_130);  permute_68 = None
        sum_42: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_130, [0], True);  view_130 = None
        view_131: "f32[3072]" = torch.ops.aten.reshape.default(sum_42, [3072]);  sum_42 = None
        view_132: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_29, [32, 512, 768]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_176: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_132, primals_34);  primals_34 = None
        mul_177: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_176, 768)
        sum_43: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True)
        mul_178: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_176, mul_30);  mul_176 = None
        sum_44: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_178, [2], True);  mul_178 = None
        mul_179: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_44);  sum_44 = None
        sub_41: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_177, sum_43);  mul_177 = sum_43 = None
        sub_42: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_179);  sub_41 = mul_179 = None
        mul_180: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_42);  div_7 = sub_42 = None
        mul_181: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_132, mul_30);  mul_30 = None
        sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_181, [0, 1]);  mul_181 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_132, [0, 1]);  view_132 = None
        add_69: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_66, mul_180);  add_66 = mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_7: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_182: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.0);  convert_element_type_7 = None
        mul_183: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_69, mul_182);  mul_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_133: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_183, [16384, 768]);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        mm_31: "f32[16384, 768]" = torch.ops.aten.mm.default(view_133, permute_69);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_11, [32, 512, -1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_30, [-1, 768]);  view_30 = None
        permute_70: "f32[768, 16384]" = torch.ops.aten.permute.default(view_31, [1, 0]);  view_31 = None
        mm_32: "f32[768, 768]" = torch.ops.aten.mm.default(permute_70, view_133);  permute_70 = None
        sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_133, [0], True);  view_133 = None
        view_134: "f32[768]" = torch.ops.aten.reshape.default(sum_47, [768]);  sum_47 = None
        view_135: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_31, [32, 512, 768]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_136: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_135, [32, 512, 12, 64]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_71: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_71, permute_10, permute_8, permute_9, expand_2, getitem_27, getitem_28, getitem_29, getitem_30, 1e-30, [True, True, True, False], scale = 0.125);  permute_71 = permute_10 = permute_8 = permute_9 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = None
        getitem_80: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_3[0]
        getitem_81: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_3[1]
        getitem_82: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_72: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_80, [0, 2, 1, 3]);  getitem_80 = None
        view_137: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_72, [32, 512, 768]);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_73: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        view_138: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_73, [32, 512, 768]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_74: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3]);  getitem_81 = None
        view_139: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_74, [32, 512, 768]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_4: "f32[32, 512, 2304]" = torch.ops.aten.cat.default([view_137, view_139, view_138], 2);  view_137 = view_139 = view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_140: "f32[16384, 2304]" = torch.ops.aten.reshape.default(cat_4, [16384, 2304]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_75: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        mm_33: "f32[16384, 768]" = torch.ops.aten.mm.default(view_140, permute_75);  permute_75 = None
        mm_34: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_76, view_140);  permute_76 = None
        sum_48: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_140, [0], True);  view_140 = None
        view_141: "f32[2304]" = torch.ops.aten.reshape.default(sum_48, [2304]);  sum_48 = None
        view_142: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_33, [32, 512, 768]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_185: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_142, primals_28);  primals_28 = None
        mul_186: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_185, 768)
        sum_49: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True)
        mul_187: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_185, mul_26);  mul_185 = None
        sum_50: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True);  mul_187 = None
        mul_188: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, sum_50);  sum_50 = None
        sub_44: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_186, sum_49);  mul_186 = sum_49 = None
        sub_45: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_188);  sub_44 = mul_188 = None
        mul_189: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_45);  div_8 = sub_45 = None
        mul_190: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_142, mul_26);  mul_26 = None
        sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 1]);  mul_190 = None
        sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_142, [0, 1]);  view_142 = None
        add_70: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_69, mul_189);  add_69 = mul_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_191: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.0);  convert_element_type_8 = None
        mul_192: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_70, mul_191);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_143: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_192, [16384, 768]);  mul_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_77: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        mm_35: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_143, permute_77);  permute_77 = None
        mm_36: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_78, view_143);  permute_78 = None
        sum_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_143, [0], True);  view_143 = None
        view_144: "f32[768]" = torch.ops.aten.reshape.default(sum_53, [768]);  sum_53 = None
        view_145: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_35, [32, 512, 3072]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_22: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_6, [32, 512, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        mul_193: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_145, mul_20);  mul_20 = None
        pow_2: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_21: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_22, mul_21);  mul_21 = None
        mul_22: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_18: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_194: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_145, add_18);  view_145 = add_18 = None
        mul_195: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_46: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_195);  mul_195 = None
        mul_196: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_193, sub_46);  mul_193 = sub_46 = None
        mul_197: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_196, 0.7978845608028654);  mul_196 = None
        mul_198: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_197, 0.044715)
        pow_11: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 2.0);  view_22 = None
        mul_199: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_11, 3.0);  pow_11 = None
        mul_200: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None
        add_71: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_197, mul_200);  mul_197 = mul_200 = None
        mul_201: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_194, 0.5);  mul_194 = None
        add_72: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_71, mul_201);  add_71 = mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_146: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_72, [16384, 3072]);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_79: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        mm_37: "f32[16384, 768]" = torch.ops.aten.mm.default(view_146, permute_79);  permute_79 = None
        mm_38: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_80, view_146);  permute_80 = None
        sum_54: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_146, [0], True);  view_146 = None
        view_147: "f32[3072]" = torch.ops.aten.reshape.default(sum_54, [3072]);  sum_54 = None
        view_148: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_37, [32, 512, 768]);  mm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_203: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_148, primals_22);  primals_22 = None
        mul_204: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_203, 768)
        sum_55: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True)
        mul_205: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_203, mul_18);  mul_203 = None
        sum_56: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [2], True);  mul_205 = None
        mul_206: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, sum_56);  sum_56 = None
        sub_48: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_204, sum_55);  mul_204 = sum_55 = None
        sub_49: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_206);  sub_48 = mul_206 = None
        mul_207: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_49);  div_9 = sub_49 = None
        mul_208: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_148, mul_18);  mul_18 = None
        sum_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_208, [0, 1]);  mul_208 = None
        sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_148, [0, 1]);  view_148 = None
        add_73: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_70, mul_207);  add_70 = mul_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_9: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_209: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.0);  convert_element_type_9 = None
        mul_210: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_73, mul_209);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_149: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_210, [16384, 768]);  mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        mm_39: "f32[16384, 768]" = torch.ops.aten.mm.default(view_149, permute_81);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_7, [32, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_18, [-1, 768]);  view_18 = None
        permute_82: "f32[768, 16384]" = torch.ops.aten.permute.default(view_19, [1, 0]);  view_19 = None
        mm_40: "f32[768, 768]" = torch.ops.aten.mm.default(permute_82, view_149);  permute_82 = None
        sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_149, [0], True);  view_149 = None
        view_150: "f32[768]" = torch.ops.aten.reshape.default(sum_59, [768]);  sum_59 = None
        view_151: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_39, [32, 512, 768]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_152: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_151, [32, 512, 12, 64]);  view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_83: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_83, permute_6, permute_4, permute_5, expand_2, getitem_16, getitem_17, getitem_18, getitem_19, 1e-30, [True, True, True, False], scale = 0.125);  permute_83 = permute_6 = permute_4 = permute_5 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = None
        getitem_84: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_4[0]
        getitem_85: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_4[1]
        getitem_86: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_84: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_84, [0, 2, 1, 3]);  getitem_84 = None
        view_153: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_84, [32, 512, 768]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_85: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_86, [0, 2, 1, 3]);  getitem_86 = None
        view_154: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_85, [32, 512, 768]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_86: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
        view_155: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_86, [32, 512, 768]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_5: "f32[32, 512, 2304]" = torch.ops.aten.cat.default([view_153, view_155, view_154], 2);  view_153 = view_155 = view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_156: "f32[16384, 2304]" = torch.ops.aten.reshape.default(cat_5, [16384, 2304]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_87: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        mm_41: "f32[16384, 768]" = torch.ops.aten.mm.default(view_156, permute_87);  permute_87 = None
        mm_42: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_88, view_156);  permute_88 = None
        sum_60: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
        view_157: "f32[2304]" = torch.ops.aten.reshape.default(sum_60, [2304]);  sum_60 = None
        view_158: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_41, [32, 512, 768]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_212: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_158, primals_16);  primals_16 = None
        mul_213: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_212, 768)
        sum_61: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_212, [2], True)
        mul_214: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_212, mul_14);  mul_212 = None
        sum_62: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_214, [2], True);  mul_214 = None
        mul_215: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_14, sum_62);  sum_62 = None
        sub_51: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_213, sum_61);  mul_213 = sum_61 = None
        sub_52: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_215);  sub_51 = mul_215 = None
        mul_216: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_52);  div_10 = sub_52 = None
        mul_217: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_158, mul_14);  mul_14 = None
        sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_217, [0, 1]);  mul_217 = None
        sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_158, [0, 1]);  view_158 = None
        add_74: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_73, mul_216);  add_73 = mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_10: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_218: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.0);  convert_element_type_10 = None
        mul_219: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_74, mul_218);  mul_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_159: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_219, [16384, 768]);  mul_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_89: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        mm_43: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_159, permute_89);  permute_89 = None
        mm_44: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_90, view_159);  permute_90 = None
        sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_159, [0], True);  view_159 = None
        view_160: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
        view_161: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(mm_43, [32, 512, 3072]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_10: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_8: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        mul_220: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_161, mul_8);  mul_8 = None
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_9: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_10, mul_9);  mul_9 = None
        mul_10: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_10);  mul_10 = None
        add_10: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_221: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_161, add_10);  view_161 = add_10 = None
        mul_222: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_53: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_222);  mul_222 = None
        mul_223: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_220, sub_53);  mul_220 = sub_53 = None
        mul_224: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_223, 0.7978845608028654);  mul_223 = None
        mul_225: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_224, 0.044715)
        pow_12: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 2.0);  view_10 = None
        mul_226: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_12, 3.0);  pow_12 = None
        mul_227: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        add_75: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_224, mul_227);  mul_224 = mul_227 = None
        mul_228: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_221, 0.5);  mul_221 = None
        add_76: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_75, mul_228);  add_75 = mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_162: "f32[16384, 3072]" = torch.ops.aten.reshape.default(add_76, [16384, 3072]);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_91: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        mm_45: "f32[16384, 768]" = torch.ops.aten.mm.default(view_162, permute_91);  permute_91 = None
        mm_46: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_92, view_162);  permute_92 = None
        sum_66: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_162, [0], True);  view_162 = None
        view_163: "f32[3072]" = torch.ops.aten.reshape.default(sum_66, [3072]);  sum_66 = None
        view_164: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_45, [32, 512, 768]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_230: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_164, primals_10);  primals_10 = None
        mul_231: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_230, 768)
        sum_67: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_230, [2], True)
        mul_232: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_230, mul_6);  mul_230 = None
        sum_68: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_232, [2], True);  mul_232 = None
        mul_233: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_6, sum_68);  sum_68 = None
        sub_55: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_231, sum_67);  mul_231 = sum_67 = None
        sub_56: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_233);  sub_55 = mul_233 = None
        mul_234: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_56);  div_11 = sub_56 = None
        mul_235: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_164, mul_6);  mul_6 = None
        sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_235, [0, 1]);  mul_235 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_164, [0, 1]);  view_164 = None
        add_77: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_74, mul_234);  add_74 = mul_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_11: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_236: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.0);  convert_element_type_11 = None
        mul_237: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_77, mul_236);  mul_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_165: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_237, [16384, 768]);  mul_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_93: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        mm_47: "f32[16384, 768]" = torch.ops.aten.mm.default(view_165, permute_93);  permute_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_3, [32, 512, -1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_6, [-1, 768]);  view_6 = None
        permute_94: "f32[768, 16384]" = torch.ops.aten.permute.default(view_7, [1, 0]);  view_7 = None
        mm_48: "f32[768, 768]" = torch.ops.aten.mm.default(permute_94, view_165);  permute_94 = None
        sum_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
        view_166: "f32[768]" = torch.ops.aten.reshape.default(sum_71, [768]);  sum_71 = None
        view_167: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_47, [32, 512, 768]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_168: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_167, [32, 512, 12, 64]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_95, permute_2, permute, permute_1, expand_2, getitem_5, getitem_6, getitem_7, getitem_8, 1e-30, [True, True, True, False], scale = 0.125);  permute_95 = permute_2 = permute = permute_1 = expand_2 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = None
        getitem_88: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_5[0]
        getitem_89: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_5[1]
        getitem_90: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_96: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_88, [0, 2, 1, 3]);  getitem_88 = None
        view_169: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_96, [32, 512, 768]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_97: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        view_170: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_97, [32, 512, 768]);  permute_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_98: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_89, [0, 2, 1, 3]);  getitem_89 = None
        view_171: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_98, [32, 512, 768]);  permute_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_6: "f32[32, 512, 2304]" = torch.ops.aten.cat.default([view_169, view_171, view_170], 2);  view_169 = view_171 = view_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_172: "f32[16384, 2304]" = torch.ops.aten.reshape.default(cat_6, [16384, 2304]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_99: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        mm_49: "f32[16384, 768]" = torch.ops.aten.mm.default(view_172, permute_99);  permute_99 = None
        mm_50: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_100, view_172);  permute_100 = None
        sum_72: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_172, [0], True);  view_172 = None
        view_173: "f32[2304]" = torch.ops.aten.reshape.default(sum_72, [2304]);  sum_72 = None
        view_174: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_49, [32, 512, 768]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_239: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_174, primals_4);  primals_4 = None
        mul_240: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_239, 768)
        sum_73: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 1.0);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_1, getitem_1);  mul_1 = getitem_1 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_241: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_239, mul_2);  mul_239 = None
        sum_74: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True);  mul_241 = None
        mul_242: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_74);  sum_74 = None
        sub_58: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_240, sum_73);  mul_240 = sum_73 = None
        sub_59: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_58, mul_242);  sub_58 = mul_242 = None
        div_12: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_243: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_59);  div_12 = sub_59 = None
        mul_244: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_174, mul_2);  mul_2 = None
        sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_244, [0, 1]);  mul_244 = None
        sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_174, [0, 1]);  view_174 = None
        add_78: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_77, mul_243);  add_77 = mul_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        convert_element_type_12: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_245: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.0);  convert_element_type_12 = None
        mul_246: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_78, mul_245);  add_78 = mul_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        sum_77: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_246, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(unsqueeze, -1)
        unsqueeze_10: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        where_6: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_10, full_default_2, sum_77);  unsqueeze_10 = sum_77 = None
        full_default_14: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_14, [unsqueeze], where_6, True);  full_default_14 = unsqueeze = where_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_2: "b8[32, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_11: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_7: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_11, full_default_2, mul_246);  unsqueeze_11 = full_default_2 = mul_246 = None
        full_default_16: "f32[50257, 768]" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[50257, 768]" = torch.ops.aten.index_put.default(full_default_16, [primals_1], where_7, True);  full_default_16 = primals_1 = where_7 = None
        add_79: "f32[50257, 768]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_1);  slice_tensor = index_put_1 = None
        return (None, add_79, index_put, sum_75, sum_76, view_173, mm_50, view_166, mm_48, sum_69, sum_70, view_163, mm_46, view_160, mm_44, sum_63, sum_64, view_157, mm_42, view_150, mm_40, sum_57, sum_58, view_147, mm_38, view_144, mm_36, sum_51, sum_52, view_141, mm_34, view_134, mm_32, sum_45, sum_46, view_131, mm_30, view_128, mm_28, sum_39, sum_40, view_125, mm_26, view_118, mm_24, sum_33, sum_34, view_115, mm_22, view_112, mm_20, sum_27, sum_28, view_109, mm_18, view_102, mm_16, sum_21, sum_22, view_99, mm_14, view_96, mm_12, sum_15, sum_16, view_93, mm_10, view_86, mm_8, sum_9, sum_10, view_83, mm_6, view_80, mm_4, sum_3, sum_4)
