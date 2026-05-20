class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[256, 128]", primals_2: "f32[30522, 768]", primals_3: "i64[1, 512]", primals_5: "f32[768]", primals_7: "f32[768, 768]", primals_9: "f32[768, 768]", primals_11: "f32[768, 768]", primals_13: "f32[768, 768]", primals_15: "f32[768]", primals_17: "f32[3072, 768]", primals_19: "f32[768, 3072]", primals_21: "f32[768]", primals_23: "f32[768, 768]", primals_25: "f32[768, 768]", primals_27: "f32[768, 768]", primals_29: "f32[768, 768]", primals_31: "f32[768]", primals_33: "f32[3072, 768]", primals_35: "f32[768, 3072]", primals_37: "f32[768]", primals_39: "f32[768, 768]", primals_41: "f32[768, 768]", primals_43: "f32[768, 768]", primals_45: "f32[768, 768]", primals_47: "f32[768]", primals_49: "f32[3072, 768]", primals_51: "f32[768, 3072]", primals_53: "f32[768]", primals_55: "f32[768, 768]", primals_57: "f32[768, 768]", primals_59: "f32[768, 768]", primals_61: "f32[768, 768]", primals_63: "f32[768]", primals_65: "f32[3072, 768]", primals_67: "f32[768, 3072]", primals_69: "f32[768]", primals_71: "f32[768, 768]", primals_73: "f32[768, 768]", primals_75: "f32[768, 768]", primals_77: "f32[768, 768]", primals_79: "f32[768]", primals_81: "f32[3072, 768]", primals_83: "f32[768, 3072]", primals_85: "f32[768]", primals_87: "f32[768, 768]", primals_89: "f32[768, 768]", primals_91: "f32[768, 768]", primals_93: "f32[768, 768]", primals_95: "f32[768]", primals_97: "f32[3072, 768]", primals_99: "f32[768, 3072]", primals_101: "f32[768]", primals_103: "f32[768, 768]", primals_105: "f32[768]", primals_108: "i64[256, 128]", embedding: "f32[256, 128, 768]", embedding_1: "f32[1, 128, 768]", getitem_1: "f32[256, 128, 1]", rsqrt: "f32[256, 128, 1]", gt: "b8[256, 128, 768]", ge: "b8[1, 1, 128, 1]", view: "f32[32768, 768]", bmm: "f32[3072, 128, 128]", amax: "f32[256, 12, 128, 1]", sum_1: "f32[256, 12, 128, 1]", logical_not_1: "b8[256, 12, 128, 1]", gt_1: "b8[256, 12, 128, 128]", view_16: "f32[32768, 768]", mul_8: "f32[256, 128, 768]", view_18: "f32[32768, 768]", addmm_4: "f32[32768, 3072]", view_20: "f32[32768, 3072]", gt_2: "b8[256, 128, 768]", mul_15: "f32[256, 128, 768]", view_22: "f32[32768, 768]", where_3: "f32[256, 12, 128, 128]", gt_3: "b8[256, 12, 128, 128]", view_38: "f32[32768, 768]", mul_21: "f32[256, 128, 768]", view_40: "f32[32768, 768]", addmm_10: "f32[32768, 3072]", view_42: "f32[32768, 3072]", gt_4: "b8[256, 128, 768]", mul_28: "f32[256, 128, 768]", view_44: "f32[32768, 768]", where_5: "f32[256, 12, 128, 128]", gt_5: "b8[256, 12, 128, 128]", view_60: "f32[32768, 768]", mul_34: "f32[256, 128, 768]", view_62: "f32[32768, 768]", addmm_16: "f32[32768, 3072]", view_64: "f32[32768, 3072]", gt_6: "b8[256, 128, 768]", mul_41: "f32[256, 128, 768]", view_66: "f32[32768, 768]", where_7: "f32[256, 12, 128, 128]", gt_7: "b8[256, 12, 128, 128]", view_82: "f32[32768, 768]", mul_47: "f32[256, 128, 768]", view_84: "f32[32768, 768]", addmm_22: "f32[32768, 3072]", view_86: "f32[32768, 3072]", gt_8: "b8[256, 128, 768]", mul_54: "f32[256, 128, 768]", view_88: "f32[32768, 768]", where_9: "f32[256, 12, 128, 128]", gt_9: "b8[256, 12, 128, 128]", view_104: "f32[32768, 768]", mul_60: "f32[256, 128, 768]", view_106: "f32[32768, 768]", addmm_28: "f32[32768, 3072]", view_108: "f32[32768, 3072]", gt_10: "b8[256, 128, 768]", mul_67: "f32[256, 128, 768]", view_110: "f32[32768, 768]", where_11: "f32[256, 12, 128, 128]", gt_11: "b8[256, 12, 128, 128]", view_126: "f32[32768, 768]", mul_73: "f32[256, 128, 768]", view_128: "f32[32768, 768]", addmm_34: "f32[32768, 3072]", view_130: "f32[32768, 3072]", gt_12: "b8[256, 128, 768]", mul_80: "f32[256, 128, 768]", view_132: "f32[32768, 768]", addmm_36: "f32[32768, 768]", getitem_27: "f32[256, 128, 1]", rsqrt_13: "f32[256, 128, 1]", view_134: "f32[32768, 768]", view_135: "f32[256, 128, 30522]", amax_6: "f32[32768, 1]", log: "f32[32768, 1]", convert_element_type: "f32[]", div_9: "f32[256, 128, 1]", div_10: "f32[256, 128, 1]", permute_89: "f32[3072, 128, 128]", permute_90: "f32[3072, 64, 128]", permute_91: "f32[3072, 64, 128]", permute_92: "f32[3072, 128, 64]", div_11: "f32[256, 128, 1]", div_12: "f32[256, 128, 1]", permute_122: "f32[3072, 128, 128]", permute_123: "f32[3072, 64, 128]", permute_124: "f32[3072, 64, 128]", permute_125: "f32[3072, 128, 64]", div_13: "f32[256, 128, 1]", div_14: "f32[256, 128, 1]", permute_155: "f32[3072, 128, 128]", permute_156: "f32[3072, 64, 128]", permute_157: "f32[3072, 64, 128]", permute_158: "f32[3072, 128, 64]", div_15: "f32[256, 128, 1]", div_16: "f32[256, 128, 1]", permute_188: "f32[3072, 128, 128]", permute_189: "f32[3072, 64, 128]", permute_190: "f32[3072, 64, 128]", permute_191: "f32[3072, 128, 64]", div_17: "f32[256, 128, 1]", div_18: "f32[256, 128, 1]", permute_221: "f32[3072, 128, 128]", permute_222: "f32[3072, 64, 128]", permute_223: "f32[3072, 64, 128]", permute_224: "f32[3072, 128, 64]", div_19: "f32[256, 128, 1]", div_20: "f32[256, 128, 1]", permute_254: "f32[3072, 128, 128]", permute_255: "f32[3072, 64, 128]", permute_256: "f32[3072, 64, 128]", permute_257: "f32[3072, 128, 64]", tangents_1: "f32[]", tangents_2: "f32[256, 128, 30522]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:521 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        div_7: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
        view_137: "i64[32768]" = torch.ops.aten.reshape.default(primals_108, [-1]);  primals_108 = None
        unsqueeze_4: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(view_137, 1);  view_137 = None
        ne_3: "b8[32768, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_4, -100)
        full_default_18: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "i64[32768, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_4, full_default_18);  unsqueeze_4 = full_default_18 = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 30522]" = torch.ops.aten.reshape.default(iota_default, [1, 30522]);  iota_default = None
        expand_default: "i64[32768, 30522]" = torch.ops.aten.expand.default(where_14, [32768, 30522]);  where_14 = None
        eq_tensor: "b8[32768, 30522]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:521 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        where_self: "f32[32768, 30522]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:521 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        where_15: "f32[32768, 1]" = torch.ops.aten.where.self(ne_3, div_7, full_default_1);  ne_3 = div_7 = None
        mul_87: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(where_self, where_15);  where_self = where_15 = None
        view_136: "f32[32768, 30522]" = torch.ops.aten.reshape.default(view_135, [-1, 30522]);  view_135 = None
        sub_20: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(view_136, amax_6);  view_136 = amax_6 = None
        sub_21: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_20, log);  sub_20 = log = None
        exp_7: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_10: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(mul_87, [1], True)
        mul_88: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(exp_7, sum_10);  exp_7 = sum_10 = None
        sub_22: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
        view_138: "f32[256, 128, 30522]" = torch.ops.aten.reshape.default(sub_22, [256, 128, 30522]);  sub_22 = None
        add_56: "f32[256, 128, 30522]" = torch.ops.aten.add.Tensor(tangents_2, view_138);  tangents_2 = view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        view_139: "f32[32768, 30522]" = torch.ops.aten.reshape.default(add_56, [32768, 30522]);  add_56 = None
        permute_67: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_68: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        constant_pad_nd_default_1: "f32[32768, 30524]" = torch.ops.aten.constant_pad_nd.default(view_139, [0, 2, 0, 0])
        constant_pad_nd_default_2: "f32[30524, 768]" = torch.ops.aten.constant_pad_nd.default(permute_68, [0, 0, 0, 2]);  permute_68 = None
        mm_default_1: "f32[32768, 768]" = torch.ops.aten.mm.default(constant_pad_nd_default_1, constant_pad_nd_default_2);  constant_pad_nd_default_1 = constant_pad_nd_default_2 = None
        permute_69: "f32[30522, 32768]" = torch.ops.aten.permute.default(view_139, [1, 0])
        constant_pad_nd_default: "f32[30524, 32768]" = torch.ops.aten.constant_pad_nd.default(permute_69, [0, 0, 0, 2]);  permute_69 = None
        mm_default: "f32[30524, 768]" = torch.ops.aten.mm.default(constant_pad_nd_default, view_134);  constant_pad_nd_default = view_134 = None
        slice_tensor: "f32[30522, 768]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -2);  mm_default = None
        sum_11: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_139, [0], True);  view_139 = None
        view_140: "f32[30522]" = torch.ops.aten.reshape.default(sum_11, [30522]);  sum_11 = None
        view_141: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_default_1, [256, 128, 768]);  mm_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:516 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        mul_90: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(view_141, primals_105);  primals_105 = None
        mul_91: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_90, 768)
        sum_12: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_90, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:514 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        view_133: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(addmm_36, [256, 128, 768]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_82: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(view_133, 0.5)
        mul_83: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(view_133, 0.7071067811865476)
        erf_6: "f32[256, 128, 768]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_53: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_84: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_82, add_53);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:516 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        sub_19: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_84, getitem_27);  mul_84 = getitem_27 = None
        mul_85: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = None
        mul_92: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_90, mul_85);  mul_90 = None
        sum_13: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_92, [2], True);  mul_92 = None
        mul_93: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_85, sum_13);  sum_13 = None
        sub_24: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_91, sum_12);  mul_91 = sum_12 = None
        sub_25: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_24, mul_93);  sub_24 = mul_93 = None
        div_8: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        mul_94: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_25);  div_8 = sub_25 = None
        mul_95: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(view_141, mul_85);  mul_85 = None
        sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_95, [0, 1]);  mul_95 = None
        sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_141, [0, 1]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_97: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_53, 0.5);  add_53 = None
        mul_98: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(view_133, view_133)
        mul_99: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_98, -0.5);  mul_98 = None
        exp_8: "f32[256, 128, 768]" = torch.ops.aten.exp.default(mul_99);  mul_99 = None
        mul_100: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_101: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(view_133, mul_100);  view_133 = mul_100 = None
        add_58: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_97, mul_101);  mul_97 = mul_101 = None
        mul_102: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_94, add_58);  mul_94 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:514 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        view_142: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_102, [32768, 768]);  mul_102 = None
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_2: "f32[32768, 768]" = torch.ops.aten.mm.default(view_142, permute_72);  permute_72 = None
        permute_73: "f32[768, 32768]" = torch.ops.aten.permute.default(view_142, [1, 0])
        mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_73, view_132);  permute_73 = view_132 = None
        sum_16: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_142, [0], True);  view_142 = None
        view_143: "f32[768]" = torch.ops.aten.reshape.default(sum_16, [768]);  sum_16 = None
        view_144: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_2, [256, 128, 768]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_104: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(view_144, primals_101);  primals_101 = None
        mul_105: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_104, 768)
        sum_17: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [2], True)
        mul_106: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_104, mul_80);  mul_104 = None
        sum_18: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_106, [2], True);  mul_106 = None
        mul_107: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_18);  sum_18 = None
        sub_27: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_105, sum_17);  mul_105 = sum_17 = None
        sub_28: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_27, mul_107);  sub_27 = mul_107 = None
        mul_108: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_28);  div_9 = sub_28 = None
        mul_109: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(view_144, mul_80);  mul_80 = None
        sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_109, [0, 1]);  mul_109 = None
        sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_144, [0, 1]);  view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        convert_element_type_1: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_110: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_111: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_108, mul_110);  mul_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_145: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_111, [32768, 768]);  mul_111 = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_4: "f32[32768, 3072]" = torch.ops.aten.mm.default(view_145, permute_76);  permute_76 = None
        permute_77: "f32[768, 32768]" = torch.ops.aten.permute.default(view_145, [1, 0])
        mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_77, view_130);  permute_77 = view_130 = None
        sum_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_145, [0], True);  view_145 = None
        view_146: "f32[768]" = torch.ops.aten.reshape.default(sum_21, [768]);  sum_21 = None
        view_147: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(mm_4, [256, 128, 3072]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_129: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(addmm_34, [256, 128, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_76: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
        erf_5: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_49: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_113: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
        mul_114: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
        mul_115: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_114, -0.5);  mul_114 = None
        exp_9: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_115);  mul_115 = None
        mul_116: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_117: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_116);  view_129 = mul_116 = None
        add_60: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_113, mul_117);  mul_113 = mul_117 = None
        mul_118: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_147, add_60);  view_147 = add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_148: "f32[32768, 3072]" = torch.ops.aten.reshape.default(mul_118, [32768, 3072]);  mul_118 = None
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_6: "f32[32768, 768]" = torch.ops.aten.mm.default(view_148, permute_80);  permute_80 = None
        permute_81: "f32[3072, 32768]" = torch.ops.aten.permute.default(view_148, [1, 0])
        mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_81, view_128);  permute_81 = view_128 = None
        sum_22: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_148, [0], True);  view_148 = None
        view_149: "f32[3072]" = torch.ops.aten.reshape.default(sum_22, [3072]);  sum_22 = None
        view_150: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_6, [256, 128, 768]);  mm_6 = None
        add_61: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_108, view_150);  mul_108 = view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_120: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_61, primals_95);  primals_95 = None
        mul_121: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_120, 768)
        sum_23: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_120, [2], True)
        mul_122: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_120, mul_73);  mul_120 = None
        sum_24: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True);  mul_122 = None
        mul_123: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_73, sum_24);  sum_24 = None
        sub_30: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_121, sum_23);  mul_121 = sum_23 = None
        sub_31: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_30, mul_123);  sub_30 = mul_123 = None
        mul_124: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_31);  div_10 = sub_31 = None
        mul_125: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_61, mul_73);  mul_73 = None
        sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_125, [0, 1]);  mul_125 = None
        sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_61, [0, 1]);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_151: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_124, [32768, 768])
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_84: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_8: "f32[32768, 768]" = torch.ops.aten.mm.default(view_151, permute_84);  permute_84 = None
        permute_85: "f32[768, 32768]" = torch.ops.aten.permute.default(view_151, [1, 0])
        mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_85, view_126);  permute_85 = view_126 = None
        sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_151, [0], True);  view_151 = None
        view_152: "f32[768]" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        view_153: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_8, [256, 128, 768]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_154: "f32[256, 128, 12, 64]" = torch.ops.aten.reshape.default(view_153, [256, 128, 12, 64]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_88: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_25: "f32[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
        view_155: "f32[3072, 128, 64]" = torch.ops.aten.reshape.default(clone_25, [3072, 128, 64]);  clone_25 = None
        bmm_12: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(permute_89, view_155);  permute_89 = None
        bmm_13: "f32[3072, 128, 128]" = torch.ops.aten.bmm.default(view_155, permute_90);  view_155 = permute_90 = None
        view_156: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_12, [256, 12, 128, 64]);  bmm_12 = None
        view_157: "f32[256, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_13, [256, 12, 128, 128]);  bmm_13 = None
        convert_element_type_2: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_126: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_127: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_157, mul_126);  view_157 = mul_126 = None
        mul_128: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_127, where_11);  mul_127 = None
        sum_28: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_128, [-1], True)
        neg_1: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(where_11);  where_11 = None
        fma: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_1, sum_28, mul_128);  neg_1 = sum_28 = mul_128 = None
        view_158: "f32[3072, 128, 128]" = torch.ops.aten.reshape.default(fma, [3072, 128, 128]);  fma = None
        bmm_14: "f32[3072, 64, 128]" = torch.ops.aten.bmm.default(permute_91, view_158);  permute_91 = None
        bmm_15: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(view_158, permute_92);  view_158 = permute_92 = None
        view_159: "f32[256, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_14, [256, 12, 64, 128]);  bmm_14 = None
        view_160: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_15, [256, 12, 128, 64]);  bmm_15 = None
        mul_129: "f32[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_159, 0.3535533905932738);  view_159 = None
        permute_93: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_129, [0, 1, 3, 2]);  mul_129 = None
        mul_130: "f32[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_160, 0.3535533905932738);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_94: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        clone_27: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        view_161: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_27, [256, 128, 768]);  clone_27 = None
        view_162: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_161, [32768, 768]);  view_161 = None
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_95: "f32[768, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_10: "f32[32768, 768]" = torch.ops.aten.mm.default(view_162, permute_95);  permute_95 = None
        permute_96: "f32[768, 32768]" = torch.ops.aten.permute.default(view_162, [1, 0])
        mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_96, view_110);  permute_96 = None
        sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_162, [0], True);  view_162 = None
        view_163: "f32[768]" = torch.ops.aten.reshape.default(sum_29, [768]);  sum_29 = None
        view_164: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_10, [256, 128, 768]);  mm_10 = None
        add_62: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_124, view_164);  mul_124 = view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_99: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_93, [0, 2, 1, 3]);  permute_93 = None
        view_165: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(permute_99, [256, 128, 768]);  permute_99 = None
        clone_28: "f32[256, 128, 768]" = torch.ops.aten.clone.default(view_165, memory_format = torch.contiguous_format);  view_165 = None
        view_166: "f32[32768, 768]" = torch.ops.aten.reshape.default(clone_28, [32768, 768]);  clone_28 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_12: "f32[32768, 768]" = torch.ops.aten.mm.default(view_166, permute_100);  permute_100 = None
        permute_101: "f32[768, 32768]" = torch.ops.aten.permute.default(view_166, [1, 0])
        mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_101, view_110);  permute_101 = None
        sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_166, [0], True);  view_166 = None
        view_167: "f32[768]" = torch.ops.aten.reshape.default(sum_30, [768]);  sum_30 = None
        view_168: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_12, [256, 128, 768]);  mm_12 = None
        add_63: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_62, view_168);  add_62 = view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_104: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_130, [0, 2, 1, 3]);  mul_130 = None
        clone_29: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_169: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_29, [256, 128, 768]);  clone_29 = None
        view_170: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_169, [32768, 768]);  view_169 = None
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_105: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_14: "f32[32768, 768]" = torch.ops.aten.mm.default(view_170, permute_105);  permute_105 = None
        permute_106: "f32[768, 32768]" = torch.ops.aten.permute.default(view_170, [1, 0])
        mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_106, view_110);  permute_106 = view_110 = None
        sum_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
        view_171: "f32[768]" = torch.ops.aten.reshape.default(sum_31, [768]);  sum_31 = None
        view_172: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_14, [256, 128, 768]);  mm_14 = None
        add_64: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_63, view_172);  add_63 = view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_132: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_64, primals_85);  primals_85 = None
        mul_133: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_132, 768)
        sum_32: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_132, [2], True)
        mul_134: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_132, mul_67);  mul_132 = None
        sum_33: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_134, [2], True);  mul_134 = None
        mul_135: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_67, sum_33);  sum_33 = None
        sub_33: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_133, sum_32);  mul_133 = sum_32 = None
        sub_34: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_33, mul_135);  sub_33 = mul_135 = None
        mul_136: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_34);  div_11 = sub_34 = None
        mul_137: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_64, mul_67);  mul_67 = None
        sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_137, [0, 1]);  mul_137 = None
        sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_64, [0, 1]);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        convert_element_type_3: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_138: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_139: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_136, mul_138);  mul_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_173: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_139, [32768, 768]);  mul_139 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_16: "f32[32768, 3072]" = torch.ops.aten.mm.default(view_173, permute_109);  permute_109 = None
        permute_110: "f32[768, 32768]" = torch.ops.aten.permute.default(view_173, [1, 0])
        mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_110, view_108);  permute_110 = view_108 = None
        sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
        view_174: "f32[768]" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
        view_175: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(mm_16, [256, 128, 3072]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_107: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(addmm_28, [256, 128, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_63: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
        erf_4: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_41: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_141: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
        mul_142: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
        mul_143: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_142, -0.5);  mul_142 = None
        exp_10: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_143);  mul_143 = None
        mul_144: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_145: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_144);  view_107 = mul_144 = None
        add_66: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_141, mul_145);  mul_141 = mul_145 = None
        mul_146: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_175, add_66);  view_175 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_176: "f32[32768, 3072]" = torch.ops.aten.reshape.default(mul_146, [32768, 3072]);  mul_146 = None
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        permute_113: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_18: "f32[32768, 768]" = torch.ops.aten.mm.default(view_176, permute_113);  permute_113 = None
        permute_114: "f32[3072, 32768]" = torch.ops.aten.permute.default(view_176, [1, 0])
        mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_114, view_106);  permute_114 = view_106 = None
        sum_37: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_176, [0], True);  view_176 = None
        view_177: "f32[3072]" = torch.ops.aten.reshape.default(sum_37, [3072]);  sum_37 = None
        view_178: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_18, [256, 128, 768]);  mm_18 = None
        add_67: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_136, view_178);  mul_136 = view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_148: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_67, primals_79);  primals_79 = None
        mul_149: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_148, 768)
        sum_38: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True)
        mul_150: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_148, mul_60);  mul_148 = None
        sum_39: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
        mul_151: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_60, sum_39);  sum_39 = None
        sub_36: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_149, sum_38);  mul_149 = sum_38 = None
        sub_37: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_36, mul_151);  sub_36 = mul_151 = None
        mul_152: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_37);  div_12 = sub_37 = None
        mul_153: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_67, mul_60);  mul_60 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1]);  mul_153 = None
        sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_67, [0, 1]);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_179: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_152, [32768, 768])
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_117: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_20: "f32[32768, 768]" = torch.ops.aten.mm.default(view_179, permute_117);  permute_117 = None
        permute_118: "f32[768, 32768]" = torch.ops.aten.permute.default(view_179, [1, 0])
        mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_118, view_104);  permute_118 = view_104 = None
        sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_179, [0], True);  view_179 = None
        view_180: "f32[768]" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        view_181: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_20, [256, 128, 768]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_182: "f32[256, 128, 12, 64]" = torch.ops.aten.reshape.default(view_181, [256, 128, 12, 64]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_121: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_31: "f32[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        view_183: "f32[3072, 128, 64]" = torch.ops.aten.reshape.default(clone_31, [3072, 128, 64]);  clone_31 = None
        bmm_16: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(permute_122, view_183);  permute_122 = None
        bmm_17: "f32[3072, 128, 128]" = torch.ops.aten.bmm.default(view_183, permute_123);  view_183 = permute_123 = None
        view_184: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_16, [256, 12, 128, 64]);  bmm_16 = None
        view_185: "f32[256, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_17, [256, 12, 128, 128]);  bmm_17 = None
        convert_element_type_4: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_154: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_155: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_185, mul_154);  view_185 = mul_154 = None
        mul_156: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_155, where_9);  mul_155 = None
        sum_43: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [-1], True)
        neg_2: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(where_9);  where_9 = None
        fma_1: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_2, sum_43, mul_156);  neg_2 = sum_43 = mul_156 = None
        view_186: "f32[3072, 128, 128]" = torch.ops.aten.reshape.default(fma_1, [3072, 128, 128]);  fma_1 = None
        bmm_18: "f32[3072, 64, 128]" = torch.ops.aten.bmm.default(permute_124, view_186);  permute_124 = None
        bmm_19: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(view_186, permute_125);  view_186 = permute_125 = None
        view_187: "f32[256, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_18, [256, 12, 64, 128]);  bmm_18 = None
        view_188: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_19, [256, 12, 128, 64]);  bmm_19 = None
        mul_157: "f32[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_187, 0.3535533905932738);  view_187 = None
        permute_126: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_157, [0, 1, 3, 2]);  mul_157 = None
        mul_158: "f32[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_188, 0.3535533905932738);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_127: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        clone_33: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_189: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_33, [256, 128, 768]);  clone_33 = None
        view_190: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_189, [32768, 768]);  view_189 = None
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_128: "f32[768, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_22: "f32[32768, 768]" = torch.ops.aten.mm.default(view_190, permute_128);  permute_128 = None
        permute_129: "f32[768, 32768]" = torch.ops.aten.permute.default(view_190, [1, 0])
        mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_129, view_88);  permute_129 = None
        sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_190, [0], True);  view_190 = None
        view_191: "f32[768]" = torch.ops.aten.reshape.default(sum_44, [768]);  sum_44 = None
        view_192: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_22, [256, 128, 768]);  mm_22 = None
        add_68: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_152, view_192);  mul_152 = view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_132: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_126, [0, 2, 1, 3]);  permute_126 = None
        view_193: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(permute_132, [256, 128, 768]);  permute_132 = None
        clone_34: "f32[256, 128, 768]" = torch.ops.aten.clone.default(view_193, memory_format = torch.contiguous_format);  view_193 = None
        view_194: "f32[32768, 768]" = torch.ops.aten.reshape.default(clone_34, [32768, 768]);  clone_34 = None
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_24: "f32[32768, 768]" = torch.ops.aten.mm.default(view_194, permute_133);  permute_133 = None
        permute_134: "f32[768, 32768]" = torch.ops.aten.permute.default(view_194, [1, 0])
        mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_134, view_88);  permute_134 = None
        sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_194, [0], True);  view_194 = None
        view_195: "f32[768]" = torch.ops.aten.reshape.default(sum_45, [768]);  sum_45 = None
        view_196: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_24, [256, 128, 768]);  mm_24 = None
        add_69: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_68, view_196);  add_68 = view_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_137: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_158, [0, 2, 1, 3]);  mul_158 = None
        clone_35: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_197: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_35, [256, 128, 768]);  clone_35 = None
        view_198: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_197, [32768, 768]);  view_197 = None
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_26: "f32[32768, 768]" = torch.ops.aten.mm.default(view_198, permute_138);  permute_138 = None
        permute_139: "f32[768, 32768]" = torch.ops.aten.permute.default(view_198, [1, 0])
        mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_139, view_88);  permute_139 = view_88 = None
        sum_46: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
        view_199: "f32[768]" = torch.ops.aten.reshape.default(sum_46, [768]);  sum_46 = None
        view_200: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_26, [256, 128, 768]);  mm_26 = None
        add_70: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_69, view_200);  add_69 = view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_160: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_70, primals_69);  primals_69 = None
        mul_161: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_160, 768)
        sum_47: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_160, [2], True)
        mul_162: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_160, mul_54);  mul_160 = None
        sum_48: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_162, [2], True);  mul_162 = None
        mul_163: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_54, sum_48);  sum_48 = None
        sub_39: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_161, sum_47);  mul_161 = sum_47 = None
        sub_40: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_39, mul_163);  sub_39 = mul_163 = None
        mul_164: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_40);  div_13 = sub_40 = None
        mul_165: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_70, mul_54);  mul_54 = None
        sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_165, [0, 1]);  mul_165 = None
        sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_70, [0, 1]);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        convert_element_type_5: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_166: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_167: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_164, mul_166);  mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_201: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_167, [32768, 768]);  mul_167 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_28: "f32[32768, 3072]" = torch.ops.aten.mm.default(view_201, permute_142);  permute_142 = None
        permute_143: "f32[768, 32768]" = torch.ops.aten.permute.default(view_201, [1, 0])
        mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_143, view_86);  permute_143 = view_86 = None
        sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
        view_202: "f32[768]" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
        view_203: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(mm_28, [256, 128, 3072]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_85: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(addmm_22, [256, 128, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_50: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
        erf_3: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_33: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_169: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_33, 0.5);  add_33 = None
        mul_170: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
        mul_171: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_170, -0.5);  mul_170 = None
        exp_11: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_171);  mul_171 = None
        mul_172: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_173: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_172);  view_85 = mul_172 = None
        add_72: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_169, mul_173);  mul_169 = mul_173 = None
        mul_174: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_203, add_72);  view_203 = add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_204: "f32[32768, 3072]" = torch.ops.aten.reshape.default(mul_174, [32768, 3072]);  mul_174 = None
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        permute_146: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_30: "f32[32768, 768]" = torch.ops.aten.mm.default(view_204, permute_146);  permute_146 = None
        permute_147: "f32[3072, 32768]" = torch.ops.aten.permute.default(view_204, [1, 0])
        mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_147, view_84);  permute_147 = view_84 = None
        sum_52: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
        view_205: "f32[3072]" = torch.ops.aten.reshape.default(sum_52, [3072]);  sum_52 = None
        view_206: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_30, [256, 128, 768]);  mm_30 = None
        add_73: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_164, view_206);  mul_164 = view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_176: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_73, primals_63);  primals_63 = None
        mul_177: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_176, 768)
        sum_53: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True)
        mul_178: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_176, mul_47);  mul_176 = None
        sum_54: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_178, [2], True);  mul_178 = None
        mul_179: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_47, sum_54);  sum_54 = None
        sub_42: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_177, sum_53);  mul_177 = sum_53 = None
        sub_43: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_42, mul_179);  sub_42 = mul_179 = None
        mul_180: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_43);  div_14 = sub_43 = None
        mul_181: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_73, mul_47);  mul_47 = None
        sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_181, [0, 1]);  mul_181 = None
        sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_73, [0, 1]);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_207: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_180, [32768, 768])
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_150: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_32: "f32[32768, 768]" = torch.ops.aten.mm.default(view_207, permute_150);  permute_150 = None
        permute_151: "f32[768, 32768]" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_151, view_82);  permute_151 = view_82 = None
        sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
        view_208: "f32[768]" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        view_209: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_32, [256, 128, 768]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_210: "f32[256, 128, 12, 64]" = torch.ops.aten.reshape.default(view_209, [256, 128, 12, 64]);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_154: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_37: "f32[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_211: "f32[3072, 128, 64]" = torch.ops.aten.reshape.default(clone_37, [3072, 128, 64]);  clone_37 = None
        bmm_20: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(permute_155, view_211);  permute_155 = None
        bmm_21: "f32[3072, 128, 128]" = torch.ops.aten.bmm.default(view_211, permute_156);  view_211 = permute_156 = None
        view_212: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_20, [256, 12, 128, 64]);  bmm_20 = None
        view_213: "f32[256, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_21, [256, 12, 128, 128]);  bmm_21 = None
        convert_element_type_6: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_182: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_183: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_213, mul_182);  view_213 = mul_182 = None
        mul_184: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_183, where_7);  mul_183 = None
        sum_58: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_184, [-1], True)
        neg_3: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(where_7);  where_7 = None
        fma_2: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_3, sum_58, mul_184);  neg_3 = sum_58 = mul_184 = None
        view_214: "f32[3072, 128, 128]" = torch.ops.aten.reshape.default(fma_2, [3072, 128, 128]);  fma_2 = None
        bmm_22: "f32[3072, 64, 128]" = torch.ops.aten.bmm.default(permute_157, view_214);  permute_157 = None
        bmm_23: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(view_214, permute_158);  view_214 = permute_158 = None
        view_215: "f32[256, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_22, [256, 12, 64, 128]);  bmm_22 = None
        view_216: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_23, [256, 12, 128, 64]);  bmm_23 = None
        mul_185: "f32[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_215, 0.3535533905932738);  view_215 = None
        permute_159: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_185, [0, 1, 3, 2]);  mul_185 = None
        mul_186: "f32[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_216, 0.3535533905932738);  view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_160: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_39: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_217: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_39, [256, 128, 768]);  clone_39 = None
        view_218: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_217, [32768, 768]);  view_217 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_34: "f32[32768, 768]" = torch.ops.aten.mm.default(view_218, permute_161);  permute_161 = None
        permute_162: "f32[768, 32768]" = torch.ops.aten.permute.default(view_218, [1, 0])
        mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_162, view_66);  permute_162 = None
        sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
        view_219: "f32[768]" = torch.ops.aten.reshape.default(sum_59, [768]);  sum_59 = None
        view_220: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_34, [256, 128, 768]);  mm_34 = None
        add_74: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_180, view_220);  mul_180 = view_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_165: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_159, [0, 2, 1, 3]);  permute_159 = None
        view_221: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(permute_165, [256, 128, 768]);  permute_165 = None
        clone_40: "f32[256, 128, 768]" = torch.ops.aten.clone.default(view_221, memory_format = torch.contiguous_format);  view_221 = None
        view_222: "f32[32768, 768]" = torch.ops.aten.reshape.default(clone_40, [32768, 768]);  clone_40 = None
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_36: "f32[32768, 768]" = torch.ops.aten.mm.default(view_222, permute_166);  permute_166 = None
        permute_167: "f32[768, 32768]" = torch.ops.aten.permute.default(view_222, [1, 0])
        mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_167, view_66);  permute_167 = None
        sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
        view_223: "f32[768]" = torch.ops.aten.reshape.default(sum_60, [768]);  sum_60 = None
        view_224: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_36, [256, 128, 768]);  mm_36 = None
        add_75: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_74, view_224);  add_74 = view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_170: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_186, [0, 2, 1, 3]);  mul_186 = None
        clone_41: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_225: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_41, [256, 128, 768]);  clone_41 = None
        view_226: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_225, [32768, 768]);  view_225 = None
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_38: "f32[32768, 768]" = torch.ops.aten.mm.default(view_226, permute_171);  permute_171 = None
        permute_172: "f32[768, 32768]" = torch.ops.aten.permute.default(view_226, [1, 0])
        mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_172, view_66);  permute_172 = view_66 = None
        sum_61: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
        view_227: "f32[768]" = torch.ops.aten.reshape.default(sum_61, [768]);  sum_61 = None
        view_228: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_38, [256, 128, 768]);  mm_38 = None
        add_76: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_75, view_228);  add_75 = view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_188: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_76, primals_53);  primals_53 = None
        mul_189: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_188, 768)
        sum_62: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True)
        mul_190: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_188, mul_41);  mul_188 = None
        sum_63: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_190, [2], True);  mul_190 = None
        mul_191: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_41, sum_63);  sum_63 = None
        sub_45: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_189, sum_62);  mul_189 = sum_62 = None
        sub_46: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_191);  sub_45 = mul_191 = None
        mul_192: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_46);  div_15 = sub_46 = None
        mul_193: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_76, mul_41);  mul_41 = None
        sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_193, [0, 1]);  mul_193 = None
        sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_76, [0, 1]);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        convert_element_type_7: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_194: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_195: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_194);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_229: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_195, [32768, 768]);  mul_195 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_175: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_40: "f32[32768, 3072]" = torch.ops.aten.mm.default(view_229, permute_175);  permute_175 = None
        permute_176: "f32[768, 32768]" = torch.ops.aten.permute.default(view_229, [1, 0])
        mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_176, view_64);  permute_176 = view_64 = None
        sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_229, [0], True);  view_229 = None
        view_230: "f32[768]" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
        view_231: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(mm_40, [256, 128, 3072]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_63: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(addmm_16, [256, 128, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_37: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
        erf_2: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
        add_25: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_197: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_25, 0.5);  add_25 = None
        mul_198: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
        mul_199: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_198, -0.5);  mul_198 = None
        exp_12: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_199);  mul_199 = None
        mul_200: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_201: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_200);  view_63 = mul_200 = None
        add_78: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_197, mul_201);  mul_197 = mul_201 = None
        mul_202: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_231, add_78);  view_231 = add_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_232: "f32[32768, 3072]" = torch.ops.aten.reshape.default(mul_202, [32768, 3072]);  mul_202 = None
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        permute_179: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_42: "f32[32768, 768]" = torch.ops.aten.mm.default(view_232, permute_179);  permute_179 = None
        permute_180: "f32[3072, 32768]" = torch.ops.aten.permute.default(view_232, [1, 0])
        mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_180, view_62);  permute_180 = view_62 = None
        sum_67: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_232, [0], True);  view_232 = None
        view_233: "f32[3072]" = torch.ops.aten.reshape.default(sum_67, [3072]);  sum_67 = None
        view_234: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_42, [256, 128, 768]);  mm_42 = None
        add_79: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_192, view_234);  mul_192 = view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_204: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_79, primals_47);  primals_47 = None
        mul_205: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_204, 768)
        sum_68: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True)
        mul_206: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_204, mul_34);  mul_204 = None
        sum_69: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True);  mul_206 = None
        mul_207: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_34, sum_69);  sum_69 = None
        sub_48: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_205, sum_68);  mul_205 = sum_68 = None
        sub_49: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_207);  sub_48 = mul_207 = None
        mul_208: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_49);  div_16 = sub_49 = None
        mul_209: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_79, mul_34);  mul_34 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_209, [0, 1]);  mul_209 = None
        sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_79, [0, 1]);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_235: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_208, [32768, 768])
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_44: "f32[32768, 768]" = torch.ops.aten.mm.default(view_235, permute_183);  permute_183 = None
        permute_184: "f32[768, 32768]" = torch.ops.aten.permute.default(view_235, [1, 0])
        mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_184, view_60);  permute_184 = view_60 = None
        sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_235, [0], True);  view_235 = None
        view_236: "f32[768]" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        view_237: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_44, [256, 128, 768]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_238: "f32[256, 128, 12, 64]" = torch.ops.aten.reshape.default(view_237, [256, 128, 12, 64]);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_187: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_43: "f32[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_239: "f32[3072, 128, 64]" = torch.ops.aten.reshape.default(clone_43, [3072, 128, 64]);  clone_43 = None
        bmm_24: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(permute_188, view_239);  permute_188 = None
        bmm_25: "f32[3072, 128, 128]" = torch.ops.aten.bmm.default(view_239, permute_189);  view_239 = permute_189 = None
        view_240: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_24, [256, 12, 128, 64]);  bmm_24 = None
        view_241: "f32[256, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_25, [256, 12, 128, 128]);  bmm_25 = None
        convert_element_type_8: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_210: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_211: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_241, mul_210);  view_241 = mul_210 = None
        mul_212: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_211, where_5);  mul_211 = None
        sum_73: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_212, [-1], True)
        neg_4: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(where_5);  where_5 = None
        fma_3: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_4, sum_73, mul_212);  neg_4 = sum_73 = mul_212 = None
        view_242: "f32[3072, 128, 128]" = torch.ops.aten.reshape.default(fma_3, [3072, 128, 128]);  fma_3 = None
        bmm_26: "f32[3072, 64, 128]" = torch.ops.aten.bmm.default(permute_190, view_242);  permute_190 = None
        bmm_27: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(view_242, permute_191);  view_242 = permute_191 = None
        view_243: "f32[256, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_26, [256, 12, 64, 128]);  bmm_26 = None
        view_244: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_27, [256, 12, 128, 64]);  bmm_27 = None
        mul_213: "f32[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_243, 0.3535533905932738);  view_243 = None
        permute_192: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_213, [0, 1, 3, 2]);  mul_213 = None
        mul_214: "f32[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_244, 0.3535533905932738);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_193: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
        clone_45: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_245: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_45, [256, 128, 768]);  clone_45 = None
        view_246: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_245, [32768, 768]);  view_245 = None
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_46: "f32[32768, 768]" = torch.ops.aten.mm.default(view_246, permute_194);  permute_194 = None
        permute_195: "f32[768, 32768]" = torch.ops.aten.permute.default(view_246, [1, 0])
        mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_195, view_44);  permute_195 = None
        sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
        view_247: "f32[768]" = torch.ops.aten.reshape.default(sum_74, [768]);  sum_74 = None
        view_248: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_46, [256, 128, 768]);  mm_46 = None
        add_80: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_208, view_248);  mul_208 = view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_198: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
        view_249: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(permute_198, [256, 128, 768]);  permute_198 = None
        clone_46: "f32[256, 128, 768]" = torch.ops.aten.clone.default(view_249, memory_format = torch.contiguous_format);  view_249 = None
        view_250: "f32[32768, 768]" = torch.ops.aten.reshape.default(clone_46, [32768, 768]);  clone_46 = None
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_48: "f32[32768, 768]" = torch.ops.aten.mm.default(view_250, permute_199);  permute_199 = None
        permute_200: "f32[768, 32768]" = torch.ops.aten.permute.default(view_250, [1, 0])
        mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_44);  permute_200 = None
        sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True);  view_250 = None
        view_251: "f32[768]" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None
        view_252: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_48, [256, 128, 768]);  mm_48 = None
        add_81: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_80, view_252);  add_80 = view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_203: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_214, [0, 2, 1, 3]);  mul_214 = None
        clone_47: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_253: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_47, [256, 128, 768]);  clone_47 = None
        view_254: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_253, [32768, 768]);  view_253 = None
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_50: "f32[32768, 768]" = torch.ops.aten.mm.default(view_254, permute_204);  permute_204 = None
        permute_205: "f32[768, 32768]" = torch.ops.aten.permute.default(view_254, [1, 0])
        mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_205, view_44);  permute_205 = view_44 = None
        sum_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
        view_255: "f32[768]" = torch.ops.aten.reshape.default(sum_76, [768]);  sum_76 = None
        view_256: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_50, [256, 128, 768]);  mm_50 = None
        add_82: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_81, view_256);  add_81 = view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_216: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_82, primals_37);  primals_37 = None
        mul_217: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_216, 768)
        sum_77: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_216, [2], True)
        mul_218: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_216, mul_28);  mul_216 = None
        sum_78: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_218, [2], True);  mul_218 = None
        mul_219: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_28, sum_78);  sum_78 = None
        sub_51: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_217, sum_77);  mul_217 = sum_77 = None
        sub_52: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_219);  sub_51 = mul_219 = None
        mul_220: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_52);  div_17 = sub_52 = None
        mul_221: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_82, mul_28);  mul_28 = None
        sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_221, [0, 1]);  mul_221 = None
        sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_82, [0, 1]);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        convert_element_type_9: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_222: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_223: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_220, mul_222);  mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_257: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_223, [32768, 768]);  mul_223 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_52: "f32[32768, 3072]" = torch.ops.aten.mm.default(view_257, permute_208);  permute_208 = None
        permute_209: "f32[768, 32768]" = torch.ops.aten.permute.default(view_257, [1, 0])
        mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_209, view_42);  permute_209 = view_42 = None
        sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_257, [0], True);  view_257 = None
        view_258: "f32[768]" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
        view_259: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(mm_52, [256, 128, 3072]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_41: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(addmm_10, [256, 128, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_24: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
        erf_1: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_17: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_225: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.5);  add_17 = None
        mul_226: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
        mul_227: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_226, -0.5);  mul_226 = None
        exp_13: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_227);  mul_227 = None
        mul_228: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_229: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_228);  view_41 = mul_228 = None
        add_84: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_225, mul_229);  mul_225 = mul_229 = None
        mul_230: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_259, add_84);  view_259 = add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_260: "f32[32768, 3072]" = torch.ops.aten.reshape.default(mul_230, [32768, 3072]);  mul_230 = None
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_54: "f32[32768, 768]" = torch.ops.aten.mm.default(view_260, permute_212);  permute_212 = None
        permute_213: "f32[3072, 32768]" = torch.ops.aten.permute.default(view_260, [1, 0])
        mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_213, view_40);  permute_213 = view_40 = None
        sum_82: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True);  view_260 = None
        view_261: "f32[3072]" = torch.ops.aten.reshape.default(sum_82, [3072]);  sum_82 = None
        view_262: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_54, [256, 128, 768]);  mm_54 = None
        add_85: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_220, view_262);  mul_220 = view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_232: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_85, primals_31);  primals_31 = None
        mul_233: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_232, 768)
        sum_83: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_232, [2], True)
        mul_234: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_232, mul_21);  mul_232 = None
        sum_84: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True);  mul_234 = None
        mul_235: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_21, sum_84);  sum_84 = None
        sub_54: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_233, sum_83);  mul_233 = sum_83 = None
        sub_55: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_54, mul_235);  sub_54 = mul_235 = None
        mul_236: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_55);  div_18 = sub_55 = None
        mul_237: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_85, mul_21);  mul_21 = None
        sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_237, [0, 1]);  mul_237 = None
        sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_85, [0, 1]);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_263: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_236, [32768, 768])
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_56: "f32[32768, 768]" = torch.ops.aten.mm.default(view_263, permute_216);  permute_216 = None
        permute_217: "f32[768, 32768]" = torch.ops.aten.permute.default(view_263, [1, 0])
        mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_38);  permute_217 = view_38 = None
        sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
        view_264: "f32[768]" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        view_265: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_56, [256, 128, 768]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_266: "f32[256, 128, 12, 64]" = torch.ops.aten.reshape.default(view_265, [256, 128, 12, 64]);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_220: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_49: "f32[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_267: "f32[3072, 128, 64]" = torch.ops.aten.reshape.default(clone_49, [3072, 128, 64]);  clone_49 = None
        bmm_28: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(permute_221, view_267);  permute_221 = None
        bmm_29: "f32[3072, 128, 128]" = torch.ops.aten.bmm.default(view_267, permute_222);  view_267 = permute_222 = None
        view_268: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_28, [256, 12, 128, 64]);  bmm_28 = None
        view_269: "f32[256, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_29, [256, 12, 128, 128]);  bmm_29 = None
        convert_element_type_10: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_238: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_239: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_269, mul_238);  view_269 = mul_238 = None
        mul_240: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_239, where_3);  mul_239 = None
        sum_88: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [-1], True)
        neg_5: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(where_3);  where_3 = None
        fma_4: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_5, sum_88, mul_240);  neg_5 = sum_88 = mul_240 = None
        view_270: "f32[3072, 128, 128]" = torch.ops.aten.reshape.default(fma_4, [3072, 128, 128]);  fma_4 = None
        bmm_30: "f32[3072, 64, 128]" = torch.ops.aten.bmm.default(permute_223, view_270);  permute_223 = None
        bmm_31: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(view_270, permute_224);  view_270 = permute_224 = None
        view_271: "f32[256, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_30, [256, 12, 64, 128]);  bmm_30 = None
        view_272: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_31, [256, 12, 128, 64]);  bmm_31 = None
        mul_241: "f32[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_271, 0.3535533905932738);  view_271 = None
        permute_225: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_241, [0, 1, 3, 2]);  mul_241 = None
        mul_242: "f32[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_272, 0.3535533905932738);  view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_226: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        clone_51: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_273: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_51, [256, 128, 768]);  clone_51 = None
        view_274: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_273, [32768, 768]);  view_273 = None
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_58: "f32[32768, 768]" = torch.ops.aten.mm.default(view_274, permute_227);  permute_227 = None
        permute_228: "f32[768, 32768]" = torch.ops.aten.permute.default(view_274, [1, 0])
        mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_228, view_22);  permute_228 = None
        sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
        view_275: "f32[768]" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None
        view_276: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_58, [256, 128, 768]);  mm_58 = None
        add_86: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_236, view_276);  mul_236 = view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_231: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_225, [0, 2, 1, 3]);  permute_225 = None
        view_277: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(permute_231, [256, 128, 768]);  permute_231 = None
        clone_52: "f32[256, 128, 768]" = torch.ops.aten.clone.default(view_277, memory_format = torch.contiguous_format);  view_277 = None
        view_278: "f32[32768, 768]" = torch.ops.aten.reshape.default(clone_52, [32768, 768]);  clone_52 = None
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_60: "f32[32768, 768]" = torch.ops.aten.mm.default(view_278, permute_232);  permute_232 = None
        permute_233: "f32[768, 32768]" = torch.ops.aten.permute.default(view_278, [1, 0])
        mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_233, view_22);  permute_233 = None
        sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
        view_279: "f32[768]" = torch.ops.aten.reshape.default(sum_90, [768]);  sum_90 = None
        view_280: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_60, [256, 128, 768]);  mm_60 = None
        add_87: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_86, view_280);  add_86 = view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_236: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_242, [0, 2, 1, 3]);  mul_242 = None
        clone_53: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_281: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_53, [256, 128, 768]);  clone_53 = None
        view_282: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_281, [32768, 768]);  view_281 = None
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_62: "f32[32768, 768]" = torch.ops.aten.mm.default(view_282, permute_237);  permute_237 = None
        permute_238: "f32[768, 32768]" = torch.ops.aten.permute.default(view_282, [1, 0])
        mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_238, view_22);  permute_238 = view_22 = None
        sum_91: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        view_283: "f32[768]" = torch.ops.aten.reshape.default(sum_91, [768]);  sum_91 = None
        view_284: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_62, [256, 128, 768]);  mm_62 = None
        add_88: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_87, view_284);  add_87 = view_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_244: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_88, primals_21);  primals_21 = None
        mul_245: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_244, 768)
        sum_92: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [2], True)
        mul_246: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_244, mul_15);  mul_244 = None
        sum_93: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_246, [2], True);  mul_246 = None
        mul_247: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_15, sum_93);  sum_93 = None
        sub_57: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_245, sum_92);  mul_245 = sum_92 = None
        sub_58: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_57, mul_247);  sub_57 = mul_247 = None
        mul_248: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_58);  div_19 = sub_58 = None
        mul_249: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_88, mul_15);  mul_15 = None
        sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_249, [0, 1]);  mul_249 = None
        sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_88, [0, 1]);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        convert_element_type_11: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_250: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_251: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_248, mul_250);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_285: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_251, [32768, 768]);  mul_251 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_241: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_64: "f32[32768, 3072]" = torch.ops.aten.mm.default(view_285, permute_241);  permute_241 = None
        permute_242: "f32[768, 32768]" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_242, view_20);  permute_242 = view_20 = None
        sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        view_286: "f32[768]" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
        view_287: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(mm_64, [256, 128, 3072]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_19: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(addmm_4, [256, 128, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_11: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
        erf: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
        add_9: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_253: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_254: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
        mul_255: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_254, -0.5);  mul_254 = None
        exp_14: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_255);  mul_255 = None
        mul_256: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_257: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_256);  view_19 = mul_256 = None
        add_90: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_253, mul_257);  mul_253 = mul_257 = None
        mul_258: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(view_287, add_90);  view_287 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_288: "f32[32768, 3072]" = torch.ops.aten.reshape.default(mul_258, [32768, 3072]);  mul_258 = None
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_245: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_66: "f32[32768, 768]" = torch.ops.aten.mm.default(view_288, permute_245);  permute_245 = None
        permute_246: "f32[3072, 32768]" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_246, view_18);  permute_246 = view_18 = None
        sum_97: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
        view_289: "f32[3072]" = torch.ops.aten.reshape.default(sum_97, [3072]);  sum_97 = None
        view_290: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_66, [256, 128, 768]);  mm_66 = None
        add_91: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_248, view_290);  mul_248 = view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_260: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_91, primals_15);  primals_15 = None
        mul_261: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_260, 768)
        sum_98: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_260, [2], True)
        mul_262: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_260, mul_8);  mul_260 = None
        sum_99: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_262, [2], True);  mul_262 = None
        mul_263: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_99);  sum_99 = None
        sub_60: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_261, sum_98);  mul_261 = sum_98 = None
        sub_61: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_60, mul_263);  sub_60 = mul_263 = None
        mul_264: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_61);  div_20 = sub_61 = None
        mul_265: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_91, mul_8);  mul_8 = None
        sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_265, [0, 1]);  mul_265 = None
        sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_91, [0, 1]);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_291: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_264, [32768, 768])
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_68: "f32[32768, 768]" = torch.ops.aten.mm.default(view_291, permute_249);  permute_249 = None
        permute_250: "f32[768, 32768]" = torch.ops.aten.permute.default(view_291, [1, 0])
        mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_250, view_16);  permute_250 = view_16 = None
        sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
        view_292: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        view_293: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_68, [256, 128, 768]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_294: "f32[256, 128, 12, 64]" = torch.ops.aten.reshape.default(view_293, [256, 128, 12, 64]);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_253: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_55: "f32[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        view_295: "f32[3072, 128, 64]" = torch.ops.aten.reshape.default(clone_55, [3072, 128, 64]);  clone_55 = None
        bmm_32: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(permute_254, view_295);  permute_254 = None
        bmm_33: "f32[3072, 128, 128]" = torch.ops.aten.bmm.default(view_295, permute_255);  view_295 = permute_255 = None
        view_296: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_32, [256, 12, 128, 64]);  bmm_32 = None
        view_297: "f32[256, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_33, [256, 12, 128, 128]);  bmm_33 = None
        convert_element_type_12: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_266: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_267: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_297, mul_266);  view_297 = mul_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[256, 1, 128, 128]" = torch.ops.aten.expand.default(ge, [256, -1, 128, 128]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_1, full_default);  expand = full_default = None
        view_11: "f32[256, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm, [256, 12, 128, 128]);  bmm = None
        add_5: "f32[256, 12, 128, 128]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = where = None
        sub_1: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(add_5, amax);  add_5 = amax = None
        exp: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        full_default_2: "f32[256, 12, 128, 128]" = torch.ops.aten.full.default([256, 12, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[256, 12, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = full_default_2 = div = None
        mul_268: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_267, where_1);  mul_267 = None
        sum_103: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [-1], True)
        neg_6: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(where_1);  where_1 = None
        fma_5: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_6, sum_103, mul_268);  neg_6 = sum_103 = mul_268 = None
        view_298: "f32[3072, 128, 128]" = torch.ops.aten.reshape.default(fma_5, [3072, 128, 128]);  fma_5 = None
        bmm_34: "f32[3072, 64, 128]" = torch.ops.aten.bmm.default(permute_256, view_298);  permute_256 = None
        bmm_35: "f32[3072, 128, 64]" = torch.ops.aten.bmm.default(view_298, permute_257);  view_298 = permute_257 = None
        view_299: "f32[256, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_34, [256, 12, 64, 128]);  bmm_34 = None
        view_300: "f32[256, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_35, [256, 12, 128, 64]);  bmm_35 = None
        mul_269: "f32[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_299, 0.3535533905932738);  view_299 = None
        permute_258: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_269, [0, 1, 3, 2]);  mul_269 = None
        mul_270: "f32[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_300, 0.3535533905932738);  view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_259: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        clone_57: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_301: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_57, [256, 128, 768]);  clone_57 = None
        view_302: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_301, [32768, 768]);  view_301 = None
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_260: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_70: "f32[32768, 768]" = torch.ops.aten.mm.default(view_302, permute_260);  permute_260 = None
        permute_261: "f32[768, 32768]" = torch.ops.aten.permute.default(view_302, [1, 0])
        mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_261, view);  permute_261 = None
        sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
        view_303: "f32[768]" = torch.ops.aten.reshape.default(sum_104, [768]);  sum_104 = None
        view_304: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_70, [256, 128, 768]);  mm_70 = None
        add_92: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_264, view_304);  mul_264 = view_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_264: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_258, [0, 2, 1, 3]);  permute_258 = None
        view_305: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(permute_264, [256, 128, 768]);  permute_264 = None
        clone_58: "f32[256, 128, 768]" = torch.ops.aten.clone.default(view_305, memory_format = torch.contiguous_format);  view_305 = None
        view_306: "f32[32768, 768]" = torch.ops.aten.reshape.default(clone_58, [32768, 768]);  clone_58 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_72: "f32[32768, 768]" = torch.ops.aten.mm.default(view_306, permute_265);  permute_265 = None
        permute_266: "f32[768, 32768]" = torch.ops.aten.permute.default(view_306, [1, 0])
        mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_266, view);  permute_266 = None
        sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_306, [0], True);  view_306 = None
        view_307: "f32[768]" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None
        view_308: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_72, [256, 128, 768]);  mm_72 = None
        add_93: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_92, view_308);  add_92 = view_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_269: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_270, [0, 2, 1, 3]);  mul_270 = None
        clone_59: "f32[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_309: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(clone_59, [256, 128, 768]);  clone_59 = None
        view_310: "f32[32768, 768]" = torch.ops.aten.reshape.default(view_309, [32768, 768]);  view_309 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_74: "f32[32768, 768]" = torch.ops.aten.mm.default(view_310, permute_270);  permute_270 = None
        permute_271: "f32[768, 32768]" = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_271, view);  permute_271 = view = None
        sum_106: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        view_311: "f32[768]" = torch.ops.aten.reshape.default(sum_106, [768]);  sum_106 = None
        view_312: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_74, [256, 128, 768]);  mm_74 = None
        add_94: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_93, view_312);  add_93 = view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:121 in forward, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        convert_element_type_13: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_271: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_272: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_94, mul_271);  add_94 = mul_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:120 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        mul_274: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_272, primals_5);  primals_5 = None
        mul_275: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_274, 768)
        sum_107: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:119 in forward, code: embeddings = inputs_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:120 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        sub: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_276: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_274, mul);  mul_274 = None
        sum_108: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True);  mul_276 = None
        mul_277: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul, sum_108);  sum_108 = None
        sub_63: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_275, sum_107);  mul_275 = sum_107 = None
        sub_64: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_277);  sub_63 = mul_277 = None
        div_21: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_278: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_64);  div_21 = sub_64 = None
        mul_279: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_272, mul);  mul = None
        sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 1]);  mul_279 = None
        sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:119 in forward, code: embeddings = inputs_embeds + position_embeddings  # (bs, max_seq_length, dim)
        sum_111: "f32[1, 128, 768]" = torch.ops.aten.sum.dim_IntList(mul_278, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:112 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 128);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:117 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        eq_6: "b8[1, 128]" = torch.ops.aten.eq.Scalar(slice_1, -1)
        unsqueeze_5: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_6, -1);  eq_6 = None
        where_16: "f32[1, 128, 768]" = torch.ops.aten.where.self(unsqueeze_5, full_default_1, sum_111);  unsqueeze_5 = sum_111 = None
        full_default_24: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default_24, [slice_1], where_16, True);  full_default_24 = slice_1 = where_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        eq_7: "b8[256, 128]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_6: "b8[256, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_7, -1);  eq_7 = None
        where_17: "f32[256, 128, 768]" = torch.ops.aten.where.self(unsqueeze_6, full_default_1, mul_278);  unsqueeze_6 = full_default_1 = mul_278 = None
        full_default_26: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[30522, 768]" = torch.ops.aten.index_put.default(full_default_26, [primals_1], where_17, True);  full_default_26 = primals_1 = where_17 = None
        add_95: "f32[30522, 768]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_1);  slice_tensor = index_put_1 = None
        return (None, add_95, None, index_put, sum_109, sum_110, mm_75, view_311, mm_73, view_307, mm_71, view_303, mm_69, view_292, sum_100, sum_101, mm_67, view_289, mm_65, view_286, sum_94, sum_95, mm_63, view_283, mm_61, view_279, mm_59, view_275, mm_57, view_264, sum_85, sum_86, mm_55, view_261, mm_53, view_258, sum_79, sum_80, mm_51, view_255, mm_49, view_251, mm_47, view_247, mm_45, view_236, sum_70, sum_71, mm_43, view_233, mm_41, view_230, sum_64, sum_65, mm_39, view_227, mm_37, view_223, mm_35, view_219, mm_33, view_208, sum_55, sum_56, mm_31, view_205, mm_29, view_202, sum_49, sum_50, mm_27, view_199, mm_25, view_195, mm_23, view_191, mm_21, view_180, sum_40, sum_41, mm_19, view_177, mm_17, view_174, sum_34, sum_35, mm_15, view_171, mm_13, view_167, mm_11, view_163, mm_9, view_152, sum_25, sum_26, mm_7, view_149, mm_5, view_146, sum_19, sum_20, mm_3, view_143, sum_14, sum_15, view_140, None)
