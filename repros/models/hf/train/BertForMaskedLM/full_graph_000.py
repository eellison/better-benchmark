class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512]", primals_2: "i64[1, 512]", primals_3: "i64[1, 512]", primals_4: "f32[30522, 768]", primals_5: "f32[2, 768]", primals_6: "f32[512, 768]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_11: "f32[768, 768]", primals_12: "f32[768]", primals_13: "f32[768, 768]", primals_14: "f32[768]", primals_15: "f32[768, 768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[3072, 768]", primals_20: "f32[3072]", primals_21: "f32[768, 3072]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768, 768]", primals_26: "f32[768]", primals_27: "f32[768, 768]", primals_28: "f32[768]", primals_29: "f32[768, 768]", primals_30: "f32[768]", primals_31: "f32[768, 768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[3072, 768]", primals_36: "f32[3072]", primals_37: "f32[768, 3072]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768, 768]", primals_42: "f32[768]", primals_43: "f32[768, 768]", primals_44: "f32[768]", primals_45: "f32[768, 768]", primals_46: "f32[768]", primals_47: "f32[768, 768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[768]", primals_51: "f32[3072, 768]", primals_52: "f32[3072]", primals_53: "f32[768, 3072]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768, 768]", primals_58: "f32[768]", primals_59: "f32[768, 768]", primals_60: "f32[768]", primals_61: "f32[768, 768]", primals_62: "f32[768]", primals_63: "f32[768, 768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[3072, 768]", primals_68: "f32[3072]", primals_69: "f32[768, 3072]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768, 768]", primals_74: "f32[768]", primals_75: "f32[768, 768]", primals_76: "f32[768]", primals_77: "f32[768, 768]", primals_78: "f32[768]", primals_79: "f32[768, 768]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[768]", primals_83: "f32[3072, 768]", primals_84: "f32[3072]", primals_85: "f32[768, 3072]", primals_86: "f32[768]", primals_87: "f32[768]", primals_88: "f32[768]", primals_89: "f32[768, 768]", primals_90: "f32[768]", primals_91: "f32[768, 768]", primals_92: "f32[768]", primals_93: "f32[768, 768]", primals_94: "f32[768]", primals_95: "f32[768, 768]", primals_96: "f32[768]", primals_97: "f32[768]", primals_98: "f32[768]", primals_99: "f32[3072, 768]", primals_100: "f32[3072]", primals_101: "f32[768, 3072]", primals_102: "f32[768]", primals_103: "f32[768]", primals_104: "f32[768]", primals_105: "f32[768, 768]", primals_106: "f32[768]", primals_107: "f32[768, 768]", primals_108: "f32[768]", primals_109: "f32[768, 768]", primals_110: "f32[768]", primals_111: "f32[768, 768]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[3072, 768]", primals_116: "f32[3072]", primals_117: "f32[768, 3072]", primals_118: "f32[768]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[768, 768]", primals_122: "f32[768]", primals_123: "f32[768, 768]", primals_124: "f32[768]", primals_125: "f32[768, 768]", primals_126: "f32[768]", primals_127: "f32[768, 768]", primals_128: "f32[768]", primals_129: "f32[768]", primals_130: "f32[768]", primals_131: "f32[3072, 768]", primals_132: "f32[3072]", primals_133: "f32[768, 3072]", primals_134: "f32[768]", primals_135: "f32[768]", primals_136: "f32[768]", primals_137: "f32[768, 768]", primals_138: "f32[768]", primals_139: "f32[768, 768]", primals_140: "f32[768]", primals_141: "f32[768, 768]", primals_142: "f32[768]", primals_143: "f32[768, 768]", primals_144: "f32[768]", primals_145: "f32[768]", primals_146: "f32[768]", primals_147: "f32[3072, 768]", primals_148: "f32[3072]", primals_149: "f32[768, 3072]", primals_150: "f32[768]", primals_151: "f32[768]", primals_152: "f32[768]", primals_153: "f32[768, 768]", primals_154: "f32[768]", primals_155: "f32[768, 768]", primals_156: "f32[768]", primals_157: "f32[768, 768]", primals_158: "f32[768]", primals_159: "f32[768, 768]", primals_160: "f32[768]", primals_161: "f32[768]", primals_162: "f32[768]", primals_163: "f32[3072, 768]", primals_164: "f32[3072]", primals_165: "f32[768, 3072]", primals_166: "f32[768]", primals_167: "f32[768]", primals_168: "f32[768]", primals_169: "f32[768, 768]", primals_170: "f32[768]", primals_171: "f32[768, 768]", primals_172: "f32[768]", primals_173: "f32[768, 768]", primals_174: "f32[768]", primals_175: "f32[768, 768]", primals_176: "f32[768]", primals_177: "f32[768]", primals_178: "f32[768]", primals_179: "f32[3072, 768]", primals_180: "f32[3072]", primals_181: "f32[768, 3072]", primals_182: "f32[768]", primals_183: "f32[768]", primals_184: "f32[768]", primals_185: "f32[768, 768]", primals_186: "f32[768]", primals_187: "f32[768, 768]", primals_188: "f32[768]", primals_189: "f32[768, 768]", primals_190: "f32[768]", primals_191: "f32[768, 768]", primals_192: "f32[768]", primals_193: "f32[768]", primals_194: "f32[768]", primals_195: "f32[3072, 768]", primals_196: "f32[3072]", primals_197: "f32[768, 3072]", primals_198: "f32[768]", primals_199: "f32[768]", primals_200: "f32[768]", primals_201: "f32[768, 768]", primals_202: "f32[768]", primals_203: "f32[768]", primals_204: "f32[768]", primals_205: "f32[30522]", primals_206: "i64[32, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:96 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand: "i64[1, 512]" = torch.ops.aten.expand.default(primals_3, [1, -1]);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:97 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather: "i64[1, 512]" = torch.ops.aten.gather.default(expand, 1, primals_2);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:98 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[32, 512]" = torch.ops.aten.expand.default(gather, [32, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_4, primals_1, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:104 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_5, expand_1);  primals_5 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:105 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:107 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_6, primals_2);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:108 in forward, code: embeddings = embeddings + position_embeddings
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, primals_7)
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_8);  mul_1 = primals_8 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[37]" = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:111 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_36: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_3);  add_3 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[512]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 512, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_2: "b8[32, 1, 512, 512]" = torch.ops.aten.expand.default(ge, [32, -1, 512, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_3, [16384, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_10, view, permute);  primals_10 = permute = None
        view_1: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm, [32, 512, 768]);  addmm = None
        view_2: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [32, 512, -1, 64]);  view_1 = None
        permute_1: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_1: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_12, view, permute_2);  primals_12 = permute_2 = None
        view_4: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [32, 512, 768]);  addmm_1 = None
        view_5: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [32, 512, -1, 64]);  view_4 = None
        permute_3: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_2: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_14, view, permute_4);  primals_14 = permute_4 = None
        view_7: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 768]);  addmm_2 = None
        view_8: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_7, [32, 512, -1, 64]);  view_7 = None
        permute_5: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_1, full_default);  expand_2 = full_default = None
        mul_4: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        permute_6: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_5: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        expand_3: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_4, [32, 12, 512, 64]);  mul_4 = None
        clone: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_9: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone, [384, 512, 64]);  clone = None
        expand_4: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_5, [32, 12, 64, 512]);  mul_5 = None
        clone_1: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_10: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_1, [384, 64, 512]);  clone_1 = None
        bmm: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10)
        view_11: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [32, 12, 512, 512])
        add_6: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = None
        amax: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_6, [-1], True)
        sub_1: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_6, amax)
        exp: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        eq: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_6, -inf);  add_6 = None
        logical_not: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_2: "f32[32, 12, 512, 512]" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  div = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_35: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_35, 0.1);  inductor_random_default_35 = None
        mul_6: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_1, where_1);  where_1 = None
        mul_7: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None
        expand_5: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_7, [32, 12, 512, 512]);  mul_7 = None
        view_12: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_5, [384, 512, 512]);  expand_5 = None
        expand_6: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_5, [32, 12, 512, 64]);  permute_5 = None
        clone_2: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_13: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_2, [384, 512, 64]);  clone_2 = None
        bmm_1: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13)
        view_14: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [32, 12, 512, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_3, [32, 512, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_15, [16384, 768]);  view_15 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0])
        addmm_3: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_16, view_16, permute_8);  primals_16 = permute_8 = None
        view_17: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [32, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_34: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_34, 0.1);  inductor_random_default_34 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_2, view_17);  view_17 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, mul_3);  mul_9 = mul_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_8: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_3);  add_7 = getitem_3 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_17)
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_18);  mul_11 = primals_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_9, [16384, 768])
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_19, [1, 0])
        addmm_4: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_20, view_18, permute_9);  primals_20 = permute_9 = None
        view_19: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_12: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_13: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_10: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_14: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_10);  mul_12 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_14, [16384, 3072]);  mul_14 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0])
        addmm_5: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_22, view_20, permute_10);  primals_22 = permute_10 = None
        view_21: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_33: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_33, 0.1);  inductor_random_default_33 = None
        mul_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, 1.1111111111111112);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, add_9);  mul_16 = add_9 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_12: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_5);  add_11 = getitem_5 = None
        mul_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_23)
        add_13: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_24);  mul_18 = primals_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_22: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_13, [16384, 768])
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0])
        addmm_6: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_26, view_22, permute_11);  primals_26 = permute_11 = None
        view_23: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [32, 512, 768]);  addmm_6 = None
        view_24: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_23, [32, 512, -1, 64]);  view_23 = None
        permute_12: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0])
        addmm_7: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_28, view_22, permute_13);  primals_28 = permute_13 = None
        view_26: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [32, 512, 768]);  addmm_7 = None
        view_27: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_26, [32, 512, -1, 64]);  view_26 = None
        permute_14: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0])
        addmm_8: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_30, view_22, permute_15);  primals_30 = permute_15 = None
        view_29: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [32, 512, 768]);  addmm_8 = None
        view_30: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_29, [32, 512, -1, 64]);  view_29 = None
        permute_16: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_19: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
        permute_17: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        mul_20: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_17, 0.3535533905932738);  permute_17 = None
        expand_7: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_19, [32, 12, 512, 64]);  mul_19 = None
        clone_4: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_31: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_4, [384, 512, 64]);  clone_4 = None
        expand_8: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_20, [32, 12, 64, 512]);  mul_20 = None
        clone_5: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_32: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_5, [384, 64, 512]);  clone_5 = None
        bmm_2: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_31, view_32)
        view_33: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [32, 12, 512, 512]);  bmm_2 = None
        add_14: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_33, where);  view_33 = None
        amax_1: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_14, [-1], True)
        sub_4: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_14, amax_1);  amax_1 = None
        exp_1: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        eq_1: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_14, -inf);  add_14 = None
        logical_not_2: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        where_3: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full_default_2, div_1);  logical_not_3 = div_1 = None
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_32: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_32, 0.1);  inductor_random_default_32 = None
        mul_21: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_4, where_3)
        mul_22: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_21, 1.1111111111111112);  mul_21 = None
        expand_9: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_22, [32, 12, 512, 512]);  mul_22 = None
        view_34: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_9, [384, 512, 512]);  expand_9 = None
        expand_10: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_16, [32, 12, 512, 64]);  permute_16 = None
        clone_6: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_35: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_6, [384, 512, 64]);  clone_6 = None
        bmm_3: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_34, view_35)
        view_36: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [32, 12, 512, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_7: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_37: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_7, [32, 512, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_37, [16384, 768]);  view_37 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_31, [1, 0])
        addmm_9: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_32, view_38, permute_19);  primals_32 = permute_19 = None
        view_39: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_31: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_31, 0.1);  inductor_random_default_31 = None
        mul_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_5, view_39);  view_39 = None
        mul_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_23, 1.1111111111111112);  mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_24, add_13);  mul_24 = add_13 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[32, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_7);  add_15 = getitem_7 = None
        mul_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_26: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_33)
        add_17: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_34);  mul_26 = primals_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_17, [16384, 768])
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_35, [1, 0])
        addmm_10: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_36, view_40, permute_20);  primals_36 = permute_20 = None
        view_41: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_27: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_28: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_18: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_29: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_18);  mul_27 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_29, [16384, 3072]);  mul_29 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0])
        addmm_11: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_38, view_42, permute_21);  primals_38 = permute_21 = None
        view_43: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [32, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_30: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_30, 0.1);  inductor_random_default_30 = None
        mul_30: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_6, view_43);  view_43 = None
        mul_31: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_19: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_31, add_17);  mul_31 = add_17 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[32, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_20: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_6: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_9);  add_19 = getitem_9 = None
        mul_32: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_33: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_39)
        add_21: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, primals_40);  mul_33 = primals_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_44: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_21, [16384, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_41, [1, 0])
        addmm_12: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_42, view_44, permute_22);  primals_42 = permute_22 = None
        view_45: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [32, 512, 768]);  addmm_12 = None
        view_46: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_45, [32, 512, -1, 64]);  view_45 = None
        permute_23: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0])
        addmm_13: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_44, view_44, permute_24);  primals_44 = permute_24 = None
        view_48: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [32, 512, 768]);  addmm_13 = None
        view_49: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_48, [32, 512, -1, 64]);  view_48 = None
        permute_25: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0])
        addmm_14: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_46, view_44, permute_26);  primals_46 = permute_26 = None
        view_51: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [32, 512, 768]);  addmm_14 = None
        view_52: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_51, [32, 512, -1, 64]);  view_51 = None
        permute_27: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_34: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_23, 0.3535533905932738);  permute_23 = None
        permute_28: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        mul_35: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_28, 0.3535533905932738);  permute_28 = None
        expand_11: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_34, [32, 12, 512, 64]);  mul_34 = None
        clone_8: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_53: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_8, [384, 512, 64]);  clone_8 = None
        expand_12: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_35, [32, 12, 64, 512]);  mul_35 = None
        clone_9: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_54: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_9, [384, 64, 512]);  clone_9 = None
        bmm_4: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_53, view_54)
        view_55: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [32, 12, 512, 512]);  bmm_4 = None
        add_22: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_55, where);  view_55 = None
        amax_2: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_22, [-1], True)
        sub_7: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_22, amax_2);  amax_2 = None
        exp_2: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        eq_2: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_22, -inf);  add_22 = None
        logical_not_4: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        where_5: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full_default_2, div_2);  logical_not_5 = div_2 = None
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_29: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_29, 0.1);  inductor_random_default_29 = None
        mul_36: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_7, where_5)
        mul_37: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None
        expand_13: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_37, [32, 12, 512, 512]);  mul_37 = None
        view_56: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_13, [384, 512, 512]);  expand_13 = None
        expand_14: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_27, [32, 12, 512, 64]);  permute_27 = None
        clone_10: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_57: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_10, [384, 512, 64]);  clone_10 = None
        bmm_5: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_56, view_57)
        view_58: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [32, 12, 512, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_11: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_59: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_11, [32, 512, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_59, [16384, 768]);  view_59 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_47, [1, 0])
        addmm_15: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_48, view_60, permute_30);  primals_48 = permute_30 = None
        view_61: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [32, 512, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_28: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_28, 0.1);  inductor_random_default_28 = None
        mul_38: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_8, view_61);  view_61 = None
        mul_39: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, 1.1111111111111112);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_39, add_21);  mul_39 = add_21 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[32, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_24: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_8: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_11);  add_23 = getitem_11 = None
        mul_40: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_41: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_49)
        add_25: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_50);  mul_41 = primals_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_25, [16384, 768])
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_51, [1, 0])
        addmm_16: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_52, view_62, permute_31);  primals_52 = permute_31 = None
        view_63: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_42: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_43: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_26: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_44: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_42, add_26);  mul_42 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_44, [16384, 3072]);  mul_44 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_53, [1, 0])
        addmm_17: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_54, view_64, permute_32);  primals_54 = permute_32 = None
        view_65: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [32, 512, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_27: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_27, 0.1);  inductor_random_default_27 = None
        mul_45: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_9, view_65);  view_65 = None
        mul_46: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, 1.1111111111111112);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, add_25);  mul_46 = add_25 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[32, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_28: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_9: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_13);  add_27 = getitem_13 = None
        mul_47: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_48: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_47, primals_55)
        add_29: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_48, primals_56);  mul_48 = primals_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_66: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_29, [16384, 768])
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0])
        addmm_18: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_58, view_66, permute_33);  primals_58 = permute_33 = None
        view_67: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [32, 512, 768]);  addmm_18 = None
        view_68: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_67, [32, 512, -1, 64]);  view_67 = None
        permute_34: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_59, [1, 0])
        addmm_19: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_60, view_66, permute_35);  primals_60 = permute_35 = None
        view_70: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 768]);  addmm_19 = None
        view_71: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_70, [32, 512, -1, 64]);  view_70 = None
        permute_36: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0])
        addmm_20: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_62, view_66, permute_37);  primals_62 = permute_37 = None
        view_73: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [32, 512, 768]);  addmm_20 = None
        view_74: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_73, [32, 512, -1, 64]);  view_73 = None
        permute_38: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_49: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_34, 0.3535533905932738);  permute_34 = None
        permute_39: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        mul_50: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_39, 0.3535533905932738);  permute_39 = None
        expand_15: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_49, [32, 12, 512, 64]);  mul_49 = None
        clone_12: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_75: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_12, [384, 512, 64]);  clone_12 = None
        expand_16: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_50, [32, 12, 64, 512]);  mul_50 = None
        clone_13: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_76: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_13, [384, 64, 512]);  clone_13 = None
        bmm_6: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_75, view_76)
        view_77: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [32, 12, 512, 512]);  bmm_6 = None
        add_30: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_77, where);  view_77 = None
        amax_3: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_30, [-1], True)
        sub_10: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_30, amax_3);  amax_3 = None
        exp_3: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        eq_3: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_30, -inf);  add_30 = None
        logical_not_6: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        where_7: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full_default_2, div_3);  logical_not_7 = div_3 = None
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_26: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_51: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_10, where_7)
        mul_52: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_51, 1.1111111111111112);  mul_51 = None
        expand_17: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_52, [32, 12, 512, 512]);  mul_52 = None
        view_78: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_17, [384, 512, 512]);  expand_17 = None
        expand_18: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_38, [32, 12, 512, 64]);  permute_38 = None
        clone_14: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_79: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_14, [384, 512, 64]);  clone_14 = None
        bmm_7: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_78, view_79)
        view_80: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [32, 12, 512, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_15: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_81: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_15, [32, 512, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_82: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_81, [16384, 768]);  view_81 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_63, [1, 0])
        addmm_21: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_64, view_82, permute_41);  primals_64 = permute_41 = None
        view_83: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [32, 512, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_25: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_53: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_11, view_83);  view_83 = None
        mul_54: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_54, add_29);  mul_54 = add_29 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[32, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_32: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_11: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_15);  add_31 = getitem_15 = None
        mul_55: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_56: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_55, primals_65)
        add_33: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_56, primals_66);  mul_56 = primals_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_33, [16384, 768])
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_67, [1, 0])
        addmm_22: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_68, view_84, permute_42);  primals_68 = permute_42 = None
        view_85: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_57: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_58: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
        add_34: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_59: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_57, add_34);  mul_57 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_59, [16384, 3072]);  mul_59 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0])
        addmm_23: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_70, view_86, permute_43);  primals_70 = permute_43 = None
        view_87: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_24: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_60: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_12, view_87);  view_87 = None
        mul_61: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_35: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, add_33);  mul_61 = add_33 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[32, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_36: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_12: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_17);  add_35 = getitem_17 = None
        mul_62: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_63: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_62, primals_71)
        add_37: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_63, primals_72);  mul_63 = primals_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_88: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_37, [16384, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0])
        addmm_24: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_74, view_88, permute_44);  primals_74 = permute_44 = None
        view_89: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [32, 512, 768]);  addmm_24 = None
        view_90: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_89, [32, 512, -1, 64]);  view_89 = None
        permute_45: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0])
        addmm_25: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_76, view_88, permute_46);  primals_76 = permute_46 = None
        view_92: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [32, 512, 768]);  addmm_25 = None
        view_93: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_92, [32, 512, -1, 64]);  view_92 = None
        permute_47: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_77, [1, 0])
        addmm_26: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_78, view_88, permute_48);  primals_78 = permute_48 = None
        view_95: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [32, 512, 768]);  addmm_26 = None
        view_96: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_95, [32, 512, -1, 64]);  view_95 = None
        permute_49: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_64: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_45, 0.3535533905932738);  permute_45 = None
        permute_50: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        mul_65: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_50, 0.3535533905932738);  permute_50 = None
        expand_19: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_64, [32, 12, 512, 64]);  mul_64 = None
        clone_16: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_97: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_16, [384, 512, 64]);  clone_16 = None
        expand_20: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_65, [32, 12, 64, 512]);  mul_65 = None
        clone_17: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_98: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_17, [384, 64, 512]);  clone_17 = None
        bmm_8: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_97, view_98)
        view_99: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [32, 12, 512, 512]);  bmm_8 = None
        add_38: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_99, where);  view_99 = None
        amax_4: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_38, [-1], True)
        sub_13: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_38, amax_4);  amax_4 = None
        exp_4: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        eq_4: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_38, -inf);  add_38 = None
        logical_not_8: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        where_9: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full_default_2, div_4);  logical_not_9 = div_4 = None
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_23: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_13: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_66: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_13, where_9)
        mul_67: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None
        expand_21: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_67, [32, 12, 512, 512]);  mul_67 = None
        view_100: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_21, [384, 512, 512]);  expand_21 = None
        expand_22: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_49, [32, 12, 512, 64]);  permute_49 = None
        clone_18: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_101: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_18, [384, 512, 64]);  clone_18 = None
        bmm_9: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_100, view_101)
        view_102: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [32, 12, 512, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_19: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_103: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_19, [32, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_103, [16384, 768]);  view_103 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0])
        addmm_27: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_80, view_104, permute_52);  primals_80 = permute_52 = None
        view_105: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [32, 512, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_22: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_14: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_68: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_14, view_105);  view_105 = None
        mul_69: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_68, 1.1111111111111112);  mul_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_69, add_37);  mul_69 = add_37 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[32, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_40: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_14: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_19);  add_39 = getitem_19 = None
        mul_70: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_71: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_81)
        add_41: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_71, primals_82);  mul_71 = primals_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_41, [16384, 768])
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_83, [1, 0])
        addmm_28: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_84, view_106, permute_53);  primals_84 = permute_53 = None
        view_107: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_72: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_73: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_42: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_74: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_72, add_42);  mul_72 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_74, [16384, 3072]);  mul_74 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0])
        addmm_29: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_86, view_108, permute_54);  primals_86 = permute_54 = None
        view_109: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_29, [32, 512, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_21: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_15: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_75: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_15, view_109);  view_109 = None
        mul_76: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_75, 1.1111111111111112);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_76, add_41);  mul_76 = add_41 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[32, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_44: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_15: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_21);  add_43 = getitem_21 = None
        mul_77: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_78: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_87)
        add_45: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_78, primals_88);  mul_78 = primals_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_110: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_45, [16384, 768])
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0])
        addmm_30: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_90, view_110, permute_55);  primals_90 = permute_55 = None
        view_111: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [32, 512, 768]);  addmm_30 = None
        view_112: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_111, [32, 512, -1, 64]);  view_111 = None
        permute_56: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0])
        addmm_31: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_92, view_110, permute_57);  primals_92 = permute_57 = None
        view_114: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_31, [32, 512, 768]);  addmm_31 = None
        view_115: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_114, [32, 512, -1, 64]);  view_114 = None
        permute_58: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0])
        addmm_32: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_94, view_110, permute_59);  primals_94 = permute_59 = None
        view_117: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [32, 512, 768]);  addmm_32 = None
        view_118: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_117, [32, 512, -1, 64]);  view_117 = None
        permute_60: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_79: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_56, 0.3535533905932738);  permute_56 = None
        permute_61: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        mul_80: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_61, 0.3535533905932738);  permute_61 = None
        expand_23: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_79, [32, 12, 512, 64]);  mul_79 = None
        clone_20: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_119: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_20, [384, 512, 64]);  clone_20 = None
        expand_24: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_80, [32, 12, 64, 512]);  mul_80 = None
        clone_21: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_120: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_21, [384, 64, 512]);  clone_21 = None
        bmm_10: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_119, view_120)
        view_121: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [32, 12, 512, 512]);  bmm_10 = None
        add_46: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_121, where);  view_121 = None
        amax_5: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_46, [-1], True)
        sub_16: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_46, amax_5);  amax_5 = None
        exp_5: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        eq_5: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_46, -inf);  add_46 = None
        logical_not_10: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        where_11: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full_default_2, div_5);  logical_not_11 = div_5 = None
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_20: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_16: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_81: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_16, where_11)
        mul_82: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_81, 1.1111111111111112);  mul_81 = None
        expand_25: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_82, [32, 12, 512, 512]);  mul_82 = None
        view_122: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_25, [384, 512, 512]);  expand_25 = None
        expand_26: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_60, [32, 12, 512, 64]);  permute_60 = None
        clone_22: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_123: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_22, [384, 512, 64]);  clone_22 = None
        bmm_11: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_122, view_123)
        view_124: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [32, 12, 512, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_23: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_125: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_23, [32, 512, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_125, [16384, 768]);  view_125 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_95, [1, 0])
        addmm_33: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_96, view_126, permute_63);  primals_96 = permute_63 = None
        view_127: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_33, [32, 512, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_19: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_17: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_83: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_17, view_127);  view_127 = None
        mul_84: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_83, 1.1111111111111112);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_47: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_84, add_45);  mul_84 = add_45 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[32, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_48: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_17: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_23);  add_47 = getitem_23 = None
        mul_85: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_86: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_97)
        add_49: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_98);  mul_86 = primals_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_49, [16384, 768])
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_99, [1, 0])
        addmm_34: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_100, view_128, permute_64);  primals_100 = permute_64 = None
        view_129: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_87: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_88: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_50: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_89: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_87, add_50);  mul_87 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_89, [16384, 3072]);  mul_89 = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_101, [1, 0])
        addmm_35: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_102, view_130, permute_65);  primals_102 = permute_65 = None
        view_131: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_35, [32, 512, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_18: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_18: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_90: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_18, view_131);  view_131 = None
        mul_91: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, 1.1111111111111112);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_51: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_91, add_49);  mul_91 = add_49 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[32, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_52: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_18: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_25);  add_51 = getitem_25 = None
        mul_92: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_93: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, primals_103)
        add_53: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_93, primals_104);  mul_93 = primals_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_132: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_53, [16384, 768])
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0])
        addmm_36: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_106, view_132, permute_66);  primals_106 = permute_66 = None
        view_133: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [32, 512, 768]);  addmm_36 = None
        view_134: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_133, [32, 512, -1, 64]);  view_133 = None
        permute_67: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_107, [1, 0])
        addmm_37: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_108, view_132, permute_68);  primals_108 = permute_68 = None
        view_136: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_37, [32, 512, 768]);  addmm_37 = None
        view_137: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_136, [32, 512, -1, 64]);  view_136 = None
        permute_69: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0])
        addmm_38: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_110, view_132, permute_70);  primals_110 = permute_70 = None
        view_139: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_38, [32, 512, 768]);  addmm_38 = None
        view_140: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_139, [32, 512, -1, 64]);  view_139 = None
        permute_71: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_94: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_67, 0.3535533905932738);  permute_67 = None
        permute_72: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_69, [0, 1, 3, 2]);  permute_69 = None
        mul_95: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_72, 0.3535533905932738);  permute_72 = None
        expand_27: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_94, [32, 12, 512, 64]);  mul_94 = None
        clone_24: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_141: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_24, [384, 512, 64]);  clone_24 = None
        expand_28: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_95, [32, 12, 64, 512]);  mul_95 = None
        clone_25: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_142: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_25, [384, 64, 512]);  clone_25 = None
        bmm_12: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_141, view_142)
        view_143: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [32, 12, 512, 512]);  bmm_12 = None
        add_54: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_143, where);  view_143 = None
        amax_6: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_54, [-1], True)
        sub_19: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_54, amax_6);  amax_6 = None
        exp_6: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        eq_6: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_54, -inf);  add_54 = None
        logical_not_12: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        where_13: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_13, full_default_2, div_6);  logical_not_13 = div_6 = None
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_17: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_19: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_96: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_19, where_13)
        mul_97: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None
        expand_29: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_97, [32, 12, 512, 512]);  mul_97 = None
        view_144: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_29, [384, 512, 512]);  expand_29 = None
        expand_30: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_71, [32, 12, 512, 64]);  permute_71 = None
        clone_26: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_145: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_26, [384, 512, 64]);  clone_26 = None
        bmm_13: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_144, view_145)
        view_146: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [32, 12, 512, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_27: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_147: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_27, [32, 512, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_147, [16384, 768]);  view_147 = None
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_111, [1, 0])
        addmm_39: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_112, view_148, permute_74);  primals_112 = permute_74 = None
        view_149: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_39, [32, 512, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_16: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_20: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_98: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_20, view_149);  view_149 = None
        mul_99: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_55: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_99, add_53);  mul_99 = add_53 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[32, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_56: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_20: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_27);  add_55 = getitem_27 = None
        mul_100: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
        mul_101: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_100, primals_113)
        add_57: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_101, primals_114);  mul_101 = primals_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_57, [16384, 768])
        permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_115, [1, 0])
        addmm_40: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_116, view_150, permute_75);  primals_116 = permute_75 = None
        view_151: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_102: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_103: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_6: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_58: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_104: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_102, add_58);  mul_102 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_104, [16384, 3072]);  mul_104 = None
        permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0])
        addmm_41: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_118, view_152, permute_76);  primals_118 = permute_76 = None
        view_153: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_41, [32, 512, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_15: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_21: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_105: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_21, view_153);  view_153 = None
        mul_106: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_106, add_57);  mul_106 = add_57 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[32, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_60: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_21: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_29);  add_59 = getitem_29 = None
        mul_107: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
        mul_108: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, primals_119)
        add_61: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_108, primals_120);  mul_108 = primals_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_154: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_61, [16384, 768])
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_121, [1, 0])
        addmm_42: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_122, view_154, permute_77);  primals_122 = permute_77 = None
        view_155: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_42, [32, 512, 768]);  addmm_42 = None
        view_156: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_155, [32, 512, -1, 64]);  view_155 = None
        permute_78: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(primals_123, [1, 0])
        addmm_43: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_124, view_154, permute_79);  primals_124 = permute_79 = None
        view_158: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_43, [32, 512, 768]);  addmm_43 = None
        view_159: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_158, [32, 512, -1, 64]);  view_158 = None
        permute_80: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(primals_125, [1, 0])
        addmm_44: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_126, view_154, permute_81);  primals_126 = permute_81 = None
        view_161: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_44, [32, 512, 768]);  addmm_44 = None
        view_162: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_161, [32, 512, -1, 64]);  view_161 = None
        permute_82: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_109: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_78, 0.3535533905932738);  permute_78 = None
        permute_83: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        mul_110: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_83, 0.3535533905932738);  permute_83 = None
        expand_31: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_109, [32, 12, 512, 64]);  mul_109 = None
        clone_28: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_163: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_28, [384, 512, 64]);  clone_28 = None
        expand_32: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_110, [32, 12, 64, 512]);  mul_110 = None
        clone_29: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_164: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_29, [384, 64, 512]);  clone_29 = None
        bmm_14: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_163, view_164)
        view_165: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_14, [32, 12, 512, 512]);  bmm_14 = None
        add_62: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_165, where);  view_165 = None
        amax_7: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_62, [-1], True)
        sub_22: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_62, amax_7);  amax_7 = None
        exp_7: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        eq_7: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_62, -inf);  add_62 = None
        logical_not_14: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        where_15: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_15, full_default_2, div_7);  logical_not_15 = div_7 = None
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_14: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_22: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_111: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_22, where_15)
        mul_112: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_111, 1.1111111111111112);  mul_111 = None
        expand_33: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_112, [32, 12, 512, 512]);  mul_112 = None
        view_166: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_33, [384, 512, 512]);  expand_33 = None
        expand_34: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_82, [32, 12, 512, 64]);  permute_82 = None
        clone_30: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_167: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_30, [384, 512, 64]);  clone_30 = None
        bmm_15: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_166, view_167)
        view_168: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [32, 12, 512, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_31: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_31, [32, 512, -1]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_169, [16384, 768]);  view_169 = None
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_127, [1, 0])
        addmm_45: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_128, view_170, permute_85);  primals_128 = permute_85 = None
        view_171: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_45, [32, 512, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_13: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_23: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_113: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_23, view_171);  view_171 = None
        mul_114: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, 1.1111111111111112);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_114, add_61);  mul_114 = add_61 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[32, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_64: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_23: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_31);  add_63 = getitem_31 = None
        mul_115: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
        mul_116: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, primals_129)
        add_65: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_116, primals_130);  mul_116 = primals_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_65, [16384, 768])
        permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_131, [1, 0])
        addmm_46: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_132, view_172, permute_86);  primals_132 = permute_86 = None
        view_173: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_117: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_118: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_7: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_66: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_119: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_117, add_66);  mul_117 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_119, [16384, 3072]);  mul_119 = None
        permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0])
        addmm_47: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_134, view_174, permute_87);  primals_134 = permute_87 = None
        view_175: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_47, [32, 512, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_12: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_24: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_120: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_24, view_175);  view_175 = None
        mul_121: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_121, add_65);  mul_121 = add_65 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[32, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_68: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_24: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_33);  add_67 = getitem_33 = None
        mul_122: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
        mul_123: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, primals_135)
        add_69: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_123, primals_136);  mul_123 = primals_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_176: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_69, [16384, 768])
        permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0])
        addmm_48: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_138, view_176, permute_88);  primals_138 = permute_88 = None
        view_177: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_48, [32, 512, 768]);  addmm_48 = None
        view_178: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_177, [32, 512, -1, 64]);  view_177 = None
        permute_89: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(primals_139, [1, 0])
        addmm_49: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_140, view_176, permute_90);  primals_140 = permute_90 = None
        view_180: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_49, [32, 512, 768]);  addmm_49 = None
        view_181: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_180, [32, 512, -1, 64]);  view_180 = None
        permute_91: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0])
        addmm_50: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_142, view_176, permute_92);  primals_142 = permute_92 = None
        view_183: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_50, [32, 512, 768]);  addmm_50 = None
        view_184: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_183, [32, 512, -1, 64]);  view_183 = None
        permute_93: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_124: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_89, 0.3535533905932738);  permute_89 = None
        permute_94: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        mul_125: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_94, 0.3535533905932738);  permute_94 = None
        expand_35: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_124, [32, 12, 512, 64]);  mul_124 = None
        clone_32: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_185: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_32, [384, 512, 64]);  clone_32 = None
        expand_36: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_125, [32, 12, 64, 512]);  mul_125 = None
        clone_33: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_186: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_33, [384, 64, 512]);  clone_33 = None
        bmm_16: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_185, view_186)
        view_187: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [32, 12, 512, 512]);  bmm_16 = None
        add_70: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_187, where);  view_187 = None
        amax_8: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_70, [-1], True)
        sub_25: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_70, amax_8);  amax_8 = None
        exp_8: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        eq_8: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_70, -inf);  add_70 = None
        logical_not_16: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        where_17: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_17, full_default_2, div_8);  logical_not_17 = div_8 = None
        inductor_lookup_seed_default_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_11: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_25: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_126: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_25, where_17)
        mul_127: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_126, 1.1111111111111112);  mul_126 = None
        expand_37: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_127, [32, 12, 512, 512]);  mul_127 = None
        view_188: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_37, [384, 512, 512]);  expand_37 = None
        expand_38: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_93, [32, 12, 512, 64]);  permute_93 = None
        clone_34: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_189: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_34, [384, 512, 64]);  clone_34 = None
        bmm_17: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_188, view_189)
        view_190: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [32, 12, 512, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_35: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_191: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_35, [32, 512, -1]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_191, [16384, 768]);  view_191 = None
        permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_143, [1, 0])
        addmm_51: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_144, view_192, permute_96);  primals_144 = permute_96 = None
        view_193: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_51, [32, 512, 768]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_10: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_26: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_128: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_26, view_193);  view_193 = None
        mul_129: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_128, 1.1111111111111112);  mul_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_129, add_69);  mul_129 = add_69 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[32, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_72: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_26: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_35);  add_71 = getitem_35 = None
        mul_130: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
        mul_131: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_130, primals_145)
        add_73: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_131, primals_146);  mul_131 = primals_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_73, [16384, 768])
        permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_147, [1, 0])
        addmm_52: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_148, view_194, permute_97);  primals_148 = permute_97 = None
        view_195: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_132: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_133: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_8: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_74: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_134: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_132, add_74);  mul_132 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_134, [16384, 3072]);  mul_134 = None
        permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_149, [1, 0])
        addmm_53: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_150, view_196, permute_98);  primals_150 = permute_98 = None
        view_197: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_53, [32, 512, 768]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_9: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        gt_27: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_135: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_27, view_197);  view_197 = None
        mul_136: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_75: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_136, add_73);  mul_136 = add_73 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[32, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_76: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_27: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_37);  add_75 = getitem_37 = None
        mul_137: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
        mul_138: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, primals_151)
        add_77: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_138, primals_152);  mul_138 = primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_198: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_77, [16384, 768])
        permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_153, [1, 0])
        addmm_54: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_154, view_198, permute_99);  primals_154 = permute_99 = None
        view_199: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_54, [32, 512, 768]);  addmm_54 = None
        view_200: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_199, [32, 512, -1, 64]);  view_199 = None
        permute_100: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_155, [1, 0])
        addmm_55: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_156, view_198, permute_101);  primals_156 = permute_101 = None
        view_202: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_55, [32, 512, 768]);  addmm_55 = None
        view_203: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_202, [32, 512, -1, 64]);  view_202 = None
        permute_102: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_157, [1, 0])
        addmm_56: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_158, view_198, permute_103);  primals_158 = permute_103 = None
        view_205: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_56, [32, 512, 768]);  addmm_56 = None
        view_206: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_205, [32, 512, -1, 64]);  view_205 = None
        permute_104: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_139: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_100, 0.3535533905932738);  permute_100 = None
        permute_105: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_102, [0, 1, 3, 2]);  permute_102 = None
        mul_140: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_105, 0.3535533905932738);  permute_105 = None
        expand_39: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_139, [32, 12, 512, 64]);  mul_139 = None
        clone_36: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_207: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_36, [384, 512, 64]);  clone_36 = None
        expand_40: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_140, [32, 12, 64, 512]);  mul_140 = None
        clone_37: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_208: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_37, [384, 64, 512]);  clone_37 = None
        bmm_18: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_207, view_208)
        view_209: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_18, [32, 12, 512, 512]);  bmm_18 = None
        add_78: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_209, where);  view_209 = None
        amax_9: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
        sub_28: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_78, amax_9);  amax_9 = None
        exp_9: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        eq_9: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_78, -inf);  add_78 = None
        logical_not_18: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        where_19: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_19, full_default_2, div_9);  logical_not_19 = div_9 = None
        inductor_lookup_seed_default_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_8: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        gt_28: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_141: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_28, where_19)
        mul_142: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_141, 1.1111111111111112);  mul_141 = None
        expand_41: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_142, [32, 12, 512, 512]);  mul_142 = None
        view_210: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_41, [384, 512, 512]);  expand_41 = None
        expand_42: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_104, [32, 12, 512, 64]);  permute_104 = None
        clone_38: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_211: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_38, [384, 512, 64]);  clone_38 = None
        bmm_19: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_210, view_211)
        view_212: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [32, 12, 512, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_39: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_213: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_39, [32, 512, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_213, [16384, 768]);  view_213 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0])
        addmm_57: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_160, view_214, permute_107);  primals_160 = permute_107 = None
        view_215: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_57, [32, 512, 768]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_7: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        gt_29: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_143: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_29, view_215);  view_215 = None
        mul_144: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_143, 1.1111111111111112);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_79: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_144, add_77);  mul_144 = add_77 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[32, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_80: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_29: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_39);  add_79 = getitem_39 = None
        mul_145: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
        mul_146: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_145, primals_161)
        add_81: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_146, primals_162);  mul_146 = primals_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_81, [16384, 768])
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_163, [1, 0])
        addmm_58: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_164, view_216, permute_108);  primals_164 = permute_108 = None
        view_217: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_147: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_148: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_9: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_82: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_149: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_147, add_82);  mul_147 = add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_149, [16384, 3072]);  mul_149 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_165, [1, 0])
        addmm_59: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_166, view_218, permute_109);  primals_166 = permute_109 = None
        view_219: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_59, [32, 512, 768]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_6: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_30: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_150: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_30, view_219);  view_219 = None
        mul_151: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_150, 1.1111111111111112);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_151, add_81);  mul_151 = add_81 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[32, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_84: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_30: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_41);  add_83 = getitem_41 = None
        mul_152: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
        mul_153: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_152, primals_167)
        add_85: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_153, primals_168);  mul_153 = primals_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_220: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_85, [16384, 768])
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_169, [1, 0])
        addmm_60: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_170, view_220, permute_110);  primals_170 = permute_110 = None
        view_221: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_60, [32, 512, 768]);  addmm_60 = None
        view_222: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_221, [32, 512, -1, 64]);  view_221 = None
        permute_111: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0])
        addmm_61: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_172, view_220, permute_112);  primals_172 = permute_112 = None
        view_224: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_61, [32, 512, 768]);  addmm_61 = None
        view_225: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_224, [32, 512, -1, 64]);  view_224 = None
        permute_113: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(primals_173, [1, 0])
        addmm_62: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_174, view_220, permute_114);  primals_174 = permute_114 = None
        view_227: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_62, [32, 512, 768]);  addmm_62 = None
        view_228: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_227, [32, 512, -1, 64]);  view_227 = None
        permute_115: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_154: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_111, 0.3535533905932738);  permute_111 = None
        permute_116: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_113, [0, 1, 3, 2]);  permute_113 = None
        mul_155: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_116, 0.3535533905932738);  permute_116 = None
        expand_43: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_154, [32, 12, 512, 64]);  mul_154 = None
        clone_40: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_229: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_40, [384, 512, 64]);  clone_40 = None
        expand_44: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_155, [32, 12, 64, 512]);  mul_155 = None
        clone_41: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_230: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_41, [384, 64, 512]);  clone_41 = None
        bmm_20: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_229, view_230)
        view_231: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [32, 12, 512, 512]);  bmm_20 = None
        add_86: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_231, where);  view_231 = None
        amax_10: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_86, [-1], True)
        sub_31: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_86, amax_10);  amax_10 = None
        exp_10: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        eq_10: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_86, -inf);  add_86 = None
        logical_not_20: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        where_21: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_21, full_default_2, div_10);  logical_not_21 = div_10 = None
        inductor_lookup_seed_default_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_5: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_31: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_156: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_31, where_21)
        mul_157: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_156, 1.1111111111111112);  mul_156 = None
        expand_45: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_157, [32, 12, 512, 512]);  mul_157 = None
        view_232: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_45, [384, 512, 512]);  expand_45 = None
        expand_46: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_115, [32, 12, 512, 64]);  permute_115 = None
        clone_42: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_233: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_42, [384, 512, 64]);  clone_42 = None
        bmm_21: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_232, view_233)
        view_234: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [32, 12, 512, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_43: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_43, [32, 512, -1]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_235, [16384, 768]);  view_235 = None
        permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_175, [1, 0])
        addmm_63: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_176, view_236, permute_118);  primals_176 = permute_118 = None
        view_237: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_63, [32, 512, 768]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_4: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        gt_32: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_158: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_32, view_237);  view_237 = None
        mul_159: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_158, 1.1111111111111112);  mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_159, add_85);  mul_159 = add_85 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[32, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_88: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_32: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_43);  add_87 = getitem_43 = None
        mul_160: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
        mul_161: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_160, primals_177)
        add_89: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_161, primals_178);  mul_161 = primals_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_89, [16384, 768])
        permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_179, [1, 0])
        addmm_64: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_180, view_238, permute_119);  primals_180 = permute_119 = None
        view_239: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_162: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_163: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_10: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_90: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_164: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_162, add_90);  mul_162 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_164, [16384, 3072]);  mul_164 = None
        permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_181, [1, 0])
        addmm_65: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_182, view_240, permute_120);  primals_182 = permute_120 = None
        view_241: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_65, [32, 512, 768]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_3: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        gt_33: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_165: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_33, view_241);  view_241 = None
        mul_166: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_165, 1.1111111111111112);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_91: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_166, add_89);  mul_166 = add_89 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[32, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_92: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_33: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_45);  add_91 = getitem_45 = None
        mul_167: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
        mul_168: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_167, primals_183)
        add_93: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_168, primals_184);  mul_168 = primals_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_242: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_93, [16384, 768])
        permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_185, [1, 0])
        addmm_66: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_186, view_242, permute_121);  primals_186 = permute_121 = None
        view_243: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_66, [32, 512, 768]);  addmm_66 = None
        view_244: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_243, [32, 512, -1, 64]);  view_243 = None
        permute_122: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(primals_187, [1, 0])
        addmm_67: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_188, view_242, permute_123);  primals_188 = permute_123 = None
        view_246: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, [32, 512, 768]);  addmm_67 = None
        view_247: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_246, [32, 512, -1, 64]);  view_246 = None
        permute_124: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(primals_189, [1, 0])
        addmm_68: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_190, view_242, permute_125);  primals_190 = permute_125 = None
        view_249: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_68, [32, 512, 768]);  addmm_68 = None
        view_250: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_249, [32, 512, -1, 64]);  view_249 = None
        permute_126: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_169: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_122, 0.3535533905932738);  permute_122 = None
        permute_127: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
        mul_170: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_127, 0.3535533905932738);  permute_127 = None
        expand_47: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_169, [32, 12, 512, 64]);  mul_169 = None
        clone_44: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_251: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_44, [384, 512, 64]);  clone_44 = None
        expand_48: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_170, [32, 12, 64, 512]);  mul_170 = None
        clone_45: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_252: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_45, [384, 64, 512]);  clone_45 = None
        bmm_22: "f32[384, 512, 512]" = torch.ops.aten.bmm.default(view_251, view_252)
        view_253: "f32[32, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [32, 12, 512, 512]);  bmm_22 = None
        add_94: "f32[32, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_253, where);  view_253 = where = None
        amax_11: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(add_94, [-1], True)
        sub_34: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_94, amax_11);  amax_11 = None
        exp_11: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        eq_11: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_94, -inf);  add_94 = None
        logical_not_22: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        where_23: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_23, full_default_2, div_11);  logical_not_23 = full_default_2 = div_11 = None
        inductor_lookup_seed_default_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_2: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default([32, 12, 512, 512], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_34: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_171: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_34, where_23)
        mul_172: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_171, 1.1111111111111112);  mul_171 = None
        expand_49: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_172, [32, 12, 512, 512]);  mul_172 = None
        view_254: "f32[384, 512, 512]" = torch.ops.aten.reshape.default(expand_49, [384, 512, 512]);  expand_49 = None
        expand_50: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_126, [32, 12, 512, 64]);  permute_126 = None
        clone_46: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_255: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_46, [384, 512, 64]);  clone_46 = None
        bmm_23: "f32[384, 512, 64]" = torch.ops.aten.bmm.default(view_254, view_255)
        view_256: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [32, 12, 512, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_47: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_257: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_47, [32, 512, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_258: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_257, [16384, 768]);  view_257 = None
        permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_191, [1, 0])
        addmm_69: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_192, view_258, permute_129);  primals_192 = permute_129 = None
        view_259: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_69, [32, 512, 768]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_1: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        gt_35: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_173: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_35, view_259);  view_259 = None
        mul_174: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_173, 1.1111111111111112);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_95: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_174, add_93);  mul_174 = add_93 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[32, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_96: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_35: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_47);  add_95 = getitem_47 = None
        mul_175: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
        mul_176: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_175, primals_193)
        add_97: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_176, primals_194);  mul_176 = primals_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_97, [16384, 768])
        permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_195, [1, 0])
        addmm_70: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_196, view_260, permute_130);  primals_196 = permute_130 = None
        view_261: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_177: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_178: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_11: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
        add_98: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_179: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_177, add_98);  mul_177 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_179, [16384, 3072]);  mul_179 = None
        permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_197, [1, 0])
        addmm_71: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_198, view_262, permute_131);  primals_198 = permute_131 = None
        view_263: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_71, [32, 512, 768]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        gt_36: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_180: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_36, view_263);  view_263 = None
        mul_181: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_180, 1.1111111111111112);  mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_99: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_181, add_97);  mul_181 = add_97 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[32, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_100: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_36: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_49);  add_99 = getitem_49 = None
        mul_182: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
        mul_183: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_182, primals_199)
        add_101: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_183, primals_200);  mul_183 = primals_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        view_264: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_101, [16384, 768]);  add_101 = None
        permute_132: "f32[768, 768]" = torch.ops.aten.permute.default(primals_201, [1, 0])
        addmm_72: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_202, view_264, permute_132);  primals_202 = permute_132 = None
        view_265: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_72, [32, 512, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_184: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.5)
        mul_185: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476);  view_265 = None
        erf_12: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_102: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_186: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_184, add_102);  mul_184 = add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_186, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[32, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        add_103: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_37: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_186, getitem_51);  mul_186 = None
        mul_187: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_188: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_187, primals_203);  mul_187 = None
        add_104: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_188, primals_204);  mul_188 = primals_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        view_266: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_104, [16384, 768]);  add_104 = None
        permute_133: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        constant_pad_nd_default_3: "f32[768, 30524]" = torch.ops.aten.constant_pad_nd.default(permute_133, [0, 2, 0, 0]);  permute_133 = None
        full_default_47: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[30524]" = torch.ops.aten.cat.default([primals_205, full_default_47]);  primals_205 = full_default_47 = None
        addmm_default: "f32[16384, 30524]" = torch.ops.aten.addmm.default(cat_default, view_266, constant_pad_nd_default_3);  cat_default = constant_pad_nd_default_3 = None
        slice_tensor_1: "f32[16384, 30522]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        view_267: "f32[32, 512, 30522]" = torch.ops.aten.reshape.default(slice_tensor_1, [32, 512, 30522]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:979 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_268: "f32[16384, 30522]" = torch.ops.aten.reshape.default(view_267, [-1, 30522])
        view_269: "i64[16384]" = torch.ops.aten.reshape.default(primals_206, [-1])
        amax_12: "f32[16384, 1]" = torch.ops.aten.amax.default(view_268, [1], True)
        sub_38: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = None
        exp_12: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_38)
        sum_13: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[16384, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = None
        ne: "b8[16384]" = torch.ops.aten.ne.Scalar(view_269, -100)
        full_default_36: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[16384]" = torch.ops.aten.where.self(ne, view_269, full_default_36);  view_269 = full_default_36 = None
        unsqueeze_3: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather_1: "f32[16384, 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_3);  sub_39 = unsqueeze_3 = None
        squeeze: "f32[16384]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg: "f32[16384]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_25: "f32[16384]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_14: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
        div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_15: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_16: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_155: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_156: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
        permute_157: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_158: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_17: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_18: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_188: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        permute_189: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        permute_190: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_191: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_19: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_20: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_221: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_222: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        permute_223: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_224: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_21: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_22: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_254: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_255: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
        permute_256: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_257: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_23: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_24: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_287: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_288: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_289: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_290: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_25: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_26: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_320: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        permute_321: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        permute_322: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_323: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_27: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_28: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_353: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_354: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        permute_355: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_356: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_29: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_30: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_386: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        permute_387: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
        permute_388: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_389: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_31: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_32: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_419: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_420: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        permute_421: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_422: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_33: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_34: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_452: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_453: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_454: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_455: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_35: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_36: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_485: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_486: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_487: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_488: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_37: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_38: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_518: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_519: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_520: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_521: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_39: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_12, view_267, primals_1, primals_2, primals_4, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_206, gather, mul, gt, ge, view, bmm, amax, sum_1, logical_not_1, gt_1, view_16, gt_2, mul_10, view_18, addmm_4, view_20, gt_3, mul_17, view_22, where_3, gt_4, view_38, gt_5, mul_25, view_40, addmm_10, view_42, gt_6, mul_32, view_44, where_5, gt_7, view_60, gt_8, mul_40, view_62, addmm_16, view_64, gt_9, mul_47, view_66, where_7, gt_10, view_82, gt_11, mul_55, view_84, addmm_22, view_86, gt_12, mul_62, view_88, where_9, gt_13, view_104, gt_14, mul_70, view_106, addmm_28, view_108, gt_15, mul_77, view_110, where_11, gt_16, view_126, gt_17, mul_85, view_128, addmm_34, view_130, gt_18, mul_92, view_132, where_13, gt_19, view_148, gt_20, mul_100, view_150, addmm_40, view_152, gt_21, mul_107, view_154, where_15, gt_22, view_170, gt_23, mul_115, view_172, addmm_46, view_174, gt_24, mul_122, view_176, where_17, gt_25, view_192, gt_26, mul_130, view_194, addmm_52, view_196, gt_27, mul_137, view_198, where_19, gt_28, view_214, gt_29, mul_145, view_216, addmm_58, view_218, gt_30, mul_152, view_220, where_21, gt_31, view_236, gt_32, mul_160, view_238, addmm_64, view_240, gt_33, mul_167, view_242, where_23, gt_34, view_258, gt_35, mul_175, view_260, addmm_70, view_262, gt_36, mul_182, view_264, addmm_72, getitem_51, rsqrt_25, view_266, view_267, amax_12, log, convert_element_type, div_15, div_16, permute_155, permute_156, permute_157, permute_158, div_17, div_18, permute_188, permute_189, permute_190, permute_191, div_19, div_20, permute_221, permute_222, permute_223, permute_224, div_21, div_22, permute_254, permute_255, permute_256, permute_257, div_23, div_24, permute_287, permute_288, permute_289, permute_290, div_25, div_26, permute_320, permute_321, permute_322, permute_323, div_27, div_28, permute_353, permute_354, permute_355, permute_356, div_29, div_30, permute_386, permute_387, permute_388, permute_389, div_31, div_32, permute_419, permute_420, permute_421, permute_422, div_33, div_34, permute_452, permute_453, permute_454, permute_455, div_35, div_36, permute_485, permute_486, permute_487, permute_488, div_37, div_38, permute_518, permute_519, permute_520, permute_521, div_39)
