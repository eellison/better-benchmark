class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024]", primals_2: "f32[50257, 768]", primals_3: "f32[1024, 768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[2304]", primals_7: "f32[768, 2304]", primals_8: "f32[768]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[3072]", primals_13: "f32[768, 3072]", primals_14: "f32[768]", primals_15: "f32[3072, 768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[2304]", primals_19: "f32[768, 2304]", primals_20: "f32[768]", primals_21: "f32[768, 768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[3072]", primals_25: "f32[768, 3072]", primals_26: "f32[768]", primals_27: "f32[3072, 768]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[2304]", primals_31: "f32[768, 2304]", primals_32: "f32[768]", primals_33: "f32[768, 768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[3072]", primals_37: "f32[768, 3072]", primals_38: "f32[768]", primals_39: "f32[3072, 768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[2304]", primals_43: "f32[768, 2304]", primals_44: "f32[768]", primals_45: "f32[768, 768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[3072]", primals_49: "f32[768, 3072]", primals_50: "f32[768]", primals_51: "f32[3072, 768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[2304]", primals_55: "f32[768, 2304]", primals_56: "f32[768]", primals_57: "f32[768, 768]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[3072]", primals_61: "f32[768, 3072]", primals_62: "f32[768]", primals_63: "f32[3072, 768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[2304]", primals_67: "f32[768, 2304]", primals_68: "f32[768]", primals_69: "f32[768, 768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[3072]", primals_73: "f32[768, 3072]", primals_74: "f32[768]", primals_75: "f32[3072, 768]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[2304]", primals_79: "f32[768, 2304]", primals_80: "f32[768]", primals_81: "f32[768, 768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[3072]", primals_85: "f32[768, 3072]", primals_86: "f32[768]", primals_87: "f32[3072, 768]", primals_88: "f32[768]", primals_89: "f32[768]", primals_90: "f32[2304]", primals_91: "f32[768, 2304]", primals_92: "f32[768]", primals_93: "f32[768, 768]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[3072]", primals_97: "f32[768, 3072]", primals_98: "f32[768]", primals_99: "f32[3072, 768]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[2304]", primals_103: "f32[768, 2304]", primals_104: "f32[768]", primals_105: "f32[768, 768]", primals_106: "f32[768]", primals_107: "f32[768]", primals_108: "f32[3072]", primals_109: "f32[768, 3072]", primals_110: "f32[768]", primals_111: "f32[3072, 768]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[2304]", primals_115: "f32[768, 2304]", primals_116: "f32[768]", primals_117: "f32[768, 768]", primals_118: "f32[768]", primals_119: "f32[768]", primals_120: "f32[3072]", primals_121: "f32[768, 3072]", primals_122: "f32[768]", primals_123: "f32[3072, 768]", primals_124: "f32[768]", primals_125: "f32[768]", primals_126: "f32[2304]", primals_127: "f32[768, 2304]", primals_128: "f32[768]", primals_129: "f32[768, 768]", primals_130: "f32[768]", primals_131: "f32[768]", primals_132: "f32[3072]", primals_133: "f32[768, 3072]", primals_134: "f32[768]", primals_135: "f32[3072, 768]", primals_136: "f32[768]", primals_137: "f32[768]", primals_138: "f32[2304]", primals_139: "f32[768, 2304]", primals_140: "f32[768]", primals_141: "f32[768, 768]", primals_142: "f32[768]", primals_143: "f32[768]", primals_144: "f32[3072]", primals_145: "f32[768, 3072]", primals_146: "f32[768]", primals_147: "f32[3072, 768]", primals_148: "f32[768]", primals_149: "f32[768]", primals_150: "f32[2, 768]", primals_151: "i64[8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:581 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:582 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_3, unsqueeze);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand: "i64[8, 1024]" = torch.ops.aten.expand.default(unsqueeze, [8, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[8, 1]" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub: "i64[8, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[8, 1025]" = torch.ops.aten.cat.default([sub, expand], -1);  sub = expand = None
        slice_2: "i64[8, 1024]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 1024)
        slice_3: "i64[8, 1024]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 1025);  cat = None
        sub_1: "i64[8, 1024]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[8, 1024]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1)
        unsqueeze_2: "i64[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_5: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        unsqueeze_6: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[8, 1, 1024, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[8, 1, 1, 1024]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[8, 1, 1024, 1024]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[8, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(bitwise_and_1, [8, -1, 1024, 1024]);  bitwise_and_1 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[25]" = torch.ops.prims.inductor_seeds.default(25, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_24: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(mul_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_4: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_1, getitem_1)
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_4);  mul_2 = None
        add_5: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_5);  mul_3 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_5, [-1, 768]);  add_5 = None
        addmm: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_6, view_1, primals_7);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_2: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm, [8, 1024, 2304]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_2, 768, 2);  view_2 = None
        getitem_2: "f32[8, 1024, 768]" = split[0]
        getitem_3: "f32[8, 1024, 768]" = split[1]
        getitem_4: "f32[8, 1024, 768]" = split[2];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_3: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_3, [8, 1024, -1, 64]);  getitem_3 = None
        permute: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_4: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [8, 1024, -1, 64]);  getitem_4 = None
        permute_1: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_5: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, [8, 1024, -1, 64]);  getitem_2 = None
        permute_2: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_2, full_default_1);  expand_1 = full_default_1 = None
        expand_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where, [8, 12, 1024, 1024])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand_2, True, 0.1, scale = 0.125)
        getitem_5: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention[0]
        getitem_6: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention[1]
        getitem_7: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_8: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_3, [8, 1024, -1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_6, [-1, 768]);  view_6 = None
        addmm_1: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_8, view_7, primals_9);  primals_8 = view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_8: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_1, [8, 1024, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_23: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_1, view_8);  view_8 = None
        mul_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_6: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_5, mul_1);  mul_5 = mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_9: "f32[8, 1024, 1]" = var_mean_1[0]
        getitem_10: "f32[8, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_6, getitem_10);  getitem_10 = None
        mul_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_6, primals_10)
        add_8: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_7, primals_11);  mul_7 = primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_9: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_8, [-1, 768]);  add_8 = None
        addmm_2: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_12, view_9, primals_13);  primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_10: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_2, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_8: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        pow_1: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_9: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_10, mul_9);  view_10 = mul_9 = None
        mul_10: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_10);  mul_10 = None
        add_10: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_11: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_8, add_10);  mul_8 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_11: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_11, [-1, 3072]);  mul_11 = None
        addmm_3: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_14, view_11, primals_15);  primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_12: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_3, [8, 1024, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_22: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_12: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_2, view_12);  view_12 = None
        mul_13: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_12, 1.1111111111111112);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_11: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_6, mul_13);  add_6 = mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_11: "f32[8, 1024, 1]" = var_mean_2[0]
        getitem_12: "f32[8, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        add_12: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_4: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_12);  getitem_12 = None
        mul_14: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_15: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_16)
        add_13: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_17);  mul_15 = primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_13: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_13, [-1, 768]);  add_13 = None
        addmm_4: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_18, view_13, primals_19);  primals_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_14: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_4, [8, 1024, 2304]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_14, 768, 2);  view_14 = None
        getitem_13: "f32[8, 1024, 768]" = split_1[0]
        getitem_14: "f32[8, 1024, 768]" = split_1[1]
        getitem_15: "f32[8, 1024, 768]" = split_1[2];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_15: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_14, [8, 1024, -1, 64]);  getitem_14 = None
        permute_4: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_16: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_15, [8, 1024, -1, 64]);  getitem_15 = None
        permute_5: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_17: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_13, [8, 1024, -1, 64]);  getitem_13 = None
        permute_6: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_2, True, 0.1, scale = 0.125)
        getitem_16: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_1[0]
        getitem_17: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention_1[1]
        getitem_18: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
        getitem_19: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_7, [8, 1024, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_18, [-1, 768]);  view_18 = None
        addmm_5: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_20, view_19, primals_21);  primals_20 = view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_20: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_5, [8, 1024, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_21: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_16: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_3, view_20);  view_20 = None
        mul_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, 1.1111111111111112);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_14: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_17, add_11);  mul_17 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 1024, 1]" = var_mean_3[0]
        getitem_21: "f32[8, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        add_15: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_5: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_21);  getitem_21 = None
        mul_18: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_19: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_22)
        add_16: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_23);  mul_19 = primals_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_21: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_16, [-1, 768]);  add_16 = None
        addmm_6: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_24, view_21, primals_25);  primals_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_22: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_6, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_2: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_21: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_22, mul_21);  view_22 = mul_21 = None
        mul_22: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_18: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_23: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_18);  mul_20 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_23: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_7: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_26, view_23, primals_27);  primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_24: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_7, [8, 1024, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_20: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_24: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_4, view_24);  view_24 = None
        mul_25: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_24, 1.1111111111111112);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_19: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_14, mul_25);  add_14 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 1024, 1]" = var_mean_4[0]
        getitem_23: "f32[8, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        add_20: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_6: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_23);  getitem_23 = None
        mul_26: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_27: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_28)
        add_21: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_29);  mul_27 = primals_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_25: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_21, [-1, 768]);  add_21 = None
        addmm_8: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_30, view_25, primals_31);  primals_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_26: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_8, [8, 1024, 2304]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_26, 768, 2);  view_26 = None
        getitem_24: "f32[8, 1024, 768]" = split_2[0]
        getitem_25: "f32[8, 1024, 768]" = split_2[1]
        getitem_26: "f32[8, 1024, 768]" = split_2[2];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_27: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_25, [8, 1024, -1, 64]);  getitem_25 = None
        permute_8: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_28: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_26, [8, 1024, -1, 64]);  getitem_26 = None
        permute_9: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_29: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_24, [8, 1024, -1, 64]);  getitem_24 = None
        permute_10: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_2, True, 0.1, scale = 0.125)
        getitem_27: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_2[0]
        getitem_28: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention_2[1]
        getitem_29: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
        getitem_30: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_11, [8, 1024, -1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_30, [-1, 768]);  view_30 = None
        addmm_9: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_32, view_31, primals_33);  primals_32 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_32: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_9, [8, 1024, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_19: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_28: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_5, view_32);  view_32 = None
        mul_29: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_28, 1.1111111111111112);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_22: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_29, add_19);  mul_29 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_31: "f32[8, 1024, 1]" = var_mean_5[0]
        getitem_32: "f32[8, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        add_23: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_7: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_32);  getitem_32 = None
        mul_30: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
        mul_31: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_34)
        add_24: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_35);  mul_31 = primals_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_33: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_24, [-1, 768]);  add_24 = None
        addmm_10: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_36, view_33, primals_37);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_34: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_32: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        pow_3: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_33: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_34, mul_33);  view_34 = mul_33 = None
        mul_34: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_34);  mul_34 = None
        add_26: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_35: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_26);  mul_32 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_35: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_35, [-1, 3072]);  mul_35 = None
        addmm_11: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_38, view_35, primals_39);  primals_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_36: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_11, [8, 1024, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_18: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_36: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_6, view_36);  view_36 = None
        mul_37: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_27: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_22, mul_37);  add_22 = mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_33: "f32[8, 1024, 1]" = var_mean_6[0]
        getitem_34: "f32[8, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        add_28: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_8: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_34);  getitem_34 = None
        mul_38: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_39: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_38, primals_40)
        add_29: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_39, primals_41);  mul_39 = primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_37: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_29, [-1, 768]);  add_29 = None
        addmm_12: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_42, view_37, primals_43);  primals_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_38: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_12, [8, 1024, 2304]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_38, 768, 2);  view_38 = None
        getitem_35: "f32[8, 1024, 768]" = split_3[0]
        getitem_36: "f32[8, 1024, 768]" = split_3[1]
        getitem_37: "f32[8, 1024, 768]" = split_3[2];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_39: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_36, [8, 1024, -1, 64]);  getitem_36 = None
        permute_12: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_40: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_37, [8, 1024, -1, 64]);  getitem_37 = None
        permute_13: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_41: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_35, [8, 1024, -1, 64]);  getitem_35 = None
        permute_14: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_2, True, 0.1, scale = 0.125)
        getitem_38: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_3[0]
        getitem_39: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention_3[1]
        getitem_40: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
        getitem_41: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_15, [8, 1024, -1]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_42, [-1, 768]);  view_42 = None
        addmm_13: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_44, view_43, primals_45);  primals_44 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_44: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_13, [8, 1024, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_17: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_40: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_7, view_44);  view_44 = None
        mul_41: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, 1.1111111111111112);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_30: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_41, add_27);  mul_41 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 1024, 1]" = var_mean_7[0]
        getitem_43: "f32[8, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        add_31: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_9: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_43);  getitem_43 = None
        mul_42: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = None
        mul_43: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_46)
        add_32: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_47);  mul_43 = primals_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_45: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_32, [-1, 768]);  add_32 = None
        addmm_14: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_48, view_45, primals_49);  primals_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_46: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_14, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        pow_4: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_45: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_33: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_46, mul_45);  view_46 = mul_45 = None
        mul_46: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_3: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_34: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_47: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_34);  mul_44 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_47: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_15: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_50, view_47, primals_51);  primals_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_48: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_15, [8, 1024, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_16: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_48: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_8, view_48);  view_48 = None
        mul_49: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_48, 1.1111111111111112);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_35: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_30, mul_49);  add_30 = mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 1024, 1]" = var_mean_8[0]
        getitem_45: "f32[8, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        add_36: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_10: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_45);  getitem_45 = None
        mul_50: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = None
        mul_51: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_52)
        add_37: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_53);  mul_51 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_49: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_37, [-1, 768]);  add_37 = None
        addmm_16: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_54, view_49, primals_55);  primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_50: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_16, [8, 1024, 2304]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_50, 768, 2);  view_50 = None
        getitem_46: "f32[8, 1024, 768]" = split_4[0]
        getitem_47: "f32[8, 1024, 768]" = split_4[1]
        getitem_48: "f32[8, 1024, 768]" = split_4[2];  split_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_51: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_47, [8, 1024, -1, 64]);  getitem_47 = None
        permute_16: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_52: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_48, [8, 1024, -1, 64]);  getitem_48 = None
        permute_17: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_53: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_46, [8, 1024, -1, 64]);  getitem_46 = None
        permute_18: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_2, True, 0.1, scale = 0.125)
        getitem_49: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_4[0]
        getitem_50: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention_4[1]
        getitem_51: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
        getitem_52: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_19, [8, 1024, -1]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_54, [-1, 768]);  view_54 = None
        addmm_17: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_56, view_55, primals_57);  primals_56 = view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_56: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_17, [8, 1024, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_15: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_52: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_9, view_56);  view_56 = None
        mul_53: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_52, 1.1111111111111112);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_38: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_53, add_35);  mul_53 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_53: "f32[8, 1024, 1]" = var_mean_9[0]
        getitem_54: "f32[8, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        add_39: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_11: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_54);  getitem_54 = None
        mul_54: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = None
        mul_55: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_54, primals_58)
        add_40: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_55, primals_59);  mul_55 = primals_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_57: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_40, [-1, 768]);  add_40 = None
        addmm_18: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_60, view_57, primals_61);  primals_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_58: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_18, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        pow_5: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_57: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_58, mul_57);  view_58 = mul_57 = None
        mul_58: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_42: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_59: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_56, add_42);  mul_56 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_59: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_59, [-1, 3072]);  mul_59 = None
        addmm_19: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_62, view_59, primals_63);  primals_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_60: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_19, [8, 1024, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_14: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_60: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_10, view_60);  view_60 = None
        mul_61: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_43: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_38, mul_61);  add_38 = mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_55: "f32[8, 1024, 1]" = var_mean_10[0]
        getitem_56: "f32[8, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        add_44: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_12: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_56);  getitem_56 = None
        mul_62: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = None
        mul_63: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_62, primals_64)
        add_45: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_63, primals_65);  mul_63 = primals_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_61: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_45, [-1, 768]);  add_45 = None
        addmm_20: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_66, view_61, primals_67);  primals_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_62: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_20, [8, 1024, 2304]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_62, 768, 2);  view_62 = None
        getitem_57: "f32[8, 1024, 768]" = split_5[0]
        getitem_58: "f32[8, 1024, 768]" = split_5[1]
        getitem_59: "f32[8, 1024, 768]" = split_5[2];  split_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_63: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_58, [8, 1024, -1, 64]);  getitem_58 = None
        permute_20: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_64: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_59, [8, 1024, -1, 64]);  getitem_59 = None
        permute_21: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_65: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_57, [8, 1024, -1, 64]);  getitem_57 = None
        permute_22: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_2, True, 0.1, scale = 0.125)
        getitem_60: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_5[0]
        getitem_61: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention_5[1]
        getitem_62: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
        getitem_63: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_23, [8, 1024, -1]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_66, [-1, 768]);  view_66 = None
        addmm_21: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_68, view_67, primals_69);  primals_68 = view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_68: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_21, [8, 1024, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_13: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_64: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_11, view_68);  view_68 = None
        mul_65: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_64, 1.1111111111111112);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_46: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_65, add_43);  mul_65 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_64: "f32[8, 1024, 1]" = var_mean_11[0]
        getitem_65: "f32[8, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        add_47: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_13: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_65);  getitem_65 = None
        mul_66: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = None
        mul_67: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_70)
        add_48: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_71);  mul_67 = primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_69: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_48, [-1, 768]);  add_48 = None
        addmm_22: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_72, view_69, primals_73);  primals_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_70: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        pow_6: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_69: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_49: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_70, mul_69);  view_70 = mul_69 = None
        mul_70: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_5: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_50: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_71: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_50);  mul_68 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_71: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_71, [-1, 3072]);  mul_71 = None
        addmm_23: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_74, view_71, primals_75);  primals_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_72: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_23, [8, 1024, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_12: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_72: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_12, view_72);  view_72 = None
        mul_73: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_72, 1.1111111111111112);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_51: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_46, mul_73);  add_46 = mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_66: "f32[8, 1024, 1]" = var_mean_12[0]
        getitem_67: "f32[8, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        add_52: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_14: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_67);  getitem_67 = None
        mul_74: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = None
        mul_75: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_76)
        add_53: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_77);  mul_75 = primals_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_73: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_53, [-1, 768]);  add_53 = None
        addmm_24: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_78, view_73, primals_79);  primals_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_74: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_24, [8, 1024, 2304]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_6 = torch.ops.aten.split.Tensor(view_74, 768, 2);  view_74 = None
        getitem_68: "f32[8, 1024, 768]" = split_6[0]
        getitem_69: "f32[8, 1024, 768]" = split_6[1]
        getitem_70: "f32[8, 1024, 768]" = split_6[2];  split_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_75: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_69, [8, 1024, -1, 64]);  getitem_69 = None
        permute_24: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_76: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_70, [8, 1024, -1, 64]);  getitem_70 = None
        permute_25: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_77: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_68, [8, 1024, -1, 64]);  getitem_68 = None
        permute_26: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_26, permute_24, permute_25, expand_2, True, 0.1, scale = 0.125)
        getitem_71: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_6[0]
        getitem_72: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention_6[1]
        getitem_73: "i64[]" = _scaled_dot_product_efficient_attention_6[2]
        getitem_74: "i64[]" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_78: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_27, [8, 1024, -1]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_79: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_78, [-1, 768]);  view_78 = None
        addmm_25: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_80, view_79, primals_81);  primals_80 = view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_80: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_25, [8, 1024, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_11: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_13: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_76: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_13, view_80);  view_80 = None
        mul_77: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_76, 1.1111111111111112);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_54: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_77, add_51);  mul_77 = add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
        getitem_75: "f32[8, 1024, 1]" = var_mean_13[0]
        getitem_76: "f32[8, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        add_55: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_15: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_54, getitem_76);  getitem_76 = None
        mul_78: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = None
        mul_79: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_78, primals_82)
        add_56: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_79, primals_83);  mul_79 = primals_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_81: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_56, [-1, 768]);  add_56 = None
        addmm_26: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_84, view_81, primals_85);  primals_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_82: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_26, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_80: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
        pow_7: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 3.0)
        mul_81: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_57: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_82, mul_81);  view_82 = mul_81 = None
        mul_82: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_6: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_82);  mul_82 = None
        add_58: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_83: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_80, add_58);  mul_80 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_83: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_83, [-1, 3072]);  mul_83 = None
        addmm_27: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_86, view_83, primals_87);  primals_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_84: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_27, [8, 1024, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_10: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_14: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_84: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_14, view_84);  view_84 = None
        mul_85: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_84, 1.1111111111111112);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_59: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_54, mul_85);  add_54 = mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_77: "f32[8, 1024, 1]" = var_mean_14[0]
        getitem_78: "f32[8, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        add_60: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_16: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_78);  getitem_78 = None
        mul_86: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = None
        mul_87: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_86, primals_88)
        add_61: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_87, primals_89);  mul_87 = primals_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_85: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_61, [-1, 768]);  add_61 = None
        addmm_28: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_90, view_85, primals_91);  primals_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_86: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_28, [8, 1024, 2304]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_7 = torch.ops.aten.split.Tensor(view_86, 768, 2);  view_86 = None
        getitem_79: "f32[8, 1024, 768]" = split_7[0]
        getitem_80: "f32[8, 1024, 768]" = split_7[1]
        getitem_81: "f32[8, 1024, 768]" = split_7[2];  split_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_87: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_80, [8, 1024, -1, 64]);  getitem_80 = None
        permute_28: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_88: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_81, [8, 1024, -1, 64]);  getitem_81 = None
        permute_29: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_89: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_79, [8, 1024, -1, 64]);  getitem_79 = None
        permute_30: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_30, permute_28, permute_29, expand_2, True, 0.1, scale = 0.125)
        getitem_82: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_7[0]
        getitem_83: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention_7[1]
        getitem_84: "i64[]" = _scaled_dot_product_efficient_attention_7[2]
        getitem_85: "i64[]" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_90: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_31, [8, 1024, -1]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_91: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_90, [-1, 768]);  view_90 = None
        addmm_29: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_92, view_91, primals_93);  primals_92 = view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_92: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_29, [8, 1024, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_9: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_15: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_88: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_15, view_92);  view_92 = None
        mul_89: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, 1.1111111111111112);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_62: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_89, add_59);  mul_89 = add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
        getitem_86: "f32[8, 1024, 1]" = var_mean_15[0]
        getitem_87: "f32[8, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        add_63: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_17: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_62, getitem_87);  getitem_87 = None
        mul_90: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_15);  sub_17 = None
        mul_91: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_94)
        add_64: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_95);  mul_91 = primals_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_93: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_64, [-1, 768]);  add_64 = None
        addmm_30: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_96, view_93, primals_97);  primals_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_94: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_30, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
        pow_8: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_94, 3.0)
        mul_93: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_65: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_94, mul_93);  view_94 = mul_93 = None
        mul_94: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_7: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_66: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_95: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_66);  mul_92 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_95: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_95, [-1, 3072]);  mul_95 = None
        addmm_31: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_98, view_95, primals_99);  primals_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_96: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_31, [8, 1024, 768]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_8: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_16: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_96: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_16, view_96);  view_96 = None
        mul_97: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_67: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_62, mul_97);  add_62 = mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_88: "f32[8, 1024, 1]" = var_mean_16[0]
        getitem_89: "f32[8, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        add_68: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_18: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_89);  getitem_89 = None
        mul_98: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_16);  sub_18 = None
        mul_99: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_98, primals_100)
        add_69: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_99, primals_101);  mul_99 = primals_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_97: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_69, [-1, 768]);  add_69 = None
        addmm_32: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_102, view_97, primals_103);  primals_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_98: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_32, [8, 1024, 2304]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_8 = torch.ops.aten.split.Tensor(view_98, 768, 2);  view_98 = None
        getitem_90: "f32[8, 1024, 768]" = split_8[0]
        getitem_91: "f32[8, 1024, 768]" = split_8[1]
        getitem_92: "f32[8, 1024, 768]" = split_8[2];  split_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_99: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_91, [8, 1024, -1, 64]);  getitem_91 = None
        permute_32: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_100: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_92, [8, 1024, -1, 64]);  getitem_92 = None
        permute_33: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_101: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_90, [8, 1024, -1, 64]);  getitem_90 = None
        permute_34: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, permute_32, permute_33, expand_2, True, 0.1, scale = 0.125)
        getitem_93: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_8[0]
        getitem_94: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention_8[1]
        getitem_95: "i64[]" = _scaled_dot_product_efficient_attention_8[2]
        getitem_96: "i64[]" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_102: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_35, [8, 1024, -1]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_103: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_102, [-1, 768]);  view_102 = None
        addmm_33: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_104, view_103, primals_105);  primals_104 = view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_104: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_33, [8, 1024, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_7: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_17: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_100: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_17, view_104);  view_104 = None
        mul_101: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_100, 1.1111111111111112);  mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_70: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_101, add_67);  mul_101 = add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_97: "f32[8, 1024, 1]" = var_mean_17[0]
        getitem_98: "f32[8, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        add_71: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_19: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_98);  getitem_98 = None
        mul_102: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_17);  sub_19 = None
        mul_103: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_102, primals_106)
        add_72: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_103, primals_107);  mul_103 = primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_105: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_72, [-1, 768]);  add_72 = None
        addmm_34: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_108, view_105, primals_109);  primals_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_106: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_104: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        pow_9: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 3.0)
        mul_105: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_73: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_106, mul_105);  view_106 = mul_105 = None
        mul_106: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_8: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_106);  mul_106 = None
        add_74: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_107: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_104, add_74);  mul_104 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_107: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_107, [-1, 3072]);  mul_107 = None
        addmm_35: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_110, view_107, primals_111);  primals_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_108: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_35, [8, 1024, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_6: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_18: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_108: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_18, view_108);  view_108 = None
        mul_109: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_108, 1.1111111111111112);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_75: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_70, mul_109);  add_70 = mul_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_99: "f32[8, 1024, 1]" = var_mean_18[0]
        getitem_100: "f32[8, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        add_76: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_20: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_100);  getitem_100 = None
        mul_110: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_18);  sub_20 = None
        mul_111: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_110, primals_112)
        add_77: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_111, primals_113);  mul_111 = primals_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_109: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_77, [-1, 768]);  add_77 = None
        addmm_36: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_114, view_109, primals_115);  primals_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_110: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_36, [8, 1024, 2304]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_9 = torch.ops.aten.split.Tensor(view_110, 768, 2);  view_110 = None
        getitem_101: "f32[8, 1024, 768]" = split_9[0]
        getitem_102: "f32[8, 1024, 768]" = split_9[1]
        getitem_103: "f32[8, 1024, 768]" = split_9[2];  split_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_111: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_102, [8, 1024, -1, 64]);  getitem_102 = None
        permute_36: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_112: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_103, [8, 1024, -1, 64]);  getitem_103 = None
        permute_37: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_113: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_101, [8, 1024, -1, 64]);  getitem_101 = None
        permute_38: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_38, permute_36, permute_37, expand_2, True, 0.1, scale = 0.125)
        getitem_104: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_9[0]
        getitem_105: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention_9[1]
        getitem_106: "i64[]" = _scaled_dot_product_efficient_attention_9[2]
        getitem_107: "i64[]" = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_114: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_39, [8, 1024, -1]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_115: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_114, [-1, 768]);  view_114 = None
        addmm_37: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_116, view_115, primals_117);  primals_116 = view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_116: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_37, [8, 1024, 768]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_5: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_19: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_112: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_19, view_116);  view_116 = None
        mul_113: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_112, 1.1111111111111112);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_78: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_113, add_75);  mul_113 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_108: "f32[8, 1024, 1]" = var_mean_19[0]
        getitem_109: "f32[8, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        add_79: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_21: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_78, getitem_109);  getitem_109 = None
        mul_114: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_19);  sub_21 = None
        mul_115: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_114, primals_118)
        add_80: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_115, primals_119);  mul_115 = primals_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_117: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_80, [-1, 768]);  add_80 = None
        addmm_38: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_120, view_117, primals_121);  primals_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_118: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_38, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        pow_10: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_118, 3.0)
        mul_117: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_81: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_118, mul_117);  view_118 = mul_117 = None
        mul_118: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_9: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_82: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_119: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_116, add_82);  mul_116 = add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_119: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_119, [-1, 3072]);  mul_119 = None
        addmm_39: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_122, view_119, primals_123);  primals_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_120: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_39, [8, 1024, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_4: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_20: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_120: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_20, view_120);  view_120 = None
        mul_121: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_83: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_78, mul_121);  add_78 = mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_110: "f32[8, 1024, 1]" = var_mean_20[0]
        getitem_111: "f32[8, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        add_84: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_22: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_111);  getitem_111 = None
        mul_122: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_20);  sub_22 = None
        mul_123: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_122, primals_124)
        add_85: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_123, primals_125);  mul_123 = primals_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_121: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_85, [-1, 768]);  add_85 = None
        addmm_40: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_126, view_121, primals_127);  primals_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_122: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_40, [8, 1024, 2304]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_10 = torch.ops.aten.split.Tensor(view_122, 768, 2);  view_122 = None
        getitem_112: "f32[8, 1024, 768]" = split_10[0]
        getitem_113: "f32[8, 1024, 768]" = split_10[1]
        getitem_114: "f32[8, 1024, 768]" = split_10[2];  split_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_123: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_113, [8, 1024, -1, 64]);  getitem_113 = None
        permute_40: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_124: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_114, [8, 1024, -1, 64]);  getitem_114 = None
        permute_41: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_125: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_112, [8, 1024, -1, 64]);  getitem_112 = None
        permute_42: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_40, permute_41, expand_2, True, 0.1, scale = 0.125)
        getitem_115: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_10[0]
        getitem_116: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention_10[1]
        getitem_117: "i64[]" = _scaled_dot_product_efficient_attention_10[2]
        getitem_118: "i64[]" = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_126: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_43, [8, 1024, -1]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_127: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_126, [-1, 768]);  view_126 = None
        addmm_41: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_128, view_127, primals_129);  primals_128 = view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_128: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_41, [8, 1024, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_3: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_21: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_124: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_21, view_128);  view_128 = None
        mul_125: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_124, 1.1111111111111112);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_86: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_125, add_83);  mul_125 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
        getitem_119: "f32[8, 1024, 1]" = var_mean_21[0]
        getitem_120: "f32[8, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        add_87: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_23: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_86, getitem_120);  getitem_120 = None
        mul_126: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_21);  sub_23 = None
        mul_127: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_126, primals_130)
        add_88: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_127, primals_131);  mul_127 = primals_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_129: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_88, [-1, 768]);  add_88 = None
        addmm_42: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_132, view_129, primals_133);  primals_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_130: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_42, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_128: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.5)
        pow_11: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_130, 3.0)
        mul_129: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_89: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_130, mul_129);  view_130 = mul_129 = None
        mul_130: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_10: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_130);  mul_130 = None
        add_90: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_131: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_128, add_90);  mul_128 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_131: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_131, [-1, 3072]);  mul_131 = None
        addmm_43: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_134, view_131, primals_135);  primals_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_132: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_43, [8, 1024, 768]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_2: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_22: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_132: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_22, view_132);  view_132 = None
        mul_133: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_132, 1.1111111111111112);  mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_91: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_86, mul_133);  add_86 = mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_121: "f32[8, 1024, 1]" = var_mean_22[0]
        getitem_122: "f32[8, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        add_92: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_24: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_122);  getitem_122 = None
        mul_134: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_22);  sub_24 = None
        mul_135: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_134, primals_136)
        add_93: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_135, primals_137);  mul_135 = primals_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_133: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_93, [-1, 768]);  add_93 = None
        addmm_44: "f32[8192, 2304]" = torch.ops.aten.addmm.default(primals_138, view_133, primals_139);  primals_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_134: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_44, [8, 1024, 2304]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_11 = torch.ops.aten.split.Tensor(view_134, 768, 2);  view_134 = None
        getitem_123: "f32[8, 1024, 768]" = split_11[0]
        getitem_124: "f32[8, 1024, 768]" = split_11[1]
        getitem_125: "f32[8, 1024, 768]" = split_11[2];  split_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_135: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_124, [8, 1024, -1, 64]);  getitem_124 = None
        permute_44: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_136: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_125, [8, 1024, -1, 64]);  getitem_125 = None
        permute_45: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_137: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_123, [8, 1024, -1, 64]);  getitem_123 = None
        permute_46: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_46, permute_44, permute_45, expand_2, True, 0.1, scale = 0.125);  expand_2 = None
        getitem_126: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_11[0]
        getitem_127: "f32[8, 12, 1024]" = _scaled_dot_product_efficient_attention_11[1]
        getitem_128: "i64[]" = _scaled_dot_product_efficient_attention_11[2]
        getitem_129: "i64[]" = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_138: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_47, [8, 1024, -1]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_139: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_138, [-1, 768]);  view_138 = None
        addmm_45: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_140, view_139, primals_141);  primals_140 = view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_140: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_45, [8, 1024, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_1: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_23: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_136: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_23, view_140);  view_140 = None
        mul_137: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_136, 1.1111111111111112);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_94: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_137, add_91);  mul_137 = add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
        getitem_130: "f32[8, 1024, 1]" = var_mean_23[0]
        getitem_131: "f32[8, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        add_95: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        sub_25: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_94, getitem_131);  getitem_131 = None
        mul_138: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_23);  sub_25 = None
        mul_139: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_138, primals_142)
        add_96: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_139, primals_143);  mul_139 = primals_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_141: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_96, [-1, 768]);  add_96 = None
        addmm_46: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_144, view_141, primals_145);  primals_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_142: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_46, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        pow_12: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 3.0)
        mul_141: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_97: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_142, mul_141);  view_142 = mul_141 = None
        mul_142: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_11: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_98: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_143: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_140, add_98);  mul_140 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_143: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_143, [-1, 3072]);  mul_143 = None
        addmm_47: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_146, view_143, primals_147);  primals_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_144: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_47, [8, 1024, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_24: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_144: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_24, view_144);  view_144 = None
        mul_145: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_99: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_94, mul_145);  add_94 = mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_132: "f32[8, 1024, 1]" = var_mean_24[0]
        getitem_133: "f32[8, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
        add_100: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_26: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_133);  add_99 = getitem_133 = None
        mul_146: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_24);  sub_26 = None
        mul_147: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_146, primals_148)
        add_101: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_147, primals_149);  mul_147 = primals_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:920 in forward, code: logits = self.score(hidden_states)
        permute_48: "f32[768, 2]" = torch.ops.aten.permute.default(primals_150, [1, 0])
        view_146: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_101, [8192, 768]);  add_101 = None
        mm: "f32[8192, 2]" = torch.ops.aten.mm.default(view_146, permute_48);  permute_48 = None
        view_147: "f32[8, 1024, 2]" = torch.ops.aten.reshape.default(mm, [8, 1024, 2]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:933 in forward, code: non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
        ne_1: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(primals_1, 0)
        convert_element_type: "i32[8, 1024]" = torch.ops.prims.convert_element_type.default(ne_1, torch.int32);  ne_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:934 in forward, code: token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
        iota_5: "i32[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:935 in forward, code: last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        mul_148: "i32[8, 1024]" = torch.ops.aten.mul.Tensor(iota_5, convert_element_type);  iota_5 = convert_element_type = None
        argmax: "i64[8]" = torch.ops.aten.argmax.default(mul_148, -1);  mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:943 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        index_2: "f32[8, 2]" = torch.ops.aten.index.Tensor(view_147, [iota_1, argmax]);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:963 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        amax: "f32[8, 1]" = torch.ops.aten.amax.default(index_2, [1], True)
        sub_27: "f32[8, 2]" = torch.ops.aten.sub.Tensor(index_2, amax);  amax = None
        exp: "f32[8, 2]" = torch.ops.aten.exp.default(sub_27)
        sum_1: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_28: "f32[8, 2]" = torch.ops.aten.sub.Tensor(sub_27, log);  sub_27 = log = None
        ne_2: "b8[8]" = torch.ops.aten.ne.Scalar(primals_151, -100)
        full_default_25: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[8]" = torch.ops.aten.where.self(ne_2, primals_151, full_default_25)
        unsqueeze_10: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather: "f32[8, 1]" = torch.ops.aten.gather.default(sub_28, 1, unsqueeze_10);  sub_28 = unsqueeze_10 = None
        squeeze: "f32[8]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_13: "f32[8]" = torch.ops.aten.where.self(ne_2, neg, full_default_2);  neg = full_default_2 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_1);  sum_3 = None
        unsqueeze_11: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(primals_151, 1);  primals_151 = None
        ne_5: "b8[8, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_11, -100)
        where_14: "i64[8, 1]" = torch.ops.aten.where.self(ne_5, unsqueeze_11, full_default_25);  unsqueeze_11 = full_default_25 = None

        # No stacktrace found for following nodes
        iota_default: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 2]" = torch.ops.aten.reshape.default(iota_default, [1, 2]);  iota_default = None
        expand_default: "i64[8, 2]" = torch.ops.aten.expand.default(where_14, [8, 2]);  where_14 = None
        eq_tensor: "b8[8, 2]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        div_2: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_54: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_143, [1, 0]);  view_143 = None
        permute_56: "f32[768, 8192]" = torch.ops.aten.permute.default(view_141, [1, 0]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_3: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_64: "f32[768, 8192]" = torch.ops.aten.permute.default(view_133, [1, 0]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_4: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_66: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_131, [1, 0]);  view_131 = None
        permute_68: "f32[768, 8192]" = torch.ops.aten.permute.default(view_129, [1, 0]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_5: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_76: "f32[768, 8192]" = torch.ops.aten.permute.default(view_121, [1, 0]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_6: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_78: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_119, [1, 0]);  view_119 = None
        permute_80: "f32[768, 8192]" = torch.ops.aten.permute.default(view_117, [1, 0]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_7: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_88: "f32[768, 8192]" = torch.ops.aten.permute.default(view_109, [1, 0]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_8: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_90: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_107, [1, 0]);  view_107 = None
        permute_92: "f32[768, 8192]" = torch.ops.aten.permute.default(view_105, [1, 0]);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_9: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_100: "f32[768, 8192]" = torch.ops.aten.permute.default(view_97, [1, 0]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_10: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_102: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_95, [1, 0]);  view_95 = None
        permute_104: "f32[768, 8192]" = torch.ops.aten.permute.default(view_93, [1, 0]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_11: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_112: "f32[768, 8192]" = torch.ops.aten.permute.default(view_85, [1, 0]);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_12: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_114: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_83, [1, 0]);  view_83 = None
        permute_116: "f32[768, 8192]" = torch.ops.aten.permute.default(view_81, [1, 0]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_13: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_124: "f32[768, 8192]" = torch.ops.aten.permute.default(view_73, [1, 0]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_14: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_126: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_71, [1, 0]);  view_71 = None
        permute_128: "f32[768, 8192]" = torch.ops.aten.permute.default(view_69, [1, 0]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_15: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_136: "f32[768, 8192]" = torch.ops.aten.permute.default(view_61, [1, 0]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_16: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_138: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_59, [1, 0]);  view_59 = None
        permute_140: "f32[768, 8192]" = torch.ops.aten.permute.default(view_57, [1, 0]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_17: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_148: "f32[768, 8192]" = torch.ops.aten.permute.default(view_49, [1, 0]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_18: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_150: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_47, [1, 0]);  view_47 = None
        permute_152: "f32[768, 8192]" = torch.ops.aten.permute.default(view_45, [1, 0]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_19: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_160: "f32[768, 8192]" = torch.ops.aten.permute.default(view_37, [1, 0]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_20: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_162: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_35, [1, 0]);  view_35 = None
        permute_164: "f32[768, 8192]" = torch.ops.aten.permute.default(view_33, [1, 0]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_21: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_172: "f32[768, 8192]" = torch.ops.aten.permute.default(view_25, [1, 0]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_22: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_174: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_23, [1, 0]);  view_23 = None
        permute_176: "f32[768, 8192]" = torch.ops.aten.permute.default(view_21, [1, 0]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_23: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_184: "f32[768, 8192]" = torch.ops.aten.permute.default(view_13, [1, 0]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_24: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_186: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_11, [1, 0]);  view_11 = None
        permute_188: "f32[768, 8192]" = torch.ops.aten.permute.default(view_9, [1, 0]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_25: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_196: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0]);  view_1 = None
        return (div, index_2, primals_1, primals_4, primals_7, primals_9, primals_10, primals_13, primals_15, primals_16, primals_19, primals_21, primals_22, primals_25, primals_27, primals_28, primals_31, primals_33, primals_34, primals_37, primals_39, primals_40, primals_43, primals_45, primals_46, primals_49, primals_51, primals_52, primals_55, primals_57, primals_58, primals_61, primals_63, primals_64, primals_67, primals_69, primals_70, primals_73, primals_75, primals_76, primals_79, primals_81, primals_82, primals_85, primals_87, primals_88, primals_91, primals_93, primals_94, primals_97, primals_99, primals_100, primals_103, primals_105, primals_106, primals_109, primals_111, primals_112, primals_115, primals_117, primals_118, primals_121, primals_123, primals_124, primals_127, primals_129, primals_130, primals_133, primals_135, primals_136, primals_139, primals_141, primals_142, primals_145, primals_147, primals_148, primals_150, embedding, unsqueeze, embedding_1, iota_1, gt, getitem_1, rsqrt, permute, permute_1, permute_2, where, getitem_5, getitem_6, getitem_7, getitem_8, gt_1, mul_6, addmm_2, gt_2, mul_14, permute_4, permute_5, permute_6, getitem_16, getitem_17, getitem_18, getitem_19, gt_3, mul_18, addmm_6, gt_4, mul_26, permute_8, permute_9, permute_10, getitem_27, getitem_28, getitem_29, getitem_30, gt_5, mul_30, addmm_10, gt_6, mul_38, permute_12, permute_13, permute_14, getitem_38, getitem_39, getitem_40, getitem_41, gt_7, mul_42, addmm_14, gt_8, mul_50, permute_16, permute_17, permute_18, getitem_49, getitem_50, getitem_51, getitem_52, gt_9, mul_54, addmm_18, gt_10, mul_62, permute_20, permute_21, permute_22, getitem_60, getitem_61, getitem_62, getitem_63, gt_11, mul_66, addmm_22, gt_12, mul_74, permute_24, permute_25, permute_26, getitem_71, getitem_72, getitem_73, getitem_74, gt_13, mul_78, addmm_26, gt_14, mul_86, permute_28, permute_29, permute_30, getitem_82, getitem_83, getitem_84, getitem_85, gt_15, mul_90, addmm_30, gt_16, mul_98, permute_32, permute_33, permute_34, getitem_93, getitem_94, getitem_95, getitem_96, gt_17, mul_102, addmm_34, gt_18, mul_110, permute_36, permute_37, permute_38, getitem_104, getitem_105, getitem_106, getitem_107, gt_19, mul_114, addmm_38, gt_20, mul_122, permute_40, permute_41, permute_42, getitem_115, getitem_116, getitem_117, getitem_118, gt_21, mul_126, addmm_42, gt_22, mul_134, permute_44, permute_45, permute_46, getitem_126, getitem_127, getitem_128, getitem_129, gt_23, mul_138, addmm_46, gt_24, mul_146, view_146, argmax, index_2, convert_element_type_1, ne_5, eq_tensor, div_2, permute_54, permute_56, div_3, permute_64, div_4, permute_66, permute_68, div_5, permute_76, div_6, permute_78, permute_80, div_7, permute_88, div_8, permute_90, permute_92, div_9, permute_100, div_10, permute_102, permute_104, div_11, permute_112, div_12, permute_114, permute_116, div_13, permute_124, div_14, permute_126, permute_128, div_15, permute_136, div_16, permute_138, permute_140, div_17, permute_148, div_18, permute_150, permute_152, div_19, permute_160, div_20, permute_162, permute_164, div_21, permute_172, div_22, permute_174, permute_176, div_23, permute_184, div_24, permute_186, permute_188, div_25, permute_196)
