class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024][1024, 1]cuda:0", primals_2: "f32[50257, 768][768, 1]cuda:0", primals_3: "f32[1024, 768][768, 1]cuda:0", primals_4: "f32[768][1]cuda:0", primals_5: "f32[768][1]cuda:0", primals_6: "f32[2304][1]cuda:0", primals_7: "f32[768, 2304][2304, 1]cuda:0", primals_8: "f32[768][1]cuda:0", primals_9: "f32[768, 768][768, 1]cuda:0", primals_10: "f32[768][1]cuda:0", primals_11: "f32[768][1]cuda:0", primals_12: "f32[3072][1]cuda:0", primals_13: "f32[768, 3072][3072, 1]cuda:0", primals_14: "f32[768][1]cuda:0", primals_15: "f32[3072, 768][768, 1]cuda:0", primals_16: "f32[768][1]cuda:0", primals_17: "f32[768][1]cuda:0", primals_18: "f32[2304][1]cuda:0", primals_19: "f32[768, 2304][2304, 1]cuda:0", primals_20: "f32[768][1]cuda:0", primals_21: "f32[768, 768][768, 1]cuda:0", primals_22: "f32[768][1]cuda:0", primals_23: "f32[768][1]cuda:0", primals_24: "f32[3072][1]cuda:0", primals_25: "f32[768, 3072][3072, 1]cuda:0", primals_26: "f32[768][1]cuda:0", primals_27: "f32[3072, 768][768, 1]cuda:0", primals_28: "f32[768][1]cuda:0", primals_29: "f32[768][1]cuda:0", primals_30: "f32[2304][1]cuda:0", primals_31: "f32[768, 2304][2304, 1]cuda:0", primals_32: "f32[768][1]cuda:0", primals_33: "f32[768, 768][768, 1]cuda:0", primals_34: "f32[768][1]cuda:0", primals_35: "f32[768][1]cuda:0", primals_36: "f32[3072][1]cuda:0", primals_37: "f32[768, 3072][3072, 1]cuda:0", primals_38: "f32[768][1]cuda:0", primals_39: "f32[3072, 768][768, 1]cuda:0", primals_40: "f32[768][1]cuda:0", primals_41: "f32[768][1]cuda:0", primals_42: "f32[2304][1]cuda:0", primals_43: "f32[768, 2304][2304, 1]cuda:0", primals_44: "f32[768][1]cuda:0", primals_45: "f32[768, 768][768, 1]cuda:0", primals_46: "f32[768][1]cuda:0", primals_47: "f32[768][1]cuda:0", primals_48: "f32[3072][1]cuda:0", primals_49: "f32[768, 3072][3072, 1]cuda:0", primals_50: "f32[768][1]cuda:0", primals_51: "f32[3072, 768][768, 1]cuda:0", primals_52: "f32[768][1]cuda:0", primals_53: "f32[768][1]cuda:0", primals_54: "f32[2304][1]cuda:0", primals_55: "f32[768, 2304][2304, 1]cuda:0", primals_56: "f32[768][1]cuda:0", primals_57: "f32[768, 768][768, 1]cuda:0", primals_58: "f32[768][1]cuda:0", primals_59: "f32[768][1]cuda:0", primals_60: "f32[3072][1]cuda:0", primals_61: "f32[768, 3072][3072, 1]cuda:0", primals_62: "f32[768][1]cuda:0", primals_63: "f32[3072, 768][768, 1]cuda:0", primals_64: "f32[768][1]cuda:0", primals_65: "f32[768][1]cuda:0", primals_66: "f32[2304][1]cuda:0", primals_67: "f32[768, 2304][2304, 1]cuda:0", primals_68: "f32[768][1]cuda:0", primals_69: "f32[768, 768][768, 1]cuda:0", primals_70: "f32[768][1]cuda:0", primals_71: "f32[768][1]cuda:0", primals_72: "f32[3072][1]cuda:0", primals_73: "f32[768, 3072][3072, 1]cuda:0", primals_74: "f32[768][1]cuda:0", primals_75: "f32[3072, 768][768, 1]cuda:0", primals_76: "f32[768][1]cuda:0", primals_77: "f32[768][1]cuda:0", primals_78: "f32[2304][1]cuda:0", primals_79: "f32[768, 2304][2304, 1]cuda:0", primals_80: "f32[768][1]cuda:0", primals_81: "f32[768, 768][768, 1]cuda:0", primals_82: "f32[768][1]cuda:0", primals_83: "f32[768][1]cuda:0", primals_84: "f32[3072][1]cuda:0", primals_85: "f32[768, 3072][3072, 1]cuda:0", primals_86: "f32[768][1]cuda:0", primals_87: "f32[3072, 768][768, 1]cuda:0", primals_88: "f32[768][1]cuda:0", primals_89: "f32[768][1]cuda:0", primals_90: "f32[2304][1]cuda:0", primals_91: "f32[768, 2304][2304, 1]cuda:0", primals_92: "f32[768][1]cuda:0", primals_93: "f32[768, 768][768, 1]cuda:0", primals_94: "f32[768][1]cuda:0", primals_95: "f32[768][1]cuda:0", primals_96: "f32[3072][1]cuda:0", primals_97: "f32[768, 3072][3072, 1]cuda:0", primals_98: "f32[768][1]cuda:0", primals_99: "f32[3072, 768][768, 1]cuda:0", primals_100: "f32[768][1]cuda:0", primals_101: "f32[768][1]cuda:0", primals_102: "f32[2304][1]cuda:0", primals_103: "f32[768, 2304][2304, 1]cuda:0", primals_104: "f32[768][1]cuda:0", primals_105: "f32[768, 768][768, 1]cuda:0", primals_106: "f32[768][1]cuda:0", primals_107: "f32[768][1]cuda:0", primals_108: "f32[3072][1]cuda:0", primals_109: "f32[768, 3072][3072, 1]cuda:0", primals_110: "f32[768][1]cuda:0", primals_111: "f32[3072, 768][768, 1]cuda:0", primals_112: "f32[768][1]cuda:0", primals_113: "f32[768][1]cuda:0", primals_114: "f32[2304][1]cuda:0", primals_115: "f32[768, 2304][2304, 1]cuda:0", primals_116: "f32[768][1]cuda:0", primals_117: "f32[768, 768][768, 1]cuda:0", primals_118: "f32[768][1]cuda:0", primals_119: "f32[768][1]cuda:0", primals_120: "f32[3072][1]cuda:0", primals_121: "f32[768, 3072][3072, 1]cuda:0", primals_122: "f32[768][1]cuda:0", primals_123: "f32[3072, 768][768, 1]cuda:0", primals_124: "f32[768][1]cuda:0", primals_125: "f32[768][1]cuda:0", primals_126: "f32[2304][1]cuda:0", primals_127: "f32[768, 2304][2304, 1]cuda:0", primals_128: "f32[768][1]cuda:0", primals_129: "f32[768, 768][768, 1]cuda:0", primals_130: "f32[768][1]cuda:0", primals_131: "f32[768][1]cuda:0", primals_132: "f32[3072][1]cuda:0", primals_133: "f32[768, 3072][3072, 1]cuda:0", primals_134: "f32[768][1]cuda:0", primals_135: "f32[3072, 768][768, 1]cuda:0", primals_136: "f32[768][1]cuda:0", primals_137: "f32[768][1]cuda:0", primals_138: "f32[2304][1]cuda:0", primals_139: "f32[768, 2304][2304, 1]cuda:0", primals_140: "f32[768][1]cuda:0", primals_141: "f32[768, 768][768, 1]cuda:0", primals_142: "f32[768][1]cuda:0", primals_143: "f32[768][1]cuda:0", primals_144: "f32[3072][1]cuda:0", primals_145: "f32[768, 3072][3072, 1]cuda:0", primals_146: "f32[768][1]cuda:0", primals_147: "f32[3072, 768][768, 1]cuda:0", primals_148: "f32[768][1]cuda:0", primals_149: "f32[768][1]cuda:0", primals_150: "f32[2, 768][768, 1]cuda:0", primals_151: "i64[8][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:581 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:582 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_3, unsqueeze);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_1: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand: "i64[8, 1024][0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze, [8, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        full_default: "i64[8, 1][1, 1]cuda:0" = torch.ops.aten.full.default([8, 1], -1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[8, 1025][1025, 1]cuda:0" = torch.ops.aten.cat.default([full_default, expand], -1);  full_default = expand = None
        slice_2: "i64[8, 1024][1025, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat, -1, 0, 1024)
        slice_3: "i64[8, 1024][1025, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat, -1, 1, 1025);  cat = None
        sub_1: "i64[8, 1024][1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[8, 1024][1024, 1]cuda:0" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[8][1]cuda:0" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[8, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_1, 1)
        unsqueeze_2: "i64[8, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[8, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_5: "i64[1, 1, 1024][1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        unsqueeze_6: "i64[1, 1, 1024, 1][1024, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 1, 1, 1024][1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full, le);  full = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[8, 1, 1024, 1][1024, 1024, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[8, 1, 1, 1024][1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(bitwise_and_1, [8, -1, 1024, 1024]);  bitwise_and_1 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[25][1]cuda:0" = torch.ops.prims.inductor_seeds.default(25, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_24: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_1: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(mul_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_1, getitem_1)
        mul_2: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_3: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, primals_4);  mul_2 = None
        add_5: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, primals_5);  mul_3 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_1: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_5, [-1, 768]);  add_5 = None
        convert_element_type: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_1: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_2: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        addmm: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_2, convert_element_type_1, convert_element_type);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_2: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [8, 1024, 2304]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_2, 768, 2);  view_2 = None
        getitem_2: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split[0]
        getitem_3: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split[1]
        getitem_4: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split[2];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_3: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_3, [8, 1024, -1, 64]);  getitem_3 = None
        permute: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_4: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_4, [8, 1024, -1, 64]);  getitem_4 = None
        permute_1: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_5: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_2, [8, 1024, -1, 64]);  getitem_2 = None
        permute_2: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_2, full_default_1);  expand_1 = full_default_2 = full_default_1 = None
        expand_2: "bf16[8, 12, 1024, 1024][1048576, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(where, [8, 12, 1024, 1024])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand_2, True, 0.1, scale = 0.125)
        getitem_5: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention[0]
        getitem_6: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention[1]
        getitem_7: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention[2]
        getitem_8: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_3, [8, 1024, -1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [-1, 768]);  view_6 = None
        convert_element_type_6: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_7: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        addmm_1: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_7, view_7, convert_element_type_6);  convert_element_type_7 = view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_8: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [8, 1024, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_23: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_23: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_23, torch.bfloat16);  inductor_random_default_23 = None
        gt_1: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_23, 0.1);  convert_element_type_default_23 = None
        mul_4: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, view_8);  view_8 = None
        mul_5: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_6: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, mul_1);  mul_5 = mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_1[0]
        getitem_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_6, getitem_10);  getitem_10 = None
        mul_6: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_7: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, primals_10)
        add_8: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_7, primals_11);  mul_7 = primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_9: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_8, [-1, 768]);  add_8 = None
        convert_element_type_11: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_12: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        convert_element_type_13: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        addmm_2: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_13, convert_element_type_12, convert_element_type_11);  convert_element_type_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_10: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_8: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        convert_element_type_17: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_10, torch.float32)
        pow_1: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_17, 3.0);  convert_element_type_17 = None
        mul_9: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_10, mul_9);  view_10 = mul_9 = None
        mul_10: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_10);  mul_10 = None
        add_10: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_11: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, add_10);  mul_8 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_11: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_11, [-1, 3072]);  mul_11 = None
        convert_element_type_18: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_19: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        convert_element_type_20: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        addmm_3: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_20, convert_element_type_19, convert_element_type_18);  convert_element_type_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_12: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [8, 1024, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_22: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_22: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_22, torch.bfloat16);  inductor_random_default_22 = None
        gt_2: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_22, 0.1);  convert_element_type_default_22 = None
        mul_12: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_2, view_12);  view_12 = None
        mul_13: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, 1.1111111111111112);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_11: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_6, mul_13);  add_6 = mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_2[0]
        getitem_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_4: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_11, getitem_12);  getitem_12 = None
        mul_14: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_15: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, primals_16)
        add_13: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, primals_17);  mul_15 = primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_13: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_13, [-1, 768]);  add_13 = None
        convert_element_type_24: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_25: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.bfloat16);  view_13 = None
        convert_element_type_26: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        addmm_4: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_26, convert_element_type_25, convert_element_type_24);  convert_element_type_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_14: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [8, 1024, 2304]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_14, 768, 2);  view_14 = None
        getitem_13: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_1[0]
        getitem_14: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_1[1]
        getitem_15: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_1[2];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_15: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_14, [8, 1024, -1, 64]);  getitem_14 = None
        permute_4: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_16: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_15, [8, 1024, -1, 64]);  getitem_15 = None
        permute_5: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_17: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_13, [8, 1024, -1, 64]);  getitem_13 = None
        permute_6: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_2, True, 0.1, scale = 0.125)
        getitem_16: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_1[0]
        getitem_17: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_1[1]
        getitem_18: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_1[2]
        getitem_19: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_7, [8, 1024, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_18, [-1, 768]);  view_18 = None
        convert_element_type_30: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_31: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        addmm_5: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_31, view_19, convert_element_type_30);  convert_element_type_31 = view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_20: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [8, 1024, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_3: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_21: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_21: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_21, torch.bfloat16);  inductor_random_default_21 = None
        gt_3: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_21, 0.1);  convert_element_type_default_21 = None
        mul_16: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_3, view_20);  view_20 = None
        mul_17: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, 1.1111111111111112);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_14: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, add_11);  mul_17 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_3[0]
        getitem_21: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_5: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_14, getitem_21);  getitem_21 = None
        mul_18: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_19: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, primals_22)
        add_16: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, primals_23);  mul_19 = primals_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_21: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_16, [-1, 768]);  add_16 = None
        convert_element_type_35: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_36: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_21, torch.bfloat16);  view_21 = None
        convert_element_type_37: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        addmm_6: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_37, convert_element_type_36, convert_element_type_35);  convert_element_type_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_22: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        convert_element_type_41: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_22, torch.float32)
        pow_2: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_41, 3.0);  convert_element_type_41 = None
        mul_21: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_22, mul_21);  view_22 = mul_21 = None
        mul_22: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_18: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_23: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, add_18);  mul_20 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_23: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_23, [-1, 3072]);  mul_23 = None
        convert_element_type_42: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        convert_element_type_43: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_23, torch.bfloat16);  view_23 = None
        convert_element_type_44: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        addmm_7: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_44, convert_element_type_43, convert_element_type_42);  convert_element_type_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_24: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [8, 1024, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_4: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_20: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_20: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_20, torch.bfloat16);  inductor_random_default_20 = None
        gt_4: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_20, 0.1);  convert_element_type_default_20 = None
        mul_24: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_4, view_24);  view_24 = None
        mul_25: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, 1.1111111111111112);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_19: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_14, mul_25);  add_14 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_4[0]
        getitem_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_6: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_19, getitem_23);  getitem_23 = None
        mul_26: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_27: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, primals_28)
        add_21: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, primals_29);  mul_27 = primals_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_25: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_21, [-1, 768]);  add_21 = None
        convert_element_type_48: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        convert_element_type_49: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_25, torch.bfloat16);  view_25 = None
        convert_element_type_50: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        addmm_8: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_50, convert_element_type_49, convert_element_type_48);  convert_element_type_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_26: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [8, 1024, 2304]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_26, 768, 2);  view_26 = None
        getitem_24: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_2[0]
        getitem_25: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_2[1]
        getitem_26: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_2[2];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_27: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_25, [8, 1024, -1, 64]);  getitem_25 = None
        permute_8: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_28: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_26, [8, 1024, -1, 64]);  getitem_26 = None
        permute_9: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_29: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_24, [8, 1024, -1, 64]);  getitem_24 = None
        permute_10: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_2, True, 0.1, scale = 0.125)
        getitem_27: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_2[0]
        getitem_28: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_2[1]
        getitem_29: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_2[2]
        getitem_30: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_11, [8, 1024, -1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [-1, 768]);  view_30 = None
        convert_element_type_54: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        convert_element_type_55: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        addmm_9: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_55, view_31, convert_element_type_54);  convert_element_type_55 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_32: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [8, 1024, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_5: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_19: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_19: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_19, torch.bfloat16);  inductor_random_default_19 = None
        gt_5: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_19, 0.1);  convert_element_type_default_19 = None
        mul_28: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_5, view_32);  view_32 = None
        mul_29: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, 1.1111111111111112);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_22: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, add_19);  mul_29 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_31: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_5[0]
        getitem_32: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_7: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_22, getitem_32);  getitem_32 = None
        mul_30: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
        mul_31: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, primals_34)
        add_24: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, primals_35);  mul_31 = primals_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_33: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_24, [-1, 768]);  add_24 = None
        convert_element_type_59: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convert_element_type_60: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.bfloat16);  view_33 = None
        convert_element_type_61: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        addmm_10: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_61, convert_element_type_60, convert_element_type_59);  convert_element_type_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_34: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_32: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        convert_element_type_65: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_34, torch.float32)
        pow_3: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_65, 3.0);  convert_element_type_65 = None
        mul_33: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_34, mul_33);  view_34 = mul_33 = None
        mul_34: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_34);  mul_34 = None
        add_26: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_35: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, add_26);  mul_32 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_35: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_35, [-1, 3072]);  mul_35 = None
        convert_element_type_66: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        convert_element_type_67: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_35, torch.bfloat16);  view_35 = None
        convert_element_type_68: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        addmm_11: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_68, convert_element_type_67, convert_element_type_66);  convert_element_type_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_36: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [8, 1024, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_18: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        convert_element_type_default_18: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_18, torch.bfloat16);  inductor_random_default_18 = None
        gt_6: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_18, 0.1);  convert_element_type_default_18 = None
        mul_36: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_6, view_36);  view_36 = None
        mul_37: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_27: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_22, mul_37);  add_22 = mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_33: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_6[0]
        getitem_34: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_28: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_8: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_27, getitem_34);  getitem_34 = None
        mul_38: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_39: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, primals_40)
        add_29: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, primals_41);  mul_39 = primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_37: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_29, [-1, 768]);  add_29 = None
        convert_element_type_72: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        convert_element_type_73: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_37, torch.bfloat16);  view_37 = None
        convert_element_type_74: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        addmm_12: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_74, convert_element_type_73, convert_element_type_72);  convert_element_type_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_38: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [8, 1024, 2304]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_38, 768, 2);  view_38 = None
        getitem_35: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_3[0]
        getitem_36: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_3[1]
        getitem_37: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_3[2];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_39: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_36, [8, 1024, -1, 64]);  getitem_36 = None
        permute_12: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_40: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_37, [8, 1024, -1, 64]);  getitem_37 = None
        permute_13: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_41: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_35, [8, 1024, -1, 64]);  getitem_35 = None
        permute_14: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_2, True, 0.1, scale = 0.125)
        getitem_38: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_3[0]
        getitem_39: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_3[1]
        getitem_40: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_3[2]
        getitem_41: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_15, [8, 1024, -1]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [-1, 768]);  view_42 = None
        convert_element_type_78: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_79: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        addmm_13: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_79, view_43, convert_element_type_78);  convert_element_type_79 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_44: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [8, 1024, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_7: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_17: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_17: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_17, torch.bfloat16);  inductor_random_default_17 = None
        gt_7: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_17, 0.1);  convert_element_type_default_17 = None
        mul_40: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_7, view_44);  view_44 = None
        mul_41: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, 1.1111111111111112);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_30: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, add_27);  mul_41 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_7[0]
        getitem_43: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_31: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_9: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_30, getitem_43);  getitem_43 = None
        mul_42: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = None
        mul_43: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, primals_46)
        add_32: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, primals_47);  mul_43 = primals_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_45: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_32, [-1, 768]);  add_32 = None
        convert_element_type_83: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        convert_element_type_84: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_45, torch.bfloat16);  view_45 = None
        convert_element_type_85: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        addmm_14: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_85, convert_element_type_84, convert_element_type_83);  convert_element_type_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_46: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        convert_element_type_89: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_46, torch.float32)
        pow_4: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_89, 3.0);  convert_element_type_89 = None
        mul_45: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_33: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_46, mul_45);  view_46 = mul_45 = None
        mul_46: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_3: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_34: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_47: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, add_34);  mul_44 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_47: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_47, [-1, 3072]);  mul_47 = None
        convert_element_type_90: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        convert_element_type_91: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_47, torch.bfloat16);  view_47 = None
        convert_element_type_92: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        addmm_15: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_92, convert_element_type_91, convert_element_type_90);  convert_element_type_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_48: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [8, 1024, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_16: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_16: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_16, torch.bfloat16);  inductor_random_default_16 = None
        gt_8: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_16, 0.1);  convert_element_type_default_16 = None
        mul_48: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_8, view_48);  view_48 = None
        mul_49: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, 1.1111111111111112);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_35: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_30, mul_49);  add_30 = mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_8[0]
        getitem_45: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_36: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_10: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_35, getitem_45);  getitem_45 = None
        mul_50: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = None
        mul_51: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, primals_52)
        add_37: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, primals_53);  mul_51 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_49: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_37, [-1, 768]);  add_37 = None
        convert_element_type_96: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        convert_element_type_97: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_49, torch.bfloat16);  view_49 = None
        convert_element_type_98: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        addmm_16: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_98, convert_element_type_97, convert_element_type_96);  convert_element_type_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_50: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [8, 1024, 2304]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_50, 768, 2);  view_50 = None
        getitem_46: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_4[0]
        getitem_47: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_4[1]
        getitem_48: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_4[2];  split_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_51: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_47, [8, 1024, -1, 64]);  getitem_47 = None
        permute_16: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_52: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_48, [8, 1024, -1, 64]);  getitem_48 = None
        permute_17: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_53: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_46, [8, 1024, -1, 64]);  getitem_46 = None
        permute_18: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_2, True, 0.1, scale = 0.125)
        getitem_49: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_4[0]
        getitem_50: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_4[1]
        getitem_51: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_4[2]
        getitem_52: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_19, [8, 1024, -1]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [-1, 768]);  view_54 = None
        convert_element_type_102: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        convert_element_type_103: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        addmm_17: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_103, view_55, convert_element_type_102);  convert_element_type_103 = view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_56: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [8, 1024, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_9: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_15: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_15: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_15, torch.bfloat16);  inductor_random_default_15 = None
        gt_9: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_15, 0.1);  convert_element_type_default_15 = None
        mul_52: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_9, view_56);  view_56 = None
        mul_53: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, 1.1111111111111112);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_38: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, add_35);  mul_53 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_53: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_9[0]
        getitem_54: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_39: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_11: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_38, getitem_54);  getitem_54 = None
        mul_54: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = None
        mul_55: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, primals_58)
        add_40: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, primals_59);  mul_55 = primals_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_57: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_40, [-1, 768]);  add_40 = None
        convert_element_type_107: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        convert_element_type_108: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_57, torch.bfloat16);  view_57 = None
        convert_element_type_109: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        addmm_18: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_109, convert_element_type_108, convert_element_type_107);  convert_element_type_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_58: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        convert_element_type_113: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_58, torch.float32)
        pow_5: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_113, 3.0);  convert_element_type_113 = None
        mul_57: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_58, mul_57);  view_58 = mul_57 = None
        mul_58: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_42: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_59: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, add_42);  mul_56 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_59: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_59, [-1, 3072]);  mul_59 = None
        convert_element_type_114: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        convert_element_type_115: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_59, torch.bfloat16);  view_59 = None
        convert_element_type_116: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        addmm_19: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_116, convert_element_type_115, convert_element_type_114);  convert_element_type_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_60: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [8, 1024, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_10: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_14: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        convert_element_type_default_14: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_14, torch.bfloat16);  inductor_random_default_14 = None
        gt_10: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_14, 0.1);  convert_element_type_default_14 = None
        mul_60: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_10, view_60);  view_60 = None
        mul_61: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_43: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_38, mul_61);  add_38 = mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_55: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_10[0]
        getitem_56: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_44: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_12: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_43, getitem_56);  getitem_56 = None
        mul_62: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = None
        mul_63: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, primals_64)
        add_45: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_63, primals_65);  mul_63 = primals_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_61: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_45, [-1, 768]);  add_45 = None
        convert_element_type_120: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convert_element_type_121: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_61, torch.bfloat16);  view_61 = None
        convert_element_type_122: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        addmm_20: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_122, convert_element_type_121, convert_element_type_120);  convert_element_type_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_62: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [8, 1024, 2304]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_62, 768, 2);  view_62 = None
        getitem_57: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_5[0]
        getitem_58: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_5[1]
        getitem_59: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_5[2];  split_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_63: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_58, [8, 1024, -1, 64]);  getitem_58 = None
        permute_20: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_64: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_59, [8, 1024, -1, 64]);  getitem_59 = None
        permute_21: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_65: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_57, [8, 1024, -1, 64]);  getitem_57 = None
        permute_22: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_2, True, 0.1, scale = 0.125)
        getitem_60: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_5[0]
        getitem_61: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_5[1]
        getitem_62: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_5[2]
        getitem_63: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_23, [8, 1024, -1]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_66, [-1, 768]);  view_66 = None
        convert_element_type_126: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        convert_element_type_127: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        addmm_21: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_127, view_67, convert_element_type_126);  convert_element_type_127 = view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_68: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [8, 1024, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_11: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_13: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        convert_element_type_default_13: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_13, torch.bfloat16);  inductor_random_default_13 = None
        gt_11: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_13, 0.1);  convert_element_type_default_13 = None
        mul_64: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_11, view_68);  view_68 = None
        mul_65: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, 1.1111111111111112);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_46: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, add_43);  mul_65 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_64: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_11[0]
        getitem_65: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_47: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_13: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_46, getitem_65);  getitem_65 = None
        mul_66: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = None
        mul_67: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, primals_70)
        add_48: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, primals_71);  mul_67 = primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_69: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_48, [-1, 768]);  add_48 = None
        convert_element_type_131: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_132: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_69, torch.bfloat16);  view_69 = None
        convert_element_type_133: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        addmm_22: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_133, convert_element_type_132, convert_element_type_131);  convert_element_type_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_70: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        convert_element_type_137: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_70, torch.float32)
        pow_6: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_137, 3.0);  convert_element_type_137 = None
        mul_69: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_49: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_70, mul_69);  view_70 = mul_69 = None
        mul_70: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_5: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_50: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_71: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_50);  mul_68 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_71: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_71, [-1, 3072]);  mul_71 = None
        convert_element_type_138: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        convert_element_type_139: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_71, torch.bfloat16);  view_71 = None
        convert_element_type_140: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        addmm_23: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_140, convert_element_type_139, convert_element_type_138);  convert_element_type_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_72: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [8, 1024, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_12: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        convert_element_type_default_12: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_12, torch.bfloat16);  inductor_random_default_12 = None
        gt_12: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_12, 0.1);  convert_element_type_default_12 = None
        mul_72: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_12, view_72);  view_72 = None
        mul_73: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, 1.1111111111111112);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_51: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_46, mul_73);  add_46 = mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_66: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_12[0]
        getitem_67: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_52: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_14: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_51, getitem_67);  getitem_67 = None
        mul_74: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = None
        mul_75: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, primals_76)
        add_53: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_75, primals_77);  mul_75 = primals_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_73: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_53, [-1, 768]);  add_53 = None
        convert_element_type_144: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_145: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_73, torch.bfloat16);  view_73 = None
        convert_element_type_146: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        addmm_24: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_146, convert_element_type_145, convert_element_type_144);  convert_element_type_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_74: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [8, 1024, 2304]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_6 = torch.ops.aten.split.Tensor(view_74, 768, 2);  view_74 = None
        getitem_68: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_6[0]
        getitem_69: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_6[1]
        getitem_70: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_6[2];  split_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_75: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_69, [8, 1024, -1, 64]);  getitem_69 = None
        permute_24: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_76: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_70, [8, 1024, -1, 64]);  getitem_70 = None
        permute_25: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_77: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_68, [8, 1024, -1, 64]);  getitem_68 = None
        permute_26: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_26, permute_24, permute_25, expand_2, True, 0.1, scale = 0.125)
        getitem_71: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_6[0]
        getitem_72: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_6[1]
        getitem_73: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_6[2]
        getitem_74: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_78: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_27, [8, 1024, -1]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_79: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_78, [-1, 768]);  view_78 = None
        convert_element_type_150: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        convert_element_type_151: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        addmm_25: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_151, view_79, convert_element_type_150);  convert_element_type_151 = view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_80: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [8, 1024, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_13: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_11: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        convert_element_type_default_11: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.bfloat16);  inductor_random_default_11 = None
        gt_13: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_11, 0.1);  convert_element_type_default_11 = None
        mul_76: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_13, view_80);  view_80 = None
        mul_77: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, 1.1111111111111112);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_54: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_77, add_51);  mul_77 = add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
        getitem_75: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_13[0]
        getitem_76: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_55: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_15: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_54, getitem_76);  getitem_76 = None
        mul_78: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = None
        mul_79: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, primals_82)
        add_56: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, primals_83);  mul_79 = primals_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_81: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_56, [-1, 768]);  add_56 = None
        convert_element_type_155: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        convert_element_type_156: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_81, torch.bfloat16);  view_81 = None
        convert_element_type_157: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        addmm_26: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_157, convert_element_type_156, convert_element_type_155);  convert_element_type_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_82: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_80: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_82, 0.5)
        convert_element_type_161: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_82, torch.float32)
        pow_7: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_161, 3.0);  convert_element_type_161 = None
        mul_81: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_57: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_82, mul_81);  view_82 = mul_81 = None
        mul_82: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_6: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_82);  mul_82 = None
        add_58: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_83: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, add_58);  mul_80 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_83: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_83, [-1, 3072]);  mul_83 = None
        convert_element_type_162: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        convert_element_type_163: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_83, torch.bfloat16);  view_83 = None
        convert_element_type_164: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        addmm_27: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_164, convert_element_type_163, convert_element_type_162);  convert_element_type_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_84: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [8, 1024, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_10: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        convert_element_type_default_10: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.bfloat16);  inductor_random_default_10 = None
        gt_14: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_10, 0.1);  convert_element_type_default_10 = None
        mul_84: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_14, view_84);  view_84 = None
        mul_85: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, 1.1111111111111112);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_59: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_54, mul_85);  add_54 = mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_77: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_14[0]
        getitem_78: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_60: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_16: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_59, getitem_78);  getitem_78 = None
        mul_86: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = None
        mul_87: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, primals_88)
        add_61: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_87, primals_89);  mul_87 = primals_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_85: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_61, [-1, 768]);  add_61 = None
        convert_element_type_168: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        convert_element_type_169: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.bfloat16);  view_85 = None
        convert_element_type_170: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        addmm_28: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_170, convert_element_type_169, convert_element_type_168);  convert_element_type_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_86: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [8, 1024, 2304]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_7 = torch.ops.aten.split.Tensor(view_86, 768, 2);  view_86 = None
        getitem_79: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_7[0]
        getitem_80: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_7[1]
        getitem_81: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_7[2];  split_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_87: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_80, [8, 1024, -1, 64]);  getitem_80 = None
        permute_28: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_88: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_81, [8, 1024, -1, 64]);  getitem_81 = None
        permute_29: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_89: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_79, [8, 1024, -1, 64]);  getitem_79 = None
        permute_30: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_30, permute_28, permute_29, expand_2, True, 0.1, scale = 0.125)
        getitem_82: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_7[0]
        getitem_83: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_7[1]
        getitem_84: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_7[2]
        getitem_85: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_90: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_31, [8, 1024, -1]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_91: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_90, [-1, 768]);  view_90 = None
        convert_element_type_174: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        convert_element_type_175: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        addmm_29: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_175, view_91, convert_element_type_174);  convert_element_type_175 = view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_92: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [8, 1024, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_15: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_9: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        convert_element_type_default_9: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.bfloat16);  inductor_random_default_9 = None
        gt_15: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_9, 0.1);  convert_element_type_default_9 = None
        mul_88: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_15, view_92);  view_92 = None
        mul_89: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, 1.1111111111111112);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_62: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_89, add_59);  mul_89 = add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
        getitem_86: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_15[0]
        getitem_87: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_63: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_17: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_62, getitem_87);  getitem_87 = None
        mul_90: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_15);  sub_17 = None
        mul_91: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, primals_94)
        add_64: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, primals_95);  mul_91 = primals_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_93: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_64, [-1, 768]);  add_64 = None
        convert_element_type_179: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        convert_element_type_180: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_93, torch.bfloat16);  view_93 = None
        convert_element_type_181: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        addmm_30: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_181, convert_element_type_180, convert_element_type_179);  convert_element_type_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_94: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_94, 0.5)
        convert_element_type_185: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_94, torch.float32)
        pow_8: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_185, 3.0);  convert_element_type_185 = None
        mul_93: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_65: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_94, mul_93);  view_94 = mul_93 = None
        mul_94: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_7: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_66: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_95: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, add_66);  mul_92 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_95: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_95, [-1, 3072]);  mul_95 = None
        convert_element_type_186: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        convert_element_type_187: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_95, torch.bfloat16);  view_95 = None
        convert_element_type_188: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        addmm_31: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_188, convert_element_type_187, convert_element_type_186);  convert_element_type_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_96: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [8, 1024, 768]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_16: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_8: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        convert_element_type_default_8: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_8, torch.bfloat16);  inductor_random_default_8 = None
        gt_16: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_8, 0.1);  convert_element_type_default_8 = None
        mul_96: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_16, view_96);  view_96 = None
        mul_97: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_67: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_62, mul_97);  add_62 = mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_88: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_16[0]
        getitem_89: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_68: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_18: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_67, getitem_89);  getitem_89 = None
        mul_98: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_16);  sub_18 = None
        mul_99: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, primals_100)
        add_69: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, primals_101);  mul_99 = primals_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_97: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_69, [-1, 768]);  add_69 = None
        convert_element_type_192: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        convert_element_type_193: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_97, torch.bfloat16);  view_97 = None
        convert_element_type_194: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        addmm_32: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_194, convert_element_type_193, convert_element_type_192);  convert_element_type_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_98: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [8, 1024, 2304]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_8 = torch.ops.aten.split.Tensor(view_98, 768, 2);  view_98 = None
        getitem_90: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_8[0]
        getitem_91: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_8[1]
        getitem_92: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_8[2];  split_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_99: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_91, [8, 1024, -1, 64]);  getitem_91 = None
        permute_32: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_100: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_92, [8, 1024, -1, 64]);  getitem_92 = None
        permute_33: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_101: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_90, [8, 1024, -1, 64]);  getitem_90 = None
        permute_34: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, permute_32, permute_33, expand_2, True, 0.1, scale = 0.125)
        getitem_93: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_8[0]
        getitem_94: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_8[1]
        getitem_95: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_8[2]
        getitem_96: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_102: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_35, [8, 1024, -1]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_103: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_102, [-1, 768]);  view_102 = None
        convert_element_type_198: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_199: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        addmm_33: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_199, view_103, convert_element_type_198);  convert_element_type_199 = view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_104: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [8, 1024, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_17: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_7: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        convert_element_type_default_7: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.bfloat16);  inductor_random_default_7 = None
        gt_17: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_7, 0.1);  convert_element_type_default_7 = None
        mul_100: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_17, view_104);  view_104 = None
        mul_101: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, 1.1111111111111112);  mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_70: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, add_67);  mul_101 = add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_97: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_17[0]
        getitem_98: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_71: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_19: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_70, getitem_98);  getitem_98 = None
        mul_102: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_17);  sub_19 = None
        mul_103: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, primals_106)
        add_72: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_103, primals_107);  mul_103 = primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_105: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_72, [-1, 768]);  add_72 = None
        convert_element_type_203: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        convert_element_type_204: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_105, torch.bfloat16);  view_105 = None
        convert_element_type_205: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        addmm_34: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_205, convert_element_type_204, convert_element_type_203);  convert_element_type_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_106: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_104: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        convert_element_type_209: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_106, torch.float32)
        pow_9: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_209, 3.0);  convert_element_type_209 = None
        mul_105: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_73: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_106, mul_105);  view_106 = mul_105 = None
        mul_106: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_8: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_106);  mul_106 = None
        add_74: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_107: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, add_74);  mul_104 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_107: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_107, [-1, 3072]);  mul_107 = None
        convert_element_type_210: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        convert_element_type_211: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.bfloat16);  view_107 = None
        convert_element_type_212: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        addmm_35: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_212, convert_element_type_211, convert_element_type_210);  convert_element_type_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_108: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [8, 1024, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_6: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        convert_element_type_default_6: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.bfloat16);  inductor_random_default_6 = None
        gt_18: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_6, 0.1);  convert_element_type_default_6 = None
        mul_108: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_18, view_108);  view_108 = None
        mul_109: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 1.1111111111111112);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_75: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_70, mul_109);  add_70 = mul_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_99: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_18[0]
        getitem_100: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_76: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_20: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_75, getitem_100);  getitem_100 = None
        mul_110: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_18);  sub_20 = None
        mul_111: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, primals_112)
        add_77: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, primals_113);  mul_111 = primals_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_109: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_77, [-1, 768]);  add_77 = None
        convert_element_type_216: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convert_element_type_217: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_109, torch.bfloat16);  view_109 = None
        convert_element_type_218: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        addmm_36: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_218, convert_element_type_217, convert_element_type_216);  convert_element_type_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_110: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [8, 1024, 2304]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_9 = torch.ops.aten.split.Tensor(view_110, 768, 2);  view_110 = None
        getitem_101: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_9[0]
        getitem_102: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_9[1]
        getitem_103: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_9[2];  split_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_111: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_102, [8, 1024, -1, 64]);  getitem_102 = None
        permute_36: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_112: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_103, [8, 1024, -1, 64]);  getitem_103 = None
        permute_37: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_113: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_101, [8, 1024, -1, 64]);  getitem_101 = None
        permute_38: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_38, permute_36, permute_37, expand_2, True, 0.1, scale = 0.125)
        getitem_104: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_9[0]
        getitem_105: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_9[1]
        getitem_106: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_9[2]
        getitem_107: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_114: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [8, 1024, -1]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_115: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [-1, 768]);  view_114 = None
        convert_element_type_222: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        convert_element_type_223: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        addmm_37: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_223, view_115, convert_element_type_222);  convert_element_type_223 = view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_116: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [8, 1024, 768]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_19: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_5: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        convert_element_type_default_5: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.bfloat16);  inductor_random_default_5 = None
        gt_19: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_5, 0.1);  convert_element_type_default_5 = None
        mul_112: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_19, view_116);  view_116 = None
        mul_113: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, 1.1111111111111112);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_78: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, add_75);  mul_113 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_108: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_19[0]
        getitem_109: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_79: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_21: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_78, getitem_109);  getitem_109 = None
        mul_114: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_19);  sub_21 = None
        mul_115: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, primals_118)
        add_80: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, primals_119);  mul_115 = primals_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_117: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_80, [-1, 768]);  add_80 = None
        convert_element_type_227: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_228: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_117, torch.bfloat16);  view_117 = None
        convert_element_type_229: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        addmm_38: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_229, convert_element_type_228, convert_element_type_227);  convert_element_type_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_118: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        convert_element_type_233: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_118, torch.float32)
        pow_10: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 3.0);  convert_element_type_233 = None
        mul_117: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_81: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_118, mul_117);  view_118 = mul_117 = None
        mul_118: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_9: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_82: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_119: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, add_82);  mul_116 = add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_119: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_119, [-1, 3072]);  mul_119 = None
        convert_element_type_234: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        convert_element_type_235: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_119, torch.bfloat16);  view_119 = None
        convert_element_type_236: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        addmm_39: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_236, convert_element_type_235, convert_element_type_234);  convert_element_type_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_120: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [8, 1024, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_4: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        convert_element_type_default_4: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_4, torch.bfloat16);  inductor_random_default_4 = None
        gt_20: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_4, 0.1);  convert_element_type_default_4 = None
        mul_120: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_20, view_120);  view_120 = None
        mul_121: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_83: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_78, mul_121);  add_78 = mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_110: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_20[0]
        getitem_111: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_84: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_22: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_83, getitem_111);  getitem_111 = None
        mul_122: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_20);  sub_22 = None
        mul_123: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, primals_124)
        add_85: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, primals_125);  mul_123 = primals_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_121: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_85, [-1, 768]);  add_85 = None
        convert_element_type_240: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        convert_element_type_241: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_121, torch.bfloat16);  view_121 = None
        convert_element_type_242: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        addmm_40: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_242, convert_element_type_241, convert_element_type_240);  convert_element_type_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_122: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [8, 1024, 2304]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_10 = torch.ops.aten.split.Tensor(view_122, 768, 2);  view_122 = None
        getitem_112: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_10[0]
        getitem_113: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_10[1]
        getitem_114: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_10[2];  split_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_123: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_113, [8, 1024, -1, 64]);  getitem_113 = None
        permute_40: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_124: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_114, [8, 1024, -1, 64]);  getitem_114 = None
        permute_41: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_125: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_112, [8, 1024, -1, 64]);  getitem_112 = None
        permute_42: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_40, permute_41, expand_2, True, 0.1, scale = 0.125)
        getitem_115: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_10[0]
        getitem_116: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_10[1]
        getitem_117: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_10[2]
        getitem_118: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_126: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_43, [8, 1024, -1]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_127: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_126, [-1, 768]);  view_126 = None
        convert_element_type_246: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        convert_element_type_247: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        addmm_41: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_247, view_127, convert_element_type_246);  convert_element_type_247 = view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_128: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [8, 1024, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_21: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_3: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        convert_element_type_default_3: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.bfloat16);  inductor_random_default_3 = None
        gt_21: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_3, 0.1);  convert_element_type_default_3 = None
        mul_124: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_21, view_128);  view_128 = None
        mul_125: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, 1.1111111111111112);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_86: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_125, add_83);  mul_125 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
        getitem_119: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_21[0]
        getitem_120: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_87: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_23: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_86, getitem_120);  getitem_120 = None
        mul_126: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_21);  sub_23 = None
        mul_127: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, primals_130)
        add_88: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, primals_131);  mul_127 = primals_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_129: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_88, [-1, 768]);  add_88 = None
        convert_element_type_251: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        convert_element_type_252: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.bfloat16);  view_129 = None
        convert_element_type_253: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        addmm_42: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_253, convert_element_type_252, convert_element_type_251);  convert_element_type_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_130: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_128: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_130, 0.5)
        convert_element_type_257: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_130, torch.float32)
        pow_11: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_257, 3.0);  convert_element_type_257 = None
        mul_129: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_89: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_130, mul_129);  view_130 = mul_129 = None
        mul_130: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_10: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_130);  mul_130 = None
        add_90: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_131: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, add_90);  mul_128 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_131: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_131, [-1, 3072]);  mul_131 = None
        convert_element_type_258: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        convert_element_type_259: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_131, torch.bfloat16);  view_131 = None
        convert_element_type_260: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        addmm_43: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_260, convert_element_type_259, convert_element_type_258);  convert_element_type_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_132: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [8, 1024, 768]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_22: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_2: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        convert_element_type_default_2: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.bfloat16);  inductor_random_default_2 = None
        gt_22: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_132: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_22, view_132);  view_132 = None
        mul_133: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, 1.1111111111111112);  mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_91: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_86, mul_133);  add_86 = mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_121: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_22[0]
        getitem_122: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_92: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_24: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_91, getitem_122);  getitem_122 = None
        mul_134: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_22);  sub_24 = None
        mul_135: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, primals_136)
        add_93: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_135, primals_137);  mul_135 = primals_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_133: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_93, [-1, 768]);  add_93 = None
        convert_element_type_264: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        convert_element_type_265: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_133, torch.bfloat16);  view_133 = None
        convert_element_type_266: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        addmm_44: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_266, convert_element_type_265, convert_element_type_264);  convert_element_type_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_134: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [8, 1024, 2304]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_11 = torch.ops.aten.split.Tensor(view_134, 768, 2);  view_134 = None
        getitem_123: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_11[0]
        getitem_124: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_11[1]
        getitem_125: "bf16[8, 1024, 768][2359296, 2304, 1]cuda:0" = split_11[2];  split_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_135: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_124, [8, 1024, -1, 64]);  getitem_124 = None
        permute_44: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_136: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_125, [8, 1024, -1, 64]);  getitem_125 = None
        permute_45: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_137: "bf16[8, 1024, 12, 64][2359296, 2304, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_123, [8, 1024, -1, 64]);  getitem_123 = None
        permute_46: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_46, permute_44, permute_45, expand_2, True, 0.1, scale = 0.125);  expand_2 = None
        getitem_126: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_11[0]
        getitem_127: "f32[8, 12, 1024][12288, 1024, 1]cuda:0" = _scaled_dot_product_efficient_attention_11[1]
        getitem_128: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_11[2]
        getitem_129: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_138: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_47, [8, 1024, -1]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_139: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_138, [-1, 768]);  view_138 = None
        convert_element_type_270: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        convert_element_type_271: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        addmm_45: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_271, view_139, convert_element_type_270);  convert_element_type_271 = view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_140: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [8, 1024, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_23: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_1: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        convert_element_type_default_1: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt_23: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_136: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_23, view_140);  view_140 = None
        mul_137: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, 1.1111111111111112);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_94: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, add_91);  mul_137 = add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
        getitem_130: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_23[0]
        getitem_131: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_95: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        sub_25: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_94, getitem_131);  getitem_131 = None
        mul_138: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_23);  sub_25 = None
        mul_139: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, primals_142)
        add_96: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, primals_143);  mul_139 = primals_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_141: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_96, [-1, 768]);  add_96 = None
        convert_element_type_275: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_145, torch.bfloat16);  primals_145 = None
        convert_element_type_276: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_141, torch.bfloat16);  view_141 = None
        convert_element_type_277: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        addmm_46: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_277, convert_element_type_276, convert_element_type_275);  convert_element_type_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_142: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        convert_element_type_281: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_142, torch.float32)
        pow_12: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_281, 3.0);  convert_element_type_281 = None
        mul_141: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_97: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_142, mul_141);  view_142 = mul_141 = None
        mul_142: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_11: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_98: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_143: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, add_98);  mul_140 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_143: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_143, [-1, 3072]);  mul_143 = None
        convert_element_type_282: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        convert_element_type_283: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_143, torch.bfloat16);  view_143 = None
        convert_element_type_284: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        addmm_47: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_284, convert_element_type_283, convert_element_type_282);  convert_element_type_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_144: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [8, 1024, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        convert_element_type_default: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_24: "b8[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_144: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_24, view_144);  view_144 = None
        mul_145: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_99: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_94, mul_145);  add_94 = mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_132: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_24[0]
        getitem_133: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_100: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_26: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_99, getitem_133);  add_99 = getitem_133 = None
        mul_146: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_24);  sub_26 = None
        mul_147: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, primals_148)
        add_101: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_147, primals_149);  mul_147 = primals_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:920 in forward, code: logits = self.score(hidden_states)
        convert_element_type_288: "bf16[2, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convert_element_type_289: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None
        permute_48: "bf16[768, 2][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_288, [1, 0]);  convert_element_type_288 = None
        view_146: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_289, [8192, 768]);  convert_element_type_289 = None
        mm: "bf16[8192, 2][2, 1]cuda:0" = torch.ops.aten.mm.default(view_146, permute_48)
        view_147: "bf16[8, 1024, 2][2048, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [8, 1024, 2]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:933 in forward, code: non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
        ne_1: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, 0)
        convert_element_type_292: "i32[8, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(ne_1, torch.int32);  ne_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:934 in forward, code: token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
        iota_5: "i32[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:935 in forward, code: last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        mul_148: "i32[8, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(iota_5, convert_element_type_292);  iota_5 = convert_element_type_292 = None
        argmax: "i64[8][1]cuda:0" = torch.ops.aten.argmax.default(mul_148, -1);  mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:943 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        index_2: "bf16[8, 2][2, 1]cuda:0" = torch.ops.aten.index.Tensor(view_147, [iota_1, argmax]);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:963 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        convert_element_type_293: "f32[8, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(index_2, torch.float32)
        amax: "f32[8, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_293, [1], True)
        sub_27: "f32[8, 2][2, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_293, amax);  convert_element_type_293 = amax = None
        exp: "f32[8, 2][2, 1]cuda:0" = torch.ops.aten.exp.default(sub_27)
        sum_1: "f32[8, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_28: "f32[8, 2][2, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_27, log);  sub_27 = log = None
        convert_element_type_294: "bf16[8, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_28, torch.bfloat16);  sub_28 = None
        convert_element_type_295: "f32[8, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_294, torch.float32);  convert_element_type_294 = None
        ne_2: "b8[8][1]cuda:0" = torch.ops.aten.ne.Scalar(primals_151, -100)
        full_default_25: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[8][1]cuda:0" = torch.ops.aten.where.self(ne_2, primals_151, full_default_25)
        unsqueeze_10: "i64[8, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather: "f32[8, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_295, 1, unsqueeze_10);  convert_element_type_295 = unsqueeze_10 = None
        squeeze: "f32[8][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_26: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f32[8][1]cuda:0" = torch.ops.aten.where.self(ne_2, neg, full_default_26);  neg = full_default_26 = None
        sum_2: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_296: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_13);  where_13 = None
        div: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_296);  sum_3 = None
        unsqueeze_11: "i64[8, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_151, 1);  primals_151 = None
        ne_5: "b8[8, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_11, -100)
        where_14: "i64[8, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_5, unsqueeze_11, full_default_25);  unsqueeze_11 = full_default_25 = None

        # No stacktrace found for following nodes
        iota_default: "i64[2][1]cuda:0" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 2][2, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 2]);  iota_default = None
        expand_default: "i64[8, 2][1, 0]cuda:0" = torch.ops.aten.expand.default(where_14, [8, 2]);  where_14 = None
        eq_tensor: "b8[8, 2][2, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:920 in forward, code: logits = self.score(hidden_states)
        permute_51: "bf16[2, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        div_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_53: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_282, [1, 0]);  convert_element_type_282 = None
        permute_54: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_283, [1, 0]);  convert_element_type_283 = None
        permute_55: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_275, [1, 0]);  convert_element_type_275 = None
        permute_56: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_276, [1, 0]);  convert_element_type_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_3: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_57: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        permute_63: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        permute_64: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_265, [1, 0]);  convert_element_type_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_65: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_258, [1, 0]);  convert_element_type_258 = None
        permute_66: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        permute_67: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_251, [1, 0]);  convert_element_type_251 = None
        permute_68: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_252, [1, 0]);  convert_element_type_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_5: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_69: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_246, [1, 0]);  convert_element_type_246 = None
        permute_75: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        permute_76: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_241, [1, 0]);  convert_element_type_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_77: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_234, [1, 0]);  convert_element_type_234 = None
        permute_78: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_235, [1, 0]);  convert_element_type_235 = None
        permute_79: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_227, [1, 0]);  convert_element_type_227 = None
        permute_80: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_228, [1, 0]);  convert_element_type_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_81: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_222, [1, 0]);  convert_element_type_222 = None
        permute_87: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_216, [1, 0]);  convert_element_type_216 = None
        permute_88: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_217, [1, 0]);  convert_element_type_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_8: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_89: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_210, [1, 0]);  convert_element_type_210 = None
        permute_90: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_211, [1, 0]);  convert_element_type_211 = None
        permute_91: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_203, [1, 0]);  convert_element_type_203 = None
        permute_92: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_204, [1, 0]);  convert_element_type_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_93: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_198, [1, 0]);  convert_element_type_198 = None
        permute_99: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        permute_100: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_193, [1, 0]);  convert_element_type_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_101: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_186, [1, 0]);  convert_element_type_186 = None
        permute_102: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_187, [1, 0]);  convert_element_type_187 = None
        permute_103: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_179, [1, 0]);  convert_element_type_179 = None
        permute_104: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_180, [1, 0]);  convert_element_type_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_105: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_174, [1, 0]);  convert_element_type_174 = None
        permute_111: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_168, [1, 0]);  convert_element_type_168 = None
        permute_112: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_169, [1, 0]);  convert_element_type_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_113: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        permute_114: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_163, [1, 0]);  convert_element_type_163 = None
        permute_115: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_155, [1, 0]);  convert_element_type_155 = None
        permute_116: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_156, [1, 0]);  convert_element_type_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_117: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_150, [1, 0]);  convert_element_type_150 = None
        permute_123: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        permute_124: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_145, [1, 0]);  convert_element_type_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_14: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_125: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_138, [1, 0]);  convert_element_type_138 = None
        permute_126: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        permute_127: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_131, [1, 0]);  convert_element_type_131 = None
        permute_128: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_132, [1, 0]);  convert_element_type_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_129: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_126, [1, 0]);  convert_element_type_126 = None
        permute_135: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_120, [1, 0]);  convert_element_type_120 = None
        permute_136: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_16: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_137: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_114, [1, 0]);  convert_element_type_114 = None
        permute_138: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_115, [1, 0]);  convert_element_type_115 = None
        permute_139: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_107, [1, 0]);  convert_element_type_107 = None
        permute_140: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_141: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_102, [1, 0]);  convert_element_type_102 = None
        permute_147: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_96, [1, 0]);  convert_element_type_96 = None
        permute_148: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_97, [1, 0]);  convert_element_type_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_18: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_149: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_90, [1, 0]);  convert_element_type_90 = None
        permute_150: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_91, [1, 0]);  convert_element_type_91 = None
        permute_151: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_83, [1, 0]);  convert_element_type_83 = None
        permute_152: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_19: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_153: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_78, [1, 0]);  convert_element_type_78 = None
        permute_159: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_72, [1, 0]);  convert_element_type_72 = None
        permute_160: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_73, [1, 0]);  convert_element_type_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_161: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        permute_162: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_67, [1, 0]);  convert_element_type_67 = None
        permute_163: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_59, [1, 0]);  convert_element_type_59 = None
        permute_164: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_60, [1, 0]);  convert_element_type_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_21: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_165: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_54, [1, 0]);  convert_element_type_54 = None
        permute_171: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_48, [1, 0]);  convert_element_type_48 = None
        permute_172: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_49, [1, 0]);  convert_element_type_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_173: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_42, [1, 0]);  convert_element_type_42 = None
        permute_174: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_43, [1, 0]);  convert_element_type_43 = None
        permute_175: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_35, [1, 0]);  convert_element_type_35 = None
        permute_176: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_36, [1, 0]);  convert_element_type_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_177: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        permute_183: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        permute_184: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0]);  convert_element_type_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_185: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_18, [1, 0]);  convert_element_type_18 = None
        permute_186: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        permute_187: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        permute_188: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_25: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_189: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        permute_195: "bf16[2304, 768][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        permute_196: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        return (div, index_2, primals_1, primals_4, primals_10, primals_16, primals_22, primals_28, primals_34, primals_40, primals_46, primals_52, primals_58, primals_64, primals_70, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, embedding, unsqueeze, embedding_1, iota_1, gt, getitem_1, rsqrt, permute, permute_1, permute_2, where, getitem_5, getitem_6, getitem_7, getitem_8, gt_1, mul_6, addmm_2, gt_2, mul_14, permute_4, permute_5, permute_6, getitem_16, getitem_17, getitem_18, getitem_19, gt_3, mul_18, addmm_6, gt_4, mul_26, permute_8, permute_9, permute_10, getitem_27, getitem_28, getitem_29, getitem_30, gt_5, mul_30, addmm_10, gt_6, mul_38, permute_12, permute_13, permute_14, getitem_38, getitem_39, getitem_40, getitem_41, gt_7, mul_42, addmm_14, gt_8, mul_50, permute_16, permute_17, permute_18, getitem_49, getitem_50, getitem_51, getitem_52, gt_9, mul_54, addmm_18, gt_10, mul_62, permute_20, permute_21, permute_22, getitem_60, getitem_61, getitem_62, getitem_63, gt_11, mul_66, addmm_22, gt_12, mul_74, permute_24, permute_25, permute_26, getitem_71, getitem_72, getitem_73, getitem_74, gt_13, mul_78, addmm_26, gt_14, mul_86, permute_28, permute_29, permute_30, getitem_82, getitem_83, getitem_84, getitem_85, gt_15, mul_90, addmm_30, gt_16, mul_98, permute_32, permute_33, permute_34, getitem_93, getitem_94, getitem_95, getitem_96, gt_17, mul_102, addmm_34, gt_18, mul_110, permute_36, permute_37, permute_38, getitem_104, getitem_105, getitem_106, getitem_107, gt_19, mul_114, addmm_38, gt_20, mul_122, permute_40, permute_41, permute_42, getitem_115, getitem_116, getitem_117, getitem_118, gt_21, mul_126, addmm_42, gt_22, mul_134, permute_44, permute_45, permute_46, getitem_126, getitem_127, getitem_128, getitem_129, gt_23, mul_138, addmm_46, gt_24, mul_146, view_146, argmax, index_2, convert_element_type_296, ne_5, eq_tensor, permute_51, div_2, permute_53, permute_54, permute_55, permute_56, div_3, permute_57, permute_63, permute_64, div_4, permute_65, permute_66, permute_67, permute_68, div_5, permute_69, permute_75, permute_76, div_6, permute_77, permute_78, permute_79, permute_80, div_7, permute_81, permute_87, permute_88, div_8, permute_89, permute_90, permute_91, permute_92, div_9, permute_93, permute_99, permute_100, div_10, permute_101, permute_102, permute_103, permute_104, div_11, permute_105, permute_111, permute_112, div_12, permute_113, permute_114, permute_115, permute_116, div_13, permute_117, permute_123, permute_124, div_14, permute_125, permute_126, permute_127, permute_128, div_15, permute_129, permute_135, permute_136, div_16, permute_137, permute_138, permute_139, permute_140, div_17, permute_141, permute_147, permute_148, div_18, permute_149, permute_150, permute_151, permute_152, div_19, permute_153, permute_159, permute_160, div_20, permute_161, permute_162, permute_163, permute_164, div_21, permute_165, permute_171, permute_172, div_22, permute_173, permute_174, permute_175, permute_176, div_23, permute_177, permute_183, permute_184, div_24, permute_185, permute_186, permute_187, permute_188, div_25, permute_189, permute_195, permute_196)
