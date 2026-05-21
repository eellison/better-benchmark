class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024]", primals_2: "f32[32128, 512]", primals_3: "f32[512]", primals_4: "f32[512, 512]", primals_5: "f32[512, 512]", primals_6: "f32[512, 512]", primals_7: "f32[32, 8]", primals_8: "f32[512, 512]", primals_9: "f32[512]", primals_10: "f32[2048, 512]", primals_11: "f32[512, 2048]", primals_12: "f32[512]", primals_13: "f32[512, 512]", primals_14: "f32[512, 512]", primals_15: "f32[512, 512]", primals_16: "f32[512, 512]", primals_17: "f32[512]", primals_18: "f32[2048, 512]", primals_19: "f32[512, 2048]", primals_20: "f32[512]", primals_21: "f32[512, 512]", primals_22: "f32[512, 512]", primals_23: "f32[512, 512]", primals_24: "f32[512, 512]", primals_25: "f32[512]", primals_26: "f32[2048, 512]", primals_27: "f32[512, 2048]", primals_28: "f32[512]", primals_29: "f32[512, 512]", primals_30: "f32[512, 512]", primals_31: "f32[512, 512]", primals_32: "f32[512, 512]", primals_33: "f32[512]", primals_34: "f32[2048, 512]", primals_35: "f32[512, 2048]", primals_36: "f32[512]", primals_37: "f32[512, 512]", primals_38: "f32[512, 512]", primals_39: "f32[512, 512]", primals_40: "f32[512, 512]", primals_41: "f32[512]", primals_42: "f32[2048, 512]", primals_43: "f32[512, 2048]", primals_44: "f32[512]", primals_45: "f32[512, 512]", primals_46: "f32[512, 512]", primals_47: "f32[512, 512]", primals_48: "f32[512, 512]", primals_49: "f32[512]", primals_50: "f32[2048, 512]", primals_51: "f32[512, 2048]", primals_52: "f32[512]", primals_53: "i64[8, 1024]", primals_54: "f32[512]", primals_55: "f32[512, 512]", primals_56: "f32[512, 512]", primals_57: "f32[512, 512]", primals_58: "f32[32, 8]", primals_59: "f32[512, 512]", primals_60: "f32[512]", primals_61: "f32[512, 512]", primals_62: "f32[512, 512]", primals_63: "f32[512, 512]", primals_64: "f32[512, 512]", primals_65: "f32[512]", primals_66: "f32[2048, 512]", primals_67: "f32[512, 2048]", primals_68: "f32[512]", primals_69: "f32[512, 512]", primals_70: "f32[512, 512]", primals_71: "f32[512, 512]", primals_72: "f32[512, 512]", primals_73: "f32[512]", primals_74: "f32[512, 512]", primals_75: "f32[512, 512]", primals_76: "f32[512, 512]", primals_77: "f32[512, 512]", primals_78: "f32[512]", primals_79: "f32[2048, 512]", primals_80: "f32[512, 2048]", primals_81: "f32[512]", primals_82: "f32[512, 512]", primals_83: "f32[512, 512]", primals_84: "f32[512, 512]", primals_85: "f32[512, 512]", primals_86: "f32[512]", primals_87: "f32[512, 512]", primals_88: "f32[512, 512]", primals_89: "f32[512, 512]", primals_90: "f32[512, 512]", primals_91: "f32[512]", primals_92: "f32[2048, 512]", primals_93: "f32[512, 2048]", primals_94: "f32[512]", primals_95: "f32[512, 512]", primals_96: "f32[512, 512]", primals_97: "f32[512, 512]", primals_98: "f32[512, 512]", primals_99: "f32[512]", primals_100: "f32[512, 512]", primals_101: "f32[512, 512]", primals_102: "f32[512, 512]", primals_103: "f32[512, 512]", primals_104: "f32[512]", primals_105: "f32[2048, 512]", primals_106: "f32[512, 2048]", primals_107: "f32[512]", primals_108: "f32[512, 512]", primals_109: "f32[512, 512]", primals_110: "f32[512, 512]", primals_111: "f32[512, 512]", primals_112: "f32[512]", primals_113: "f32[512, 512]", primals_114: "f32[512, 512]", primals_115: "f32[512, 512]", primals_116: "f32[512, 512]", primals_117: "f32[512]", primals_118: "f32[2048, 512]", primals_119: "f32[512, 2048]", primals_120: "f32[512]", primals_121: "f32[512, 512]", primals_122: "f32[512, 512]", primals_123: "f32[512, 512]", primals_124: "f32[512, 512]", primals_125: "f32[512]", primals_126: "f32[512, 512]", primals_127: "f32[512, 512]", primals_128: "f32[512, 512]", primals_129: "f32[512, 512]", primals_130: "f32[512]", primals_131: "f32[2048, 512]", primals_132: "f32[512, 2048]", primals_133: "f32[512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "f32[8, 1024, 512]" = torch.ops.aten.embedding.default(primals_2, primals_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024]" = torch.ops.aten.add.Tensor(iota_2, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 1024, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(ge, [8, -1, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[64]" = torch.ops.prims.inductor_seeds.default(64, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_63: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_63, 0.1);  inductor_random_default_63 = None
        mul: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt, embedding)
        mul_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_1, 2)
        mean: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_2: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_1, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_3, mul_2);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute: "f32[512, 512]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        view_1: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_3, [8192, 512]);  mul_3 = None
        mm: "f32[8192, 512]" = torch.ops.aten.mm.default(view_1, permute);  permute = None
        view_2: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm, [8, 1024, 512]);  mm = None
        view_3: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_2, [8, 1024, -1, 64]);  view_2 = None
        permute_1: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_2: "f32[512, 512]" = torch.ops.aten.permute.default(primals_5, [1, 0])
        mm_1: "f32[8192, 512]" = torch.ops.aten.mm.default(view_1, permute_2);  permute_2 = None
        view_5: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_1, [8, 1024, 512]);  mm_1 = None
        view_6: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_5, [8, 1024, -1, 64]);  view_5 = None
        permute_3: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_4: "f32[512, 512]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        mm_2: "f32[8192, 512]" = torch.ops.aten.mm.default(view_1, permute_4);  permute_4 = None
        view_8: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_2, [8, 1024, 512]);  mm_2 = None
        view_9: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_8, [8, 1024, -1, 64]);  view_8 = None
        permute_5: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_6: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_1: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_1, [8, 8, 1024, 64]);  permute_1 = None
        clone: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_10: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone, [64, 1024, 64]);  clone = None
        expand_2: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_6, [8, 8, 64, 1024]);  permute_6 = None
        clone_1: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_11: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_1, [64, 64, 1024]);  clone_1 = None
        bmm: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_10, view_11)
        view_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm, [8, 8, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:227 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        unsqueeze_3: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(iota_2, 1)
        add_3: "i64[1024, 1]" = torch.ops.aten.add.Tensor(unsqueeze_3, 0);  unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:228 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        unsqueeze_4: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:229 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(unsqueeze_4, add_3);  unsqueeze_4 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:200 in _relative_position_bucket, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        gt_1: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_1, torch.int64);  gt_1 = None
        mul_4: "i64[1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 16);  convert_element_type = None
        add_4: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(mul_4, 0);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:201 in _relative_position_bucket, code: relative_position = torch.abs(relative_position)
        abs_1: "i64[1024, 1024]" = torch.ops.aten.abs.default(sub)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(abs_1, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_1: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_1, 8);  convert_element_type_1 = None
        log: "f32[1024, 1024]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_5: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_2: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_5, torch.int64);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_5: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_2, 8);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_2: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_5, full_default_2);  add_5 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_1: "i64[1024, 1024]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_6: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(add_4, where_1);  add_4 = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_1: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(primals_7, add_6);  primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1])
        unsqueeze_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_5, where);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_8, [-1], True)
        sub_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_8, amax);  add_8 = None
        exp: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_62: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_2: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_62, 0.1);  inductor_random_default_62 = None
        mul_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_2, div_2);  div_2 = None
        mul_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_7, [8, 8, 1024, 1024]);  mul_7 = None
        view_16: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_3, [64, 1024, 1024]);  expand_3 = None
        expand_4: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_5, [8, 8, 1024, 64]);  permute_5 = None
        clone_2: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_17: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_2, [64, 1024, 64]);  clone_2 = None
        bmm_1: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_16, view_17)
        view_18: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_1, [8, 8, 1024, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_3: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_19: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_3, [8, 1024, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_9: "f32[512, 512]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        view_20: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_19, [8192, 512]);  view_19 = None
        mm_3: "f32[8192, 512]" = torch.ops.aten.mm.default(view_20, permute_9);  permute_9 = None
        view_21: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_3, [8, 1024, 512]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_61: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_3: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_61, 0.1);  inductor_random_default_61 = None
        mul_8: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_9: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None
        add_9: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_1, mul_9);  mul_1 = mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_2: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_9, 2)
        mean_1: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_10: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_11: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_9, mul_10);  mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_10: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        view_22: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_11, [8192, 512]);  mul_11 = None
        mm_4: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_22, permute_10);  permute_10 = None
        view_23: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_4, [8, 1024, 2048]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_23);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_60: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_4: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_60, 0.1);  inductor_random_default_60 = None
        mul_12: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_4, relu)
        mul_13: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_12, 1.1111111111111112);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_11: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        view_24: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_13, [8192, 2048]);  mul_13 = None
        mm_5: "f32[8192, 512]" = torch.ops.aten.mm.default(view_24, permute_11);  permute_11 = None
        view_25: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_5, [8, 1024, 512]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_59: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_5: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_59, 0.1);  inductor_random_default_59 = None
        mul_14: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_5, view_25);  view_25 = None
        mul_15: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        add_11: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_9, mul_15);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_3: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
        mean_2: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_12: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_16: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_17: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_12, mul_16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_12: "f32[512, 512]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        view_26: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_17, [8192, 512]);  mul_17 = None
        mm_6: "f32[8192, 512]" = torch.ops.aten.mm.default(view_26, permute_12);  permute_12 = None
        view_27: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_6, [8, 1024, 512]);  mm_6 = None
        view_28: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_27, [8, 1024, -1, 64]);  view_27 = None
        permute_13: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_14: "f32[512, 512]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        mm_7: "f32[8192, 512]" = torch.ops.aten.mm.default(view_26, permute_14);  permute_14 = None
        view_30: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_7, [8, 1024, 512]);  mm_7 = None
        view_31: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_30, [8, 1024, -1, 64]);  view_30 = None
        permute_15: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_16: "f32[512, 512]" = torch.ops.aten.permute.default(primals_15, [1, 0])
        mm_8: "f32[8192, 512]" = torch.ops.aten.mm.default(view_26, permute_16);  permute_16 = None
        view_33: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_8, [8, 1024, 512]);  mm_8 = None
        view_34: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_33, [8, 1024, -1, 64]);  view_33 = None
        permute_17: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_18: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_5: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_13, [8, 8, 1024, 64]);  permute_13 = None
        clone_4: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_35: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_4, [64, 1024, 64]);  clone_4 = None
        expand_6: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_18, [8, 8, 64, 1024]);  permute_18 = None
        clone_5: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_36: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_5, [64, 64, 1024]);  clone_5 = None
        bmm_2: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_35, view_36)
        view_37: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_2, [8, 8, 1024, 1024]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_37, add_7);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_1: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_13, [-1], True)
        sub_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_13, amax_1);  add_13 = amax_1 = None
        exp_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_2: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_58: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_6: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_58, 0.1);  inductor_random_default_58 = None
        mul_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_6, div_3)
        mul_19: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_18, 1.1111111111111112);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_19, [8, 8, 1024, 1024]);  mul_19 = None
        view_41: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_7, [64, 1024, 1024]);  expand_7 = None
        expand_8: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_17, [8, 8, 1024, 64]);  permute_17 = None
        clone_6: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_42: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_6, [64, 1024, 64]);  clone_6 = None
        bmm_3: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_41, view_42)
        view_43: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_3, [8, 8, 1024, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        clone_7: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_44: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_7, [8, 1024, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_20: "f32[512, 512]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        view_45: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_44, [8192, 512]);  view_44 = None
        mm_9: "f32[8192, 512]" = torch.ops.aten.mm.default(view_45, permute_20);  permute_20 = None
        view_46: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_9, [8, 1024, 512]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_57: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_7: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_57, 0.1);  inductor_random_default_57 = None
        mul_20: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_7, view_46);  view_46 = None
        mul_21: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None
        add_14: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_11, mul_21);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_4: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_14, 2)
        mean_3: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_15: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_22: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_14, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_23: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_17, mul_22);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_21: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_18, [1, 0])
        view_47: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_23, [8192, 512]);  mul_23 = None
        mm_10: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_47, permute_21);  permute_21 = None
        view_48: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_10, [8, 1024, 2048]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_1: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_48);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_56: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_8: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_56, 0.1);  inductor_random_default_56 = None
        mul_24: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_8, relu_1)
        mul_25: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_24, 1.1111111111111112);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_22: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_19, [1, 0])
        view_49: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_25, [8192, 2048]);  mul_25 = None
        mm_11: "f32[8192, 512]" = torch.ops.aten.mm.default(view_49, permute_22);  permute_22 = None
        view_50: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_11, [8, 1024, 512]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_55: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_9: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_55, 0.1);  inductor_random_default_55 = None
        mul_26: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_9, view_50);  view_50 = None
        mul_27: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_26, 1.1111111111111112);  mul_26 = None
        add_16: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_14, mul_27);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_5: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
        mean_4: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_28: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_29: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_20, mul_28);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_23: "f32[512, 512]" = torch.ops.aten.permute.default(primals_21, [1, 0])
        view_51: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_29, [8192, 512]);  mul_29 = None
        mm_12: "f32[8192, 512]" = torch.ops.aten.mm.default(view_51, permute_23);  permute_23 = None
        view_52: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_12, [8, 1024, 512]);  mm_12 = None
        view_53: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_52, [8, 1024, -1, 64]);  view_52 = None
        permute_24: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_25: "f32[512, 512]" = torch.ops.aten.permute.default(primals_22, [1, 0])
        mm_13: "f32[8192, 512]" = torch.ops.aten.mm.default(view_51, permute_25);  permute_25 = None
        view_55: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_13, [8, 1024, 512]);  mm_13 = None
        view_56: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_55, [8, 1024, -1, 64]);  view_55 = None
        permute_26: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_27: "f32[512, 512]" = torch.ops.aten.permute.default(primals_23, [1, 0])
        mm_14: "f32[8192, 512]" = torch.ops.aten.mm.default(view_51, permute_27);  permute_27 = None
        view_58: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_14, [8, 1024, 512]);  mm_14 = None
        view_59: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_58, [8, 1024, -1, 64]);  view_58 = None
        permute_28: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_29: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        expand_9: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_24, [8, 8, 1024, 64]);  permute_24 = None
        clone_8: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_60: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_8, [64, 1024, 64]);  clone_8 = None
        expand_10: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_29, [8, 8, 64, 1024]);  permute_29 = None
        clone_9: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_61: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_9, [64, 64, 1024]);  clone_9 = None
        bmm_4: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_60, view_61)
        view_62: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_4, [8, 8, 1024, 1024]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_62, add_7);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_2: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_18, [-1], True)
        sub_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_18, amax_2);  add_18 = amax_2 = None
        exp_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_3: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_54: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_10: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_54, 0.1);  inductor_random_default_54 = None
        mul_30: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_10, div_4)
        mul_31: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_31, [8, 8, 1024, 1024]);  mul_31 = None
        view_66: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_11, [64, 1024, 1024]);  expand_11 = None
        expand_12: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_28, [8, 8, 1024, 64]);  permute_28 = None
        clone_10: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_67: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_10, [64, 1024, 64]);  clone_10 = None
        bmm_5: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_66, view_67)
        view_68: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_5, [8, 8, 1024, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        clone_11: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_69: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_11, [8, 1024, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_31: "f32[512, 512]" = torch.ops.aten.permute.default(primals_24, [1, 0])
        view_70: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_69, [8192, 512]);  view_69 = None
        mm_15: "f32[8192, 512]" = torch.ops.aten.mm.default(view_70, permute_31);  permute_31 = None
        view_71: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_15, [8, 1024, 512]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_53: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_11: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_53, 0.1);  inductor_random_default_53 = None
        mul_32: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_11, view_71);  view_71 = None
        mul_33: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_32, 1.1111111111111112);  mul_32 = None
        add_19: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_16, mul_33);  mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_6: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_19, 2)
        mean_5: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_20: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_34: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_19, rsqrt_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_35: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_25, mul_34);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_32: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_26, [1, 0])
        view_72: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_35, [8192, 512]);  mul_35 = None
        mm_16: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_72, permute_32);  permute_32 = None
        view_73: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_16, [8, 1024, 2048]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_2: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_73);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_52: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_12: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_52, 0.1);  inductor_random_default_52 = None
        mul_36: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_12, relu_2)
        mul_37: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_33: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_27, [1, 0])
        view_74: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_37, [8192, 2048]);  mul_37 = None
        mm_17: "f32[8192, 512]" = torch.ops.aten.mm.default(view_74, permute_33);  permute_33 = None
        view_75: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_17, [8, 1024, 512]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_51: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_13: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_51, 0.1);  inductor_random_default_51 = None
        mul_38: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_13, view_75);  view_75 = None
        mul_39: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_38, 1.1111111111111112);  mul_38 = None
        add_21: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_19, mul_39);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_7: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 2)
        mean_6: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_22: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_40: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_41: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_28, mul_40);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_34: "f32[512, 512]" = torch.ops.aten.permute.default(primals_29, [1, 0])
        view_76: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_41, [8192, 512]);  mul_41 = None
        mm_18: "f32[8192, 512]" = torch.ops.aten.mm.default(view_76, permute_34);  permute_34 = None
        view_77: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_18, [8, 1024, 512]);  mm_18 = None
        view_78: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_77, [8, 1024, -1, 64]);  view_77 = None
        permute_35: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_36: "f32[512, 512]" = torch.ops.aten.permute.default(primals_30, [1, 0])
        mm_19: "f32[8192, 512]" = torch.ops.aten.mm.default(view_76, permute_36);  permute_36 = None
        view_80: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_19, [8, 1024, 512]);  mm_19 = None
        view_81: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_80, [8, 1024, -1, 64]);  view_80 = None
        permute_37: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_38: "f32[512, 512]" = torch.ops.aten.permute.default(primals_31, [1, 0])
        mm_20: "f32[8192, 512]" = torch.ops.aten.mm.default(view_76, permute_38);  permute_38 = None
        view_83: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_20, [8, 1024, 512]);  mm_20 = None
        view_84: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_83, [8, 1024, -1, 64]);  view_83 = None
        permute_39: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_40: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        expand_13: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_35, [8, 8, 1024, 64]);  permute_35 = None
        clone_12: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_85: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_12, [64, 1024, 64]);  clone_12 = None
        expand_14: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_40, [8, 8, 64, 1024]);  permute_40 = None
        clone_13: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_86: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_13, [64, 64, 1024]);  clone_13 = None
        bmm_6: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_85, view_86)
        view_87: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_6, [8, 8, 1024, 1024]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_23: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_87, add_7);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_3: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_23, [-1], True)
        sub_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_23, amax_3);  add_23 = amax_3 = None
        exp_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_4: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_50: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_14: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_50, 0.1);  inductor_random_default_50 = None
        mul_42: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_14, div_5)
        mul_43: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_42, 1.1111111111111112);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_43, [8, 8, 1024, 1024]);  mul_43 = None
        view_91: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_15, [64, 1024, 1024]);  expand_15 = None
        expand_16: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_39, [8, 8, 1024, 64]);  permute_39 = None
        clone_14: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_92: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_14, [64, 1024, 64]);  clone_14 = None
        bmm_7: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_91, view_92)
        view_93: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_7, [8, 8, 1024, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_41: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        clone_15: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_94: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_15, [8, 1024, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_42: "f32[512, 512]" = torch.ops.aten.permute.default(primals_32, [1, 0])
        view_95: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_94, [8192, 512]);  view_94 = None
        mm_21: "f32[8192, 512]" = torch.ops.aten.mm.default(view_95, permute_42);  permute_42 = None
        view_96: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_21, [8, 1024, 512]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_49: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_15: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_49, 0.1);  inductor_random_default_49 = None
        mul_44: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_15, view_96);  view_96 = None
        mul_45: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_44, 1.1111111111111112);  mul_44 = None
        add_24: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_21, mul_45);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_8: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_24, 2)
        mean_7: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_25: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_46: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_47: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_33, mul_46);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_43: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_34, [1, 0])
        view_97: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_47, [8192, 512]);  mul_47 = None
        mm_22: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_97, permute_43);  permute_43 = None
        view_98: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_22, [8, 1024, 2048]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_3: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_98);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_48: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_16: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_48, 0.1);  inductor_random_default_48 = None
        mul_48: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_16, relu_3)
        mul_49: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_48, 1.1111111111111112);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_44: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_35, [1, 0])
        view_99: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_49, [8192, 2048]);  mul_49 = None
        mm_23: "f32[8192, 512]" = torch.ops.aten.mm.default(view_99, permute_44);  permute_44 = None
        view_100: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_23, [8, 1024, 512]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_47: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_17: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_47, 0.1);  inductor_random_default_47 = None
        mul_50: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_17, view_100);  view_100 = None
        mul_51: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_50, 1.1111111111111112);  mul_50 = None
        add_26: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_24, mul_51);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_9: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 2)
        mean_8: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_27: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_52: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_53: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_36, mul_52);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_45: "f32[512, 512]" = torch.ops.aten.permute.default(primals_37, [1, 0])
        view_101: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_53, [8192, 512]);  mul_53 = None
        mm_24: "f32[8192, 512]" = torch.ops.aten.mm.default(view_101, permute_45);  permute_45 = None
        view_102: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_24, [8, 1024, 512]);  mm_24 = None
        view_103: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_102, [8, 1024, -1, 64]);  view_102 = None
        permute_46: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_47: "f32[512, 512]" = torch.ops.aten.permute.default(primals_38, [1, 0])
        mm_25: "f32[8192, 512]" = torch.ops.aten.mm.default(view_101, permute_47);  permute_47 = None
        view_105: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_25, [8, 1024, 512]);  mm_25 = None
        view_106: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_105, [8, 1024, -1, 64]);  view_105 = None
        permute_48: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_49: "f32[512, 512]" = torch.ops.aten.permute.default(primals_39, [1, 0])
        mm_26: "f32[8192, 512]" = torch.ops.aten.mm.default(view_101, permute_49);  permute_49 = None
        view_108: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_26, [8, 1024, 512]);  mm_26 = None
        view_109: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_108, [8, 1024, -1, 64]);  view_108 = None
        permute_50: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_51: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_17: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_46, [8, 8, 1024, 64]);  permute_46 = None
        clone_16: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_110: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_16, [64, 1024, 64]);  clone_16 = None
        expand_18: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_51, [8, 8, 64, 1024]);  permute_51 = None
        clone_17: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_111: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_17, [64, 64, 1024]);  clone_17 = None
        bmm_8: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_110, view_111)
        view_112: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_8, [8, 8, 1024, 1024]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_28: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_112, add_7);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_4: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_28, [-1], True)
        sub_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_28, amax_4);  add_28 = amax_4 = None
        exp_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_5: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_46: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_18: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_46, 0.1);  inductor_random_default_46 = None
        mul_54: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_18, div_6)
        mul_55: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_54, 1.1111111111111112);  mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_19: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_55, [8, 8, 1024, 1024]);  mul_55 = None
        view_116: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_19, [64, 1024, 1024]);  expand_19 = None
        expand_20: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_50, [8, 8, 1024, 64]);  permute_50 = None
        clone_18: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_117: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_18, [64, 1024, 64]);  clone_18 = None
        bmm_9: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_116, view_117)
        view_118: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_9, [8, 8, 1024, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_52: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        clone_19: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_119: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_19, [8, 1024, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_53: "f32[512, 512]" = torch.ops.aten.permute.default(primals_40, [1, 0])
        view_120: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_119, [8192, 512]);  view_119 = None
        mm_27: "f32[8192, 512]" = torch.ops.aten.mm.default(view_120, permute_53);  permute_53 = None
        view_121: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_27, [8, 1024, 512]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_45: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_19: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_45, 0.1);  inductor_random_default_45 = None
        mul_56: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_19, view_121);  view_121 = None
        mul_57: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_56, 1.1111111111111112);  mul_56 = None
        add_29: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_26, mul_57);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_10: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_29, 2)
        mean_9: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_30: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_58: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_29, rsqrt_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_59: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_41, mul_58);  mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_54: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_42, [1, 0])
        view_122: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_59, [8192, 512]);  mul_59 = None
        mm_28: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_122, permute_54);  permute_54 = None
        view_123: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_28, [8, 1024, 2048]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_4: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_123);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_44: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_20: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_44, 0.1);  inductor_random_default_44 = None
        mul_60: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_20, relu_4)
        mul_61: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_55: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_43, [1, 0])
        view_124: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_61, [8192, 2048]);  mul_61 = None
        mm_29: "f32[8192, 512]" = torch.ops.aten.mm.default(view_124, permute_55);  permute_55 = None
        view_125: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_29, [8, 1024, 512]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_43: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_21: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_43, 0.1);  inductor_random_default_43 = None
        mul_62: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_21, view_125);  view_125 = None
        mul_63: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_62, 1.1111111111111112);  mul_62 = None
        add_31: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_29, mul_63);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_11: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
        mean_10: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_32: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_64: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_65: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_44, mul_64);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_56: "f32[512, 512]" = torch.ops.aten.permute.default(primals_45, [1, 0])
        view_126: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_65, [8192, 512]);  mul_65 = None
        mm_30: "f32[8192, 512]" = torch.ops.aten.mm.default(view_126, permute_56);  permute_56 = None
        view_127: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_30, [8, 1024, 512]);  mm_30 = None
        view_128: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_127, [8, 1024, -1, 64]);  view_127 = None
        permute_57: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_58: "f32[512, 512]" = torch.ops.aten.permute.default(primals_46, [1, 0])
        mm_31: "f32[8192, 512]" = torch.ops.aten.mm.default(view_126, permute_58);  permute_58 = None
        view_130: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_31, [8, 1024, 512]);  mm_31 = None
        view_131: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_130, [8, 1024, -1, 64]);  view_130 = None
        permute_59: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_60: "f32[512, 512]" = torch.ops.aten.permute.default(primals_47, [1, 0])
        mm_32: "f32[8192, 512]" = torch.ops.aten.mm.default(view_126, permute_60);  permute_60 = None
        view_133: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_32, [8, 1024, 512]);  mm_32 = None
        view_134: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_133, [8, 1024, -1, 64]);  view_133 = None
        permute_61: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_62: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        expand_21: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_57, [8, 8, 1024, 64]);  permute_57 = None
        clone_20: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_135: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_20, [64, 1024, 64]);  clone_20 = None
        expand_22: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_62, [8, 8, 64, 1024]);  permute_62 = None
        clone_21: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_136: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_21, [64, 64, 1024]);  clone_21 = None
        bmm_10: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_135, view_136)
        view_137: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_10, [8, 8, 1024, 1024]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_33: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_137, add_7);  view_137 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_5: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_33, [-1], True)
        sub_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_33, amax_5);  add_33 = amax_5 = None
        exp_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_6: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_42: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_22: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_42, 0.1);  inductor_random_default_42 = None
        mul_66: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_22, div_7)
        mul_67: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_23: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_67, [8, 8, 1024, 1024]);  mul_67 = None
        view_141: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_23, [64, 1024, 1024]);  expand_23 = None
        expand_24: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_61, [8, 8, 1024, 64]);  permute_61 = None
        clone_22: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_142: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_22, [64, 1024, 64]);  clone_22 = None
        bmm_11: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_141, view_142)
        view_143: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_11, [8, 8, 1024, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        clone_23: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_144: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_23, [8, 1024, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_64: "f32[512, 512]" = torch.ops.aten.permute.default(primals_48, [1, 0])
        view_145: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_144, [8192, 512]);  view_144 = None
        mm_33: "f32[8192, 512]" = torch.ops.aten.mm.default(view_145, permute_64);  permute_64 = None
        view_146: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_33, [8, 1024, 512]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_41: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_23: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_41, 0.1);  inductor_random_default_41 = None
        mul_68: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_23, view_146);  view_146 = None
        mul_69: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_68, 1.1111111111111112);  mul_68 = None
        add_34: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_31, mul_69);  mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_12: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
        mean_11: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_70: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_71: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_49, mul_70);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_65: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_50, [1, 0])
        view_147: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_71, [8192, 512]);  mul_71 = None
        mm_34: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_147, permute_65);  permute_65 = None
        view_148: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_34, [8, 1024, 2048]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_5: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_148);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_40: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_24: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_40, 0.1);  inductor_random_default_40 = None
        mul_72: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_24, relu_5)
        mul_73: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_72, 1.1111111111111112);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_66: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_51, [1, 0])
        view_149: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_73, [8192, 2048]);  mul_73 = None
        mm_35: "f32[8192, 512]" = torch.ops.aten.mm.default(view_149, permute_66);  permute_66 = None
        view_150: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_35, [8, 1024, 512]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_39: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_25: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_39, 0.1);  inductor_random_default_39 = None
        mul_74: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_25, view_150);  view_150 = None
        mul_75: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None
        add_36: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_34, mul_75);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_13: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_36, 2)
        mean_12: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_37: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_76: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_36, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_77: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_52, mul_76);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_38: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_26: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_38, 0.1);  inductor_random_default_38 = None
        mul_78: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_26, mul_77);  mul_77 = None
        mul_79: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:592 in _shift_right, code: shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        full_1: "i64[8, 1024]" = torch.ops.aten.full.default([8, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:593 in _shift_right, code: shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        slice_1: "i64[8, 1023]" = torch.ops.aten.slice.Tensor(primals_53, 1, 0, -1)
        clone_24: "i64[8, 1023]" = torch.ops.aten.clone.default(slice_1);  slice_1 = None
        slice_2: "i64[8, 1023]" = torch.ops.aten.slice.Tensor(full_1, 1, 1, 9223372036854775807)
        copy: "i64[8, 1023]" = torch.ops.aten.copy.default(slice_2, clone_24);  slice_2 = clone_24 = None
        slice_scatter: "i64[8, 1024]" = torch.ops.aten.slice_scatter.default(full_1, copy, 1, 1, 9223372036854775807);  full_1 = copy = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:594 in _shift_right, code: shifted_input_ids[..., 0] = decoder_start_token_id
        full_default_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_1: "i64[8]" = torch.ops.aten.select.int(slice_scatter, 1, 0)
        copy_1: "i64[8]" = torch.ops.aten.copy.default(select_1, full_default_3);  select_1 = full_default_3 = None
        select_scatter: "i64[8, 1024]" = torch.ops.aten.select_scatter.default(slice_scatter, copy_1, 1, 0);  slice_scatter = copy_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:599 in _shift_right, code: shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        eq: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(select_scatter, -100)
        full_default_4: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[8, 1024]" = torch.ops.aten.where.self(eq, full_default_4, select_scatter);  eq = select_scatter = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_2: "f32[8, 1024, 512]" = torch.ops.aten.embedding.default(primals_2, where_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_11: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_11, unsqueeze_2);  unsqueeze_11 = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_25: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le, [8, -1, 1024, 1024]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        where_3: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_25, full_default, full_default_1);  expand_25 = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        inductor_lookup_seed_default_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_37: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_27: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_37, 0.1);  inductor_random_default_37 = None
        mul_80: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_27, embedding_2)
        mul_81: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_80, 1.1111111111111112);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_14: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_81, 2)
        mean_13: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_82: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_81, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_83: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_54, mul_82);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(primals_55, [1, 0])
        view_153: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_83, [8192, 512]);  mul_83 = None
        mm_36: "f32[8192, 512]" = torch.ops.aten.mm.default(view_153, permute_67);  permute_67 = None
        view_154: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_36, [8, 1024, 512]);  mm_36 = None
        view_155: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_154, [8, 1024, -1, 64]);  view_154 = None
        permute_68: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_69: "f32[512, 512]" = torch.ops.aten.permute.default(primals_56, [1, 0])
        mm_37: "f32[8192, 512]" = torch.ops.aten.mm.default(view_153, permute_69);  permute_69 = None
        view_157: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_37, [8, 1024, 512]);  mm_37 = None
        view_158: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_157, [8, 1024, -1, 64]);  view_157 = None
        permute_70: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_71: "f32[512, 512]" = torch.ops.aten.permute.default(primals_57, [1, 0])
        mm_38: "f32[8192, 512]" = torch.ops.aten.mm.default(view_153, permute_71);  permute_71 = None
        view_160: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_38, [8, 1024, 512]);  mm_38 = None
        view_161: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_160, [8, 1024, -1, 64]);  view_160 = None
        permute_72: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_73: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        expand_27: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_68, [8, 8, 1024, 64]);  permute_68 = None
        clone_25: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_162: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_25, [64, 1024, 64]);  clone_25 = None
        expand_28: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_73, [8, 8, 64, 1024]);  permute_73 = None
        clone_26: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_163: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_26, [64, 64, 1024]);  clone_26 = None
        bmm_12: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_162, view_163)
        view_164: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_12, [8, 8, 1024, 1024]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:203 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_default_9: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub, full_default_9);  sub = full_default_9 = None
        neg: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_1: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_3: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_8: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_3, 16);  convert_element_type_3 = None
        log_1: "f32[1024, 1024]" = torch.ops.aten.log.default(div_8);  div_8 = None
        div_9: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_84: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_9, 16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_4: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_84, torch.int64);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_44: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_4, 16);  convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_10: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_2: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_44, full_default_10);  add_44 = full_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_5: "i64[1024, 1024]" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_45: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where_5, 0);  where_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_3: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(primals_58, add_45);  primals_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_74: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_17: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_74, 0);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_46: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_17, where_3);  unsqueeze_17 = where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_47: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_164, add_46);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_6: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_47, [-1], True)
        sub_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_47, amax_6);  add_47 = amax_6 = None
        exp_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_7: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_36: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        gt_28: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_85: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_28, div_10)
        mul_86: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_29: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_86, [8, 8, 1024, 1024]);  mul_86 = None
        view_168: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_29, [64, 1024, 1024]);  expand_29 = None
        expand_30: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_72, [8, 8, 1024, 64]);  permute_72 = None
        clone_27: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_169: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_27, [64, 1024, 64]);  clone_27 = None
        bmm_13: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_168, view_169)
        view_170: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_13, [8, 8, 1024, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_75: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_28: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_171: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_28, [8, 1024, -1]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_76: "f32[512, 512]" = torch.ops.aten.permute.default(primals_59, [1, 0])
        view_172: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_171, [8192, 512]);  view_171 = None
        mm_39: "f32[8192, 512]" = torch.ops.aten.mm.default(view_172, permute_76);  permute_76 = None
        view_173: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_39, [8, 1024, 512]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_35: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        gt_29: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_35, 0.1);  inductor_random_default_35 = None
        mul_87: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_29, view_173);  view_173 = None
        mul_88: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_87, 1.1111111111111112);  mul_87 = None
        add_48: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_81, mul_88);  mul_81 = mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_15: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_48, 2)
        mean_14: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_89: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_48, rsqrt_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_90: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_60, mul_89);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_77: "f32[512, 512]" = torch.ops.aten.permute.default(primals_61, [1, 0])
        view_174: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_90, [8192, 512]);  mul_90 = None
        mm_40: "f32[8192, 512]" = torch.ops.aten.mm.default(view_174, permute_77);  permute_77 = None
        view_175: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_40, [8, 1024, 512]);  mm_40 = None
        view_176: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_175, [8, 1024, -1, 64]);  view_175 = None
        permute_78: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_79: "f32[512, 512]" = torch.ops.aten.permute.default(primals_62, [1, 0])
        view_177: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_41: "f32[8192, 512]" = torch.ops.aten.mm.default(view_177, permute_79);  view_177 = permute_79 = None
        view_178: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_41, [8, 1024, 512]);  mm_41 = None
        view_179: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_178, [8, 1024, -1, 64]);  view_178 = None
        permute_80: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_179, [0, 2, 1, 3]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(primals_63, [1, 0])
        view_180: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_42: "f32[8192, 512]" = torch.ops.aten.mm.default(view_180, permute_81);  view_180 = permute_81 = None
        view_181: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_42, [8, 1024, 512]);  mm_42 = None
        view_182: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_181, [8, 1024, -1, 64]);  view_181 = None
        permute_82: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_83: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        expand_31: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_78, [8, 8, 1024, 64]);  permute_78 = None
        clone_29: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_183: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_29, [64, 1024, 64]);  clone_29 = None
        expand_32: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_83, [8, 8, 64, 1024]);  permute_83 = None
        clone_30: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_184: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_30, [64, 64, 1024]);  clone_30 = None
        bmm_14: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_183, view_184)
        view_185: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_14, [8, 8, 1024, 1024]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:300 in forward, code: position_bias = torch.zeros(
        full_default_11: "f32[1, 8, 1024, 1024]" = torch.ops.aten.full.default([1, 8, 1024, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_50: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(full_default_11, where);  full_default_11 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_51: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_185, add_50);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_7: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_51, [-1], True)
        sub_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_51, amax_7);  add_51 = amax_7 = None
        exp_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_8: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_34: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        gt_30: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_34, 0.1);  inductor_random_default_34 = None
        mul_91: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_30, div_11)
        mul_92: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_91, 1.1111111111111112);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_33: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_92, [8, 8, 1024, 1024]);  mul_92 = None
        view_189: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_33, [64, 1024, 1024]);  expand_33 = None
        expand_34: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_82, [8, 8, 1024, 64]);  permute_82 = None
        clone_31: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_190: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_31, [64, 1024, 64]);  clone_31 = None
        bmm_15: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_189, view_190)
        view_191: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_15, [8, 8, 1024, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
        clone_32: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_192: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_32, [8, 1024, -1]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_85: "f32[512, 512]" = torch.ops.aten.permute.default(primals_64, [1, 0])
        view_193: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_192, [8192, 512]);  view_192 = None
        mm_43: "f32[8192, 512]" = torch.ops.aten.mm.default(view_193, permute_85);  permute_85 = None
        view_194: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_43, [8, 1024, 512]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_33: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_31: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_33, 0.1);  inductor_random_default_33 = None
        mul_93: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_31, view_194);  view_194 = None
        mul_94: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_93, 1.1111111111111112);  mul_93 = None
        add_52: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_48, mul_94);  mul_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_16: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_52, 2)
        mean_15: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_53: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_95: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_96: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_65, mul_95);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_86: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_66, [1, 0])
        view_195: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_96, [8192, 512]);  mul_96 = None
        mm_44: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_195, permute_86);  permute_86 = None
        view_196: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_44, [8, 1024, 2048]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_6: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_196);  view_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_32: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_32: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_32, 0.1);  inductor_random_default_32 = None
        mul_97: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_32, relu_6)
        mul_98: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_97, 1.1111111111111112);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_87: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_67, [1, 0])
        view_197: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_98, [8192, 2048]);  mul_98 = None
        mm_45: "f32[8192, 512]" = torch.ops.aten.mm.default(view_197, permute_87);  permute_87 = None
        view_198: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_45, [8, 1024, 512]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_31: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        gt_33: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_31, 0.1);  inductor_random_default_31 = None
        mul_99: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_33, view_198);  view_198 = None
        mul_100: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_99, 1.1111111111111112);  mul_99 = None
        add_54: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_52, mul_100);  mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_17: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_54, 2)
        mean_16: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_55: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_101: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_102: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_68, mul_101);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_88: "f32[512, 512]" = torch.ops.aten.permute.default(primals_69, [1, 0])
        view_199: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_102, [8192, 512]);  mul_102 = None
        mm_46: "f32[8192, 512]" = torch.ops.aten.mm.default(view_199, permute_88);  permute_88 = None
        view_200: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_46, [8, 1024, 512]);  mm_46 = None
        view_201: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_200, [8, 1024, -1, 64]);  view_200 = None
        permute_89: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_90: "f32[512, 512]" = torch.ops.aten.permute.default(primals_70, [1, 0])
        mm_47: "f32[8192, 512]" = torch.ops.aten.mm.default(view_199, permute_90);  permute_90 = None
        view_203: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_47, [8, 1024, 512]);  mm_47 = None
        view_204: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_203, [8, 1024, -1, 64]);  view_203 = None
        permute_91: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_92: "f32[512, 512]" = torch.ops.aten.permute.default(primals_71, [1, 0])
        mm_48: "f32[8192, 512]" = torch.ops.aten.mm.default(view_199, permute_92);  permute_92 = None
        view_206: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_48, [8, 1024, 512]);  mm_48 = None
        view_207: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_206, [8, 1024, -1, 64]);  view_206 = None
        permute_93: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_94: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        expand_35: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_89, [8, 8, 1024, 64]);  permute_89 = None
        clone_33: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_208: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_33, [64, 1024, 64]);  clone_33 = None
        expand_36: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_94, [8, 8, 64, 1024]);  permute_94 = None
        clone_34: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_209: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_34, [64, 64, 1024]);  clone_34 = None
        bmm_16: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_208, view_209)
        view_210: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_16, [8, 8, 1024, 1024]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_56: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_210, add_46);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_8: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_56, [-1], True)
        sub_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_56, amax_8);  add_56 = amax_8 = None
        exp_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_9: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_30: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        gt_34: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_30, 0.1);  inductor_random_default_30 = None
        mul_103: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_34, div_12)
        mul_104: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_103, 1.1111111111111112);  mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_37: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_104, [8, 8, 1024, 1024]);  mul_104 = None
        view_214: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_37, [64, 1024, 1024]);  expand_37 = None
        expand_38: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_93, [8, 8, 1024, 64]);  permute_93 = None
        clone_35: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_215: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_35, [64, 1024, 64]);  clone_35 = None
        bmm_17: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_214, view_215)
        view_216: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_17, [8, 8, 1024, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
        clone_36: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_217: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_36, [8, 1024, -1]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_96: "f32[512, 512]" = torch.ops.aten.permute.default(primals_72, [1, 0])
        view_218: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_217, [8192, 512]);  view_217 = None
        mm_49: "f32[8192, 512]" = torch.ops.aten.mm.default(view_218, permute_96);  permute_96 = None
        view_219: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_49, [8, 1024, 512]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_29: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_35: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_29, 0.1);  inductor_random_default_29 = None
        mul_105: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_35, view_219);  view_219 = None
        mul_106: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None
        add_57: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_54, mul_106);  mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_18: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_57, 2)
        mean_17: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_58: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_107: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_57, rsqrt_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_108: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_73, mul_107);  mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_97: "f32[512, 512]" = torch.ops.aten.permute.default(primals_74, [1, 0])
        view_220: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_108, [8192, 512]);  mul_108 = None
        mm_50: "f32[8192, 512]" = torch.ops.aten.mm.default(view_220, permute_97);  permute_97 = None
        view_221: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_50, [8, 1024, 512]);  mm_50 = None
        view_222: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_221, [8, 1024, -1, 64]);  view_221 = None
        permute_98: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_99: "f32[512, 512]" = torch.ops.aten.permute.default(primals_75, [1, 0])
        view_223: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_51: "f32[8192, 512]" = torch.ops.aten.mm.default(view_223, permute_99);  view_223 = permute_99 = None
        view_224: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_51, [8, 1024, 512]);  mm_51 = None
        view_225: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_224, [8, 1024, -1, 64]);  view_224 = None
        permute_100: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_101: "f32[512, 512]" = torch.ops.aten.permute.default(primals_76, [1, 0])
        view_226: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_52: "f32[8192, 512]" = torch.ops.aten.mm.default(view_226, permute_101);  view_226 = permute_101 = None
        view_227: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_52, [8, 1024, 512]);  mm_52 = None
        view_228: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_227, [8, 1024, -1, 64]);  view_227 = None
        permute_102: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_103: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2]);  permute_100 = None
        expand_39: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_98, [8, 8, 1024, 64]);  permute_98 = None
        clone_37: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_229: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_37, [64, 1024, 64]);  clone_37 = None
        expand_40: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_103, [8, 8, 64, 1024]);  permute_103 = None
        clone_38: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_230: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_38, [64, 64, 1024]);  clone_38 = None
        bmm_18: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_229, view_230)
        view_231: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_18, [8, 8, 1024, 1024]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_59: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_231, add_50);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_9: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_59, [-1], True)
        sub_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_59, amax_9);  add_59 = amax_9 = None
        exp_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_10: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_28: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        gt_36: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_28, 0.1);  inductor_random_default_28 = None
        mul_109: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_36, div_13)
        mul_110: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_109, 1.1111111111111112);  mul_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_41: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_110, [8, 8, 1024, 1024]);  mul_110 = None
        view_235: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_41, [64, 1024, 1024]);  expand_41 = None
        expand_42: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_102, [8, 8, 1024, 64]);  permute_102 = None
        clone_39: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_236: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_39, [64, 1024, 64]);  clone_39 = None
        bmm_19: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_235, view_236)
        view_237: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_19, [8, 8, 1024, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_104: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
        clone_40: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_238: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_40, [8, 1024, -1]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_105: "f32[512, 512]" = torch.ops.aten.permute.default(primals_77, [1, 0])
        view_239: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_238, [8192, 512]);  view_238 = None
        mm_53: "f32[8192, 512]" = torch.ops.aten.mm.default(view_239, permute_105);  permute_105 = None
        view_240: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_53, [8, 1024, 512]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_27: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        gt_37: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_27, 0.1);  inductor_random_default_27 = None
        mul_111: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_37, view_240);  view_240 = None
        mul_112: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_111, 1.1111111111111112);  mul_111 = None
        add_60: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_57, mul_112);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_19: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_60, 2)
        mean_18: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_61: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_113: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_60, rsqrt_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_114: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_78, mul_113);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_106: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_79, [1, 0])
        view_241: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_114, [8192, 512]);  mul_114 = None
        mm_54: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_241, permute_106);  permute_106 = None
        view_242: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_54, [8, 1024, 2048]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_7: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_242);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_26: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        gt_38: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_115: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_38, relu_7)
        mul_116: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_115, 1.1111111111111112);  mul_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_107: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_80, [1, 0])
        view_243: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_116, [8192, 2048]);  mul_116 = None
        mm_55: "f32[8192, 512]" = torch.ops.aten.mm.default(view_243, permute_107);  permute_107 = None
        view_244: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_55, [8, 1024, 512]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_25: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        gt_39: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_117: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_39, view_244);  view_244 = None
        mul_118: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_117, 1.1111111111111112);  mul_117 = None
        add_62: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_60, mul_118);  mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_20: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_62, 2)
        mean_19: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_119: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_120: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_81, mul_119);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_108: "f32[512, 512]" = torch.ops.aten.permute.default(primals_82, [1, 0])
        view_245: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_120, [8192, 512]);  mul_120 = None
        mm_56: "f32[8192, 512]" = torch.ops.aten.mm.default(view_245, permute_108);  permute_108 = None
        view_246: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_56, [8, 1024, 512]);  mm_56 = None
        view_247: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_246, [8, 1024, -1, 64]);  view_246 = None
        permute_109: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_110: "f32[512, 512]" = torch.ops.aten.permute.default(primals_83, [1, 0])
        mm_57: "f32[8192, 512]" = torch.ops.aten.mm.default(view_245, permute_110);  permute_110 = None
        view_249: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_57, [8, 1024, 512]);  mm_57 = None
        view_250: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_249, [8, 1024, -1, 64]);  view_249 = None
        permute_111: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_112: "f32[512, 512]" = torch.ops.aten.permute.default(primals_84, [1, 0])
        mm_58: "f32[8192, 512]" = torch.ops.aten.mm.default(view_245, permute_112);  permute_112 = None
        view_252: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_58, [8, 1024, 512]);  mm_58 = None
        view_253: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_252, [8, 1024, -1, 64]);  view_252 = None
        permute_113: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_114: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_111, [0, 1, 3, 2]);  permute_111 = None
        expand_43: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_109, [8, 8, 1024, 64]);  permute_109 = None
        clone_41: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_254: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_41, [64, 1024, 64]);  clone_41 = None
        expand_44: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_114, [8, 8, 64, 1024]);  permute_114 = None
        clone_42: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_255: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_42, [64, 64, 1024]);  clone_42 = None
        bmm_20: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_254, view_255)
        view_256: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_20, [8, 8, 1024, 1024]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_64: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_256, add_46);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_10: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_64, [-1], True)
        sub_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_64, amax_10);  add_64 = amax_10 = None
        exp_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_11: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_24: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        gt_40: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_121: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_40, div_14)
        mul_122: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_121, 1.1111111111111112);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_45: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_122, [8, 8, 1024, 1024]);  mul_122 = None
        view_260: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_45, [64, 1024, 1024]);  expand_45 = None
        expand_46: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_113, [8, 8, 1024, 64]);  permute_113 = None
        clone_43: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_261: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_43, [64, 1024, 64]);  clone_43 = None
        bmm_21: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_260, view_261)
        view_262: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_21, [8, 8, 1024, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_115: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
        clone_44: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_263: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_44, [8, 1024, -1]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_116: "f32[512, 512]" = torch.ops.aten.permute.default(primals_85, [1, 0])
        view_264: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_263, [8192, 512]);  view_263 = None
        mm_59: "f32[8192, 512]" = torch.ops.aten.mm.default(view_264, permute_116);  permute_116 = None
        view_265: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_59, [8, 1024, 512]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_23: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        gt_41: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_123: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_41, view_265);  view_265 = None
        mul_124: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_123, 1.1111111111111112);  mul_123 = None
        add_65: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_62, mul_124);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_21: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_65, 2)
        mean_20: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_66: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_125: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_65, rsqrt_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_126: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_86, mul_125);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_117: "f32[512, 512]" = torch.ops.aten.permute.default(primals_87, [1, 0])
        view_266: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_126, [8192, 512]);  mul_126 = None
        mm_60: "f32[8192, 512]" = torch.ops.aten.mm.default(view_266, permute_117);  permute_117 = None
        view_267: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_60, [8, 1024, 512]);  mm_60 = None
        view_268: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_267, [8, 1024, -1, 64]);  view_267 = None
        permute_118: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_119: "f32[512, 512]" = torch.ops.aten.permute.default(primals_88, [1, 0])
        view_269: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_61: "f32[8192, 512]" = torch.ops.aten.mm.default(view_269, permute_119);  view_269 = permute_119 = None
        view_270: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_61, [8, 1024, 512]);  mm_61 = None
        view_271: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_270, [8, 1024, -1, 64]);  view_270 = None
        permute_120: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_121: "f32[512, 512]" = torch.ops.aten.permute.default(primals_89, [1, 0])
        view_272: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_62: "f32[8192, 512]" = torch.ops.aten.mm.default(view_272, permute_121);  view_272 = permute_121 = None
        view_273: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_62, [8, 1024, 512]);  mm_62 = None
        view_274: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_273, [8, 1024, -1, 64]);  view_273 = None
        permute_122: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_123: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2]);  permute_120 = None
        expand_47: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_118, [8, 8, 1024, 64]);  permute_118 = None
        clone_45: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_275: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_45, [64, 1024, 64]);  clone_45 = None
        expand_48: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_123, [8, 8, 64, 1024]);  permute_123 = None
        clone_46: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_276: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_46, [64, 64, 1024]);  clone_46 = None
        bmm_22: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_275, view_276)
        view_277: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_22, [8, 8, 1024, 1024]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_67: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_277, add_50);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_11: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_67, [-1], True)
        sub_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_67, amax_11);  add_67 = amax_11 = None
        exp_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_12: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_22: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        gt_42: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_127: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_42, div_15)
        mul_128: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_127, 1.1111111111111112);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_49: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_128, [8, 8, 1024, 1024]);  mul_128 = None
        view_281: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_49, [64, 1024, 1024]);  expand_49 = None
        expand_50: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_122, [8, 8, 1024, 64]);  permute_122 = None
        clone_47: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_282: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_47, [64, 1024, 64]);  clone_47 = None
        bmm_23: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_281, view_282)
        view_283: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_23, [8, 8, 1024, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_124: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
        clone_48: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_284: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_48, [8, 1024, -1]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_125: "f32[512, 512]" = torch.ops.aten.permute.default(primals_90, [1, 0])
        view_285: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_284, [8192, 512]);  view_284 = None
        mm_63: "f32[8192, 512]" = torch.ops.aten.mm.default(view_285, permute_125);  permute_125 = None
        view_286: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_63, [8, 1024, 512]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_21: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        gt_43: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_129: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_43, view_286);  view_286 = None
        mul_130: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_129, 1.1111111111111112);  mul_129 = None
        add_68: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_65, mul_130);  mul_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_22: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_68, 2)
        mean_21: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_69: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_131: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_68, rsqrt_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_132: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_91, mul_131);  mul_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_126: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_92, [1, 0])
        view_287: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_132, [8192, 512]);  mul_132 = None
        mm_64: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_287, permute_126);  permute_126 = None
        view_288: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_64, [8, 1024, 2048]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_8: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_288);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_20: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        gt_44: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_133: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_44, relu_8)
        mul_134: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_133, 1.1111111111111112);  mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_127: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_93, [1, 0])
        view_289: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_134, [8192, 2048]);  mul_134 = None
        mm_65: "f32[8192, 512]" = torch.ops.aten.mm.default(view_289, permute_127);  permute_127 = None
        view_290: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_65, [8, 1024, 512]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_19: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        gt_45: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_135: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_45, view_290);  view_290 = None
        mul_136: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None
        add_70: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_68, mul_136);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_23: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 2)
        mean_22: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_71: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_137: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_138: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_94, mul_137);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_128: "f32[512, 512]" = torch.ops.aten.permute.default(primals_95, [1, 0])
        view_291: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_138, [8192, 512]);  mul_138 = None
        mm_66: "f32[8192, 512]" = torch.ops.aten.mm.default(view_291, permute_128);  permute_128 = None
        view_292: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_66, [8, 1024, 512]);  mm_66 = None
        view_293: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_292, [8, 1024, -1, 64]);  view_292 = None
        permute_129: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_130: "f32[512, 512]" = torch.ops.aten.permute.default(primals_96, [1, 0])
        mm_67: "f32[8192, 512]" = torch.ops.aten.mm.default(view_291, permute_130);  permute_130 = None
        view_295: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_67, [8, 1024, 512]);  mm_67 = None
        view_296: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_295, [8, 1024, -1, 64]);  view_295 = None
        permute_131: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_132: "f32[512, 512]" = torch.ops.aten.permute.default(primals_97, [1, 0])
        mm_68: "f32[8192, 512]" = torch.ops.aten.mm.default(view_291, permute_132);  permute_132 = None
        view_298: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_68, [8, 1024, 512]);  mm_68 = None
        view_299: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_298, [8, 1024, -1, 64]);  view_298 = None
        permute_133: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_134: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2]);  permute_131 = None
        expand_51: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_129, [8, 8, 1024, 64]);  permute_129 = None
        clone_49: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_300: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_49, [64, 1024, 64]);  clone_49 = None
        expand_52: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_134, [8, 8, 64, 1024]);  permute_134 = None
        clone_50: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_301: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_50, [64, 64, 1024]);  clone_50 = None
        bmm_24: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_300, view_301)
        view_302: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_24, [8, 8, 1024, 1024]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_72: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_302, add_46);  view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_12: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_72, [-1], True)
        sub_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_72, amax_12);  add_72 = amax_12 = None
        exp_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_13: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_18: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        gt_46: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_139: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_46, div_16)
        mul_140: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_139, 1.1111111111111112);  mul_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_53: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_140, [8, 8, 1024, 1024]);  mul_140 = None
        view_306: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_53, [64, 1024, 1024]);  expand_53 = None
        expand_54: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_133, [8, 8, 1024, 64]);  permute_133 = None
        clone_51: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_307: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_51, [64, 1024, 64]);  clone_51 = None
        bmm_25: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_306, view_307)
        view_308: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_25, [8, 8, 1024, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_135: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
        clone_52: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_309: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_52, [8, 1024, -1]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_136: "f32[512, 512]" = torch.ops.aten.permute.default(primals_98, [1, 0])
        view_310: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_309, [8192, 512]);  view_309 = None
        mm_69: "f32[8192, 512]" = torch.ops.aten.mm.default(view_310, permute_136);  permute_136 = None
        view_311: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_69, [8, 1024, 512]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_46: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_17: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        gt_47: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_141: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_47, view_311);  view_311 = None
        mul_142: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_141, 1.1111111111111112);  mul_141 = None
        add_73: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_70, mul_142);  mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_24: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_73, 2)
        mean_23: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_74: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_143: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_73, rsqrt_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_144: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_99, mul_143);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_137: "f32[512, 512]" = torch.ops.aten.permute.default(primals_100, [1, 0])
        view_312: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_144, [8192, 512]);  mul_144 = None
        mm_70: "f32[8192, 512]" = torch.ops.aten.mm.default(view_312, permute_137);  permute_137 = None
        view_313: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_70, [8, 1024, 512]);  mm_70 = None
        view_314: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_313, [8, 1024, -1, 64]);  view_313 = None
        permute_138: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_139: "f32[512, 512]" = torch.ops.aten.permute.default(primals_101, [1, 0])
        view_315: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_71: "f32[8192, 512]" = torch.ops.aten.mm.default(view_315, permute_139);  view_315 = permute_139 = None
        view_316: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_71, [8, 1024, 512]);  mm_71 = None
        view_317: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_316, [8, 1024, -1, 64]);  view_316 = None
        permute_140: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_141: "f32[512, 512]" = torch.ops.aten.permute.default(primals_102, [1, 0])
        view_318: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_72: "f32[8192, 512]" = torch.ops.aten.mm.default(view_318, permute_141);  view_318 = permute_141 = None
        view_319: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_72, [8, 1024, 512]);  mm_72 = None
        view_320: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_319, [8, 1024, -1, 64]);  view_319 = None
        permute_142: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_143: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 2]);  permute_140 = None
        expand_55: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_138, [8, 8, 1024, 64]);  permute_138 = None
        clone_53: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_321: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_53, [64, 1024, 64]);  clone_53 = None
        expand_56: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_143, [8, 8, 64, 1024]);  permute_143 = None
        clone_54: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_322: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_54, [64, 64, 1024]);  clone_54 = None
        bmm_26: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_321, view_322)
        view_323: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_26, [8, 8, 1024, 1024]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_75: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_323, add_50);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_13: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_75, [-1], True)
        sub_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_75, amax_13);  add_75 = amax_13 = None
        exp_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_14: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_47: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_16: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        gt_48: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_145: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_48, div_17)
        mul_146: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_145, 1.1111111111111112);  mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_57: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_146, [8, 8, 1024, 1024]);  mul_146 = None
        view_327: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_57, [64, 1024, 1024]);  expand_57 = None
        expand_58: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_142, [8, 8, 1024, 64]);  permute_142 = None
        clone_55: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_328: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_55, [64, 1024, 64]);  clone_55 = None
        bmm_27: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_327, view_328)
        view_329: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_27, [8, 8, 1024, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_144: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None
        clone_56: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_330: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_56, [8, 1024, -1]);  clone_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_145: "f32[512, 512]" = torch.ops.aten.permute.default(primals_103, [1, 0])
        view_331: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_330, [8192, 512]);  view_330 = None
        mm_73: "f32[8192, 512]" = torch.ops.aten.mm.default(view_331, permute_145);  permute_145 = None
        view_332: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_73, [8, 1024, 512]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_48: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48)
        inductor_random_default_15: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        gt_49: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_147: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_49, view_332);  view_332 = None
        mul_148: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_147, 1.1111111111111112);  mul_147 = None
        add_76: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_73, mul_148);  mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_25: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_76, 2)
        mean_24: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_77: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_149: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_76, rsqrt_24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_150: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_104, mul_149);  mul_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_146: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_105, [1, 0])
        view_333: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_150, [8192, 512]);  mul_150 = None
        mm_74: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_333, permute_146);  permute_146 = None
        view_334: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_74, [8, 1024, 2048]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_9: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_334);  view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_49: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 49)
        inductor_random_default_14: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_49, 'rand');  inductor_lookup_seed_default_49 = None
        gt_50: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_151: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_50, relu_9)
        mul_152: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_151, 1.1111111111111112);  mul_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_147: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_106, [1, 0])
        view_335: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_152, [8192, 2048]);  mul_152 = None
        mm_75: "f32[8192, 512]" = torch.ops.aten.mm.default(view_335, permute_147);  permute_147 = None
        view_336: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_75, [8, 1024, 512]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_50: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50)
        inductor_random_default_13: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_50, 'rand');  inductor_lookup_seed_default_50 = None
        gt_51: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_153: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_51, view_336);  view_336 = None
        mul_154: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_153, 1.1111111111111112);  mul_153 = None
        add_78: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_76, mul_154);  mul_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_26: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 2)
        mean_25: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_79: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_155: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_156: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_107, mul_155);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_148: "f32[512, 512]" = torch.ops.aten.permute.default(primals_108, [1, 0])
        view_337: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_156, [8192, 512]);  mul_156 = None
        mm_76: "f32[8192, 512]" = torch.ops.aten.mm.default(view_337, permute_148);  permute_148 = None
        view_338: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_76, [8, 1024, 512]);  mm_76 = None
        view_339: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_338, [8, 1024, -1, 64]);  view_338 = None
        permute_149: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_150: "f32[512, 512]" = torch.ops.aten.permute.default(primals_109, [1, 0])
        mm_77: "f32[8192, 512]" = torch.ops.aten.mm.default(view_337, permute_150);  permute_150 = None
        view_341: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_77, [8, 1024, 512]);  mm_77 = None
        view_342: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_341, [8, 1024, -1, 64]);  view_341 = None
        permute_151: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_152: "f32[512, 512]" = torch.ops.aten.permute.default(primals_110, [1, 0])
        mm_78: "f32[8192, 512]" = torch.ops.aten.mm.default(view_337, permute_152);  permute_152 = None
        view_344: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_78, [8, 1024, 512]);  mm_78 = None
        view_345: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_344, [8, 1024, -1, 64]);  view_344 = None
        permute_153: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_154: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_151, [0, 1, 3, 2]);  permute_151 = None
        expand_59: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_149, [8, 8, 1024, 64]);  permute_149 = None
        clone_57: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_346: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_57, [64, 1024, 64]);  clone_57 = None
        expand_60: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_154, [8, 8, 64, 1024]);  permute_154 = None
        clone_58: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_347: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_58, [64, 64, 1024]);  clone_58 = None
        bmm_28: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_346, view_347)
        view_348: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_28, [8, 8, 1024, 1024]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_80: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_348, add_46);  view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_14: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_80, [-1], True)
        sub_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_80, amax_14);  add_80 = amax_14 = None
        exp_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_15: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_51: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 51)
        inductor_random_default_12: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_51, 'rand');  inductor_lookup_seed_default_51 = None
        gt_52: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_157: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_52, div_18)
        mul_158: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_157, 1.1111111111111112);  mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_61: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_158, [8, 8, 1024, 1024]);  mul_158 = None
        view_352: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_61, [64, 1024, 1024]);  expand_61 = None
        expand_62: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_153, [8, 8, 1024, 64]);  permute_153 = None
        clone_59: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_353: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_59, [64, 1024, 64]);  clone_59 = None
        bmm_29: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_352, view_353)
        view_354: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_29, [8, 8, 1024, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        clone_60: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_355: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_60, [8, 1024, -1]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_156: "f32[512, 512]" = torch.ops.aten.permute.default(primals_111, [1, 0])
        view_356: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_355, [8192, 512]);  view_355 = None
        mm_79: "f32[8192, 512]" = torch.ops.aten.mm.default(view_356, permute_156);  permute_156 = None
        view_357: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_79, [8, 1024, 512]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_52: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 52)
        inductor_random_default_11: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_52, 'rand');  inductor_lookup_seed_default_52 = None
        gt_53: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_159: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_53, view_357);  view_357 = None
        mul_160: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_159, 1.1111111111111112);  mul_159 = None
        add_81: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_78, mul_160);  mul_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_27: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_81, 2)
        mean_26: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_82: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_161: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, rsqrt_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_162: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_112, mul_161);  mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_157: "f32[512, 512]" = torch.ops.aten.permute.default(primals_113, [1, 0])
        view_358: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_162, [8192, 512]);  mul_162 = None
        mm_80: "f32[8192, 512]" = torch.ops.aten.mm.default(view_358, permute_157);  permute_157 = None
        view_359: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_80, [8, 1024, 512]);  mm_80 = None
        view_360: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_359, [8, 1024, -1, 64]);  view_359 = None
        permute_158: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_159: "f32[512, 512]" = torch.ops.aten.permute.default(primals_114, [1, 0])
        view_361: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_81: "f32[8192, 512]" = torch.ops.aten.mm.default(view_361, permute_159);  view_361 = permute_159 = None
        view_362: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_81, [8, 1024, 512]);  mm_81 = None
        view_363: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_362, [8, 1024, -1, 64]);  view_362 = None
        permute_160: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_363, [0, 2, 1, 3]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_161: "f32[512, 512]" = torch.ops.aten.permute.default(primals_115, [1, 0])
        view_364: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_82: "f32[8192, 512]" = torch.ops.aten.mm.default(view_364, permute_161);  view_364 = permute_161 = None
        view_365: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_82, [8, 1024, 512]);  mm_82 = None
        view_366: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_365, [8, 1024, -1, 64]);  view_365 = None
        permute_162: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_163: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_160, [0, 1, 3, 2]);  permute_160 = None
        expand_63: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_158, [8, 8, 1024, 64]);  permute_158 = None
        clone_61: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_367: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_61, [64, 1024, 64]);  clone_61 = None
        expand_64: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_163, [8, 8, 64, 1024]);  permute_163 = None
        clone_62: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_368: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_62, [64, 64, 1024]);  clone_62 = None
        bmm_30: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_367, view_368)
        view_369: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_30, [8, 8, 1024, 1024]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_83: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_369, add_50);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_15: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_83, [-1], True)
        sub_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_83, amax_15);  add_83 = amax_15 = None
        exp_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_16: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_53: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 53)
        inductor_random_default_10: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_53, 'rand');  inductor_lookup_seed_default_53 = None
        gt_54: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_163: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_54, div_19)
        mul_164: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_163, 1.1111111111111112);  mul_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_65: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_164, [8, 8, 1024, 1024]);  mul_164 = None
        view_373: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_65, [64, 1024, 1024]);  expand_65 = None
        expand_66: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_162, [8, 8, 1024, 64]);  permute_162 = None
        clone_63: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_374: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_63, [64, 1024, 64]);  clone_63 = None
        bmm_31: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_373, view_374)
        view_375: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_31, [8, 8, 1024, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_164: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
        clone_64: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_376: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_64, [8, 1024, -1]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_165: "f32[512, 512]" = torch.ops.aten.permute.default(primals_116, [1, 0])
        view_377: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_376, [8192, 512]);  view_376 = None
        mm_83: "f32[8192, 512]" = torch.ops.aten.mm.default(view_377, permute_165);  permute_165 = None
        view_378: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_83, [8, 1024, 512]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_54: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 54)
        inductor_random_default_9: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_54, 'rand');  inductor_lookup_seed_default_54 = None
        gt_55: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_165: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_55, view_378);  view_378 = None
        mul_166: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_165, 1.1111111111111112);  mul_165 = None
        add_84: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_81, mul_166);  mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_28: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 2)
        mean_27: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_85: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_167: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_168: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_117, mul_167);  mul_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_166: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_118, [1, 0])
        view_379: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_168, [8192, 512]);  mul_168 = None
        mm_84: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_379, permute_166);  permute_166 = None
        view_380: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_84, [8, 1024, 2048]);  mm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_10: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_380);  view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_55: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 55)
        inductor_random_default_8: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_55, 'rand');  inductor_lookup_seed_default_55 = None
        gt_56: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_169: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_56, relu_10)
        mul_170: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_169, 1.1111111111111112);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_167: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_119, [1, 0])
        view_381: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_170, [8192, 2048]);  mul_170 = None
        mm_85: "f32[8192, 512]" = torch.ops.aten.mm.default(view_381, permute_167);  permute_167 = None
        view_382: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_85, [8, 1024, 512]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_56: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56)
        inductor_random_default_7: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_56, 'rand');  inductor_lookup_seed_default_56 = None
        gt_57: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_171: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_57, view_382);  view_382 = None
        mul_172: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_171, 1.1111111111111112);  mul_171 = None
        add_86: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_84, mul_172);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_29: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_86, 2)
        mean_28: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_87: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_173: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_86, rsqrt_28)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_174: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_120, mul_173);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_168: "f32[512, 512]" = torch.ops.aten.permute.default(primals_121, [1, 0])
        view_383: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_174, [8192, 512]);  mul_174 = None
        mm_86: "f32[8192, 512]" = torch.ops.aten.mm.default(view_383, permute_168);  permute_168 = None
        view_384: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_86, [8, 1024, 512]);  mm_86 = None
        view_385: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_384, [8, 1024, -1, 64]);  view_384 = None
        permute_169: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_385, [0, 2, 1, 3]);  view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_170: "f32[512, 512]" = torch.ops.aten.permute.default(primals_122, [1, 0])
        mm_87: "f32[8192, 512]" = torch.ops.aten.mm.default(view_383, permute_170);  permute_170 = None
        view_387: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_87, [8, 1024, 512]);  mm_87 = None
        view_388: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_387, [8, 1024, -1, 64]);  view_387 = None
        permute_171: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_172: "f32[512, 512]" = torch.ops.aten.permute.default(primals_123, [1, 0])
        mm_88: "f32[8192, 512]" = torch.ops.aten.mm.default(view_383, permute_172);  permute_172 = None
        view_390: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_88, [8, 1024, 512]);  mm_88 = None
        view_391: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_390, [8, 1024, -1, 64]);  view_390 = None
        permute_173: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_174: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_171, [0, 1, 3, 2]);  permute_171 = None
        expand_67: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_169, [8, 8, 1024, 64]);  permute_169 = None
        clone_65: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_392: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_65, [64, 1024, 64]);  clone_65 = None
        expand_68: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_174, [8, 8, 64, 1024]);  permute_174 = None
        clone_66: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_393: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_66, [64, 64, 1024]);  clone_66 = None
        bmm_32: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_392, view_393)
        view_394: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_32, [8, 8, 1024, 1024]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_88: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_394, add_46);  view_394 = add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_16: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_88, [-1], True)
        sub_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_88, amax_16);  add_88 = amax_16 = None
        exp_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_17: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_57: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 57)
        inductor_random_default_6: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_57, 'rand');  inductor_lookup_seed_default_57 = None
        gt_58: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_175: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_58, div_20)
        mul_176: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_175, 1.1111111111111112);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_69: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_176, [8, 8, 1024, 1024]);  mul_176 = None
        view_398: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_69, [64, 1024, 1024]);  expand_69 = None
        expand_70: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_173, [8, 8, 1024, 64]);  permute_173 = None
        clone_67: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_399: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_67, [64, 1024, 64]);  clone_67 = None
        bmm_33: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_398, view_399)
        view_400: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_33, [8, 8, 1024, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_175: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        clone_68: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_401: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_68, [8, 1024, -1]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_176: "f32[512, 512]" = torch.ops.aten.permute.default(primals_124, [1, 0])
        view_402: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_401, [8192, 512]);  view_401 = None
        mm_89: "f32[8192, 512]" = torch.ops.aten.mm.default(view_402, permute_176);  permute_176 = None
        view_403: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_89, [8, 1024, 512]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_58: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 58)
        inductor_random_default_5: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_58, 'rand');  inductor_lookup_seed_default_58 = None
        gt_59: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_177: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_59, view_403);  view_403 = None
        mul_178: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_177, 1.1111111111111112);  mul_177 = None
        add_89: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_86, mul_178);  mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_30: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_89, 2)
        mean_29: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_90: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_179: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_89, rsqrt_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_180: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_125, mul_179);  mul_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_177: "f32[512, 512]" = torch.ops.aten.permute.default(primals_126, [1, 0])
        view_404: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_180, [8192, 512]);  mul_180 = None
        mm_90: "f32[8192, 512]" = torch.ops.aten.mm.default(view_404, permute_177);  permute_177 = None
        view_405: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_90, [8, 1024, 512]);  mm_90 = None
        view_406: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_405, [8, 1024, -1, 64]);  view_405 = None
        permute_178: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_179: "f32[512, 512]" = torch.ops.aten.permute.default(primals_127, [1, 0])
        view_407: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_91: "f32[8192, 512]" = torch.ops.aten.mm.default(view_407, permute_179);  view_407 = permute_179 = None
        view_408: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_91, [8, 1024, 512]);  mm_91 = None
        view_409: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_408, [8, 1024, -1, 64]);  view_408 = None
        permute_180: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_181: "f32[512, 512]" = torch.ops.aten.permute.default(primals_128, [1, 0])
        view_410: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_92: "f32[8192, 512]" = torch.ops.aten.mm.default(view_410, permute_181);  view_410 = permute_181 = None
        view_411: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_92, [8, 1024, 512]);  mm_92 = None
        view_412: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_411, [8, 1024, -1, 64]);  view_411 = None
        permute_182: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_183: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2]);  permute_180 = None
        expand_71: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_178, [8, 8, 1024, 64]);  permute_178 = None
        clone_69: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_413: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_69, [64, 1024, 64]);  clone_69 = None
        expand_72: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_183, [8, 8, 64, 1024]);  permute_183 = None
        clone_70: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_414: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_70, [64, 64, 1024]);  clone_70 = None
        bmm_34: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_413, view_414)
        view_415: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_34, [8, 8, 1024, 1024]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_91: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_415, add_50);  view_415 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_17: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(add_91, [-1], True)
        sub_19: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_91, amax_17);  add_91 = amax_17 = None
        exp_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_18: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_59: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 59)
        inductor_random_default_4: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_59, 'rand');  inductor_lookup_seed_default_59 = None
        gt_60: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_181: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_60, div_21)
        mul_182: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_181, 1.1111111111111112);  mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_73: "f32[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_182, [8, 8, 1024, 1024]);  mul_182 = None
        view_419: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(expand_73, [64, 1024, 1024]);  expand_73 = None
        expand_74: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_182, [8, 8, 1024, 64]);  permute_182 = None
        clone_71: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_420: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_71, [64, 1024, 64]);  clone_71 = None
        bmm_35: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_419, view_420)
        view_421: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_35, [8, 8, 1024, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_184: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
        clone_72: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_422: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_72, [8, 1024, -1]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_185: "f32[512, 512]" = torch.ops.aten.permute.default(primals_129, [1, 0])
        view_423: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_422, [8192, 512]);  view_422 = None
        mm_93: "f32[8192, 512]" = torch.ops.aten.mm.default(view_423, permute_185);  permute_185 = None
        view_424: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_93, [8, 1024, 512]);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_60: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 60)
        inductor_random_default_3: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_60, 'rand');  inductor_lookup_seed_default_60 = None
        gt_61: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_183: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_61, view_424);  view_424 = None
        mul_184: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_183, 1.1111111111111112);  mul_183 = None
        add_92: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_89, mul_184);  mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_31: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_92, 2)
        mean_30: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_93: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_185: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_92, rsqrt_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_186: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_130, mul_185);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_186: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_131, [1, 0])
        view_425: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_186, [8192, 512]);  mul_186 = None
        mm_94: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_425, permute_186);  permute_186 = None
        view_426: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_94, [8, 1024, 2048]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_11: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_426);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_61: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 61)
        inductor_random_default_2: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_61, 'rand');  inductor_lookup_seed_default_61 = None
        gt_62: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_187: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_62, relu_11)
        mul_188: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_187, 1.1111111111111112);  mul_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_187: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_132, [1, 0])
        view_427: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_188, [8192, 2048]);  mul_188 = None
        mm_95: "f32[8192, 512]" = torch.ops.aten.mm.default(view_427, permute_187);  permute_187 = None
        view_428: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_95, [8, 1024, 512]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_62: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 62)
        inductor_random_default_1: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_62, 'rand');  inductor_lookup_seed_default_62 = None
        gt_63: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_189: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_63, view_428);  view_428 = None
        mul_190: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_189, 1.1111111111111112);  mul_189 = None
        add_94: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_92, mul_190);  mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_32: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_94, 2)
        mean_31: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_95: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_191: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_94, rsqrt_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_192: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_133, mul_191);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_63: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 63);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_63, 'rand');  inductor_lookup_seed_default_63 = None
        gt_64: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_193: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_64, mul_192);  mul_192 = None
        mul_194: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_193, 1.1111111111111112);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_195: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_194, 0.04419417382415922);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        permute_188: "f32[512, 32128]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        view_429: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_195, [8192, 512]);  mul_195 = None
        mm_96: "f32[8192, 32128]" = torch.ops.aten.mm.default(view_429, permute_188);  permute_188 = None
        view_430: "f32[8, 1024, 32128]" = torch.ops.aten.reshape.default(mm_96, [8, 1024, 32128]);  mm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        view_431: "f32[8192, 32128]" = torch.ops.aten.reshape.default(view_430, [-1, 32128])
        view_432: "i64[8192]" = torch.ops.aten.reshape.default(primals_53, [-1])
        amax_18: "f32[8192, 1]" = torch.ops.aten.amax.default(view_431, [1], True)
        sub_20: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(view_431, amax_18);  view_431 = None
        exp_18: "f32[8192, 32128]" = torch.ops.aten.exp.default(sub_20)
        sum_19: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True);  exp_18 = None
        log_2: "f32[8192, 1]" = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_21: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(sub_20, log_2);  sub_20 = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(view_432, -100)
        where_6: "i64[8192]" = torch.ops.aten.where.self(ne, view_432, full_default_4);  view_432 = full_default_4 = None
        unsqueeze_18: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_21, 1, unsqueeze_18);  sub_21 = unsqueeze_18 = None
        squeeze: "f32[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: "f32[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_7: "f32[8192]" = torch.ops.aten.where.self(ne, neg_1, full_default);  neg_1 = full_default = None
        sum_20: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_5: "f32[]" = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None
        sum_21: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
        div_22: "f32[]" = torch.ops.aten.div.Tensor(sum_21, convert_element_type_5);  sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_1: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_206: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None
        permute_207: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_420, [0, 2, 1]);  view_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_208: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_413, [0, 2, 1]);  view_413 = None
        permute_209: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_414, [0, 2, 1]);  view_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_231: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None
        permute_232: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_399, [0, 2, 1]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_233: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_392, [0, 2, 1]);  view_392 = None
        permute_234: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_393, [0, 2, 1]);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_2: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_264: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_373, [0, 2, 1]);  view_373 = None
        permute_265: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_374, [0, 2, 1]);  view_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_266: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_367, [0, 2, 1]);  view_367 = None
        permute_267: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1]);  view_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_289: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
        permute_290: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_353, [0, 2, 1]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_291: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_346, [0, 2, 1]);  view_346 = None
        permute_292: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1]);  view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_3: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_322: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_327, [0, 2, 1]);  view_327 = None
        permute_323: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_328, [0, 2, 1]);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_324: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None
        permute_325: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_347: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_306, [0, 2, 1]);  view_306 = None
        permute_348: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_349: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_300, [0, 2, 1]);  view_300 = None
        permute_350: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1]);  view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_4: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_380: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_281, [0, 2, 1]);  view_281 = None
        permute_381: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_282, [0, 2, 1]);  view_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_382: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
        permute_383: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_405: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
        permute_406: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_261, [0, 2, 1]);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_407: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_408: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_5: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_438: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        permute_439: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_440: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_441: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_463: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
        permute_464: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_215, [0, 2, 1]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_465: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
        permute_466: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_6: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_496: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
        permute_497: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_498: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        permute_499: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_521: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        permute_522: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_524: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_162, [0, 2, 1]);  view_162 = None
        permute_525: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_7: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_555: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_556: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_557: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        permute_558: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_8: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_588: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
        permute_589: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_590: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        permute_591: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_9: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_621: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_622: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_623: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
        permute_624: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_10: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_654: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_66, [0, 2, 1]);  view_66 = None
        permute_655: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_656: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        permute_657: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_11: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_687: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
        permute_688: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_689: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_690: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_12: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_720: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        permute_721: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_723: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_724: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        return (div_22, view_430, mul_79, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, embedding, ge, gt, rsqrt, view_1, bmm, add_6, embedding_1, amax, sum_1, gt_2, view_20, gt_3, add_9, rsqrt_1, view_22, gt_4, view_24, gt_5, add_11, rsqrt_2, view_26, div_3, gt_6, view_45, gt_7, add_14, rsqrt_3, view_47, gt_8, view_49, gt_9, add_16, rsqrt_4, view_51, div_4, gt_10, view_70, gt_11, add_19, rsqrt_5, view_72, gt_12, view_74, gt_13, add_21, rsqrt_6, view_76, div_5, gt_14, view_95, gt_15, add_24, rsqrt_7, view_97, gt_16, view_99, gt_17, add_26, rsqrt_8, view_101, div_6, gt_18, view_120, gt_19, add_29, rsqrt_9, view_122, gt_20, view_124, gt_21, add_31, rsqrt_10, view_126, div_7, gt_22, view_145, gt_23, add_34, rsqrt_11, view_147, gt_24, view_149, gt_25, add_36, rsqrt_12, gt_26, mul_79, where_2, embedding_2, gt_27, rsqrt_13, view_153, add_45, div_10, gt_28, view_172, gt_29, add_48, rsqrt_14, view_174, div_11, gt_30, view_193, gt_31, add_52, rsqrt_15, view_195, gt_32, view_197, gt_33, add_54, rsqrt_16, view_199, div_12, gt_34, view_218, gt_35, add_57, rsqrt_17, view_220, div_13, gt_36, view_239, gt_37, add_60, rsqrt_18, view_241, gt_38, view_243, gt_39, add_62, rsqrt_19, view_245, div_14, gt_40, view_264, gt_41, add_65, rsqrt_20, view_266, div_15, gt_42, view_285, gt_43, add_68, rsqrt_21, view_287, gt_44, view_289, gt_45, add_70, rsqrt_22, view_291, div_16, gt_46, view_310, gt_47, add_73, rsqrt_23, view_312, div_17, gt_48, view_331, gt_49, add_76, rsqrt_24, view_333, gt_50, view_335, gt_51, add_78, rsqrt_25, view_337, div_18, gt_52, view_356, gt_53, add_81, rsqrt_26, view_358, div_19, gt_54, view_377, gt_55, add_84, rsqrt_27, view_379, gt_56, view_381, gt_57, add_86, rsqrt_28, view_383, div_20, gt_58, view_402, gt_59, add_89, rsqrt_29, view_404, div_21, gt_60, view_423, gt_61, add_92, rsqrt_30, view_425, gt_62, view_427, gt_63, add_94, rsqrt_31, gt_64, view_429, view_430, amax_18, log_2, convert_element_type_5, le_1, permute_206, permute_207, permute_208, permute_209, permute_231, permute_232, permute_233, permute_234, le_2, permute_264, permute_265, permute_266, permute_267, permute_289, permute_290, permute_291, permute_292, le_3, permute_322, permute_323, permute_324, permute_325, permute_347, permute_348, permute_349, permute_350, le_4, permute_380, permute_381, permute_382, permute_383, permute_405, permute_406, permute_407, permute_408, le_5, permute_438, permute_439, permute_440, permute_441, permute_463, permute_464, permute_465, permute_466, le_6, permute_496, permute_497, permute_498, permute_499, permute_521, permute_522, permute_524, permute_525, le_7, permute_555, permute_556, permute_557, permute_558, le_8, permute_588, permute_589, permute_590, permute_591, le_9, permute_621, permute_622, permute_623, permute_624, le_10, permute_654, permute_655, permute_656, permute_657, le_11, permute_687, permute_688, permute_689, permute_690, le_12, permute_720, permute_721, permute_723, permute_724)
