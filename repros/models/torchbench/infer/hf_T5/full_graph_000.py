class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 2048]", arg1_1: "f16[32128, 512]", arg2_1: "f16[512]", arg3_1: "f16[512, 512]", arg4_1: "f16[512, 512]", arg5_1: "f16[512, 512]", arg6_1: "f16[32, 8]", arg7_1: "f16[512, 512]", arg8_1: "f16[512]", arg9_1: "f16[2048, 512]", arg10_1: "f16[512, 2048]", arg11_1: "f16[512]", arg12_1: "f16[512, 512]", arg13_1: "f16[512, 512]", arg14_1: "f16[512, 512]", arg15_1: "f16[512, 512]", arg16_1: "f16[512]", arg17_1: "f16[2048, 512]", arg18_1: "f16[512, 2048]", arg19_1: "f16[512]", arg20_1: "f16[512, 512]", arg21_1: "f16[512, 512]", arg22_1: "f16[512, 512]", arg23_1: "f16[512, 512]", arg24_1: "f16[512]", arg25_1: "f16[2048, 512]", arg26_1: "f16[512, 2048]", arg27_1: "f16[512]", arg28_1: "f16[512, 512]", arg29_1: "f16[512, 512]", arg30_1: "f16[512, 512]", arg31_1: "f16[512, 512]", arg32_1: "f16[512]", arg33_1: "f16[2048, 512]", arg34_1: "f16[512, 2048]", arg35_1: "f16[512]", arg36_1: "f16[512, 512]", arg37_1: "f16[512, 512]", arg38_1: "f16[512, 512]", arg39_1: "f16[512, 512]", arg40_1: "f16[512]", arg41_1: "f16[2048, 512]", arg42_1: "f16[512, 2048]", arg43_1: "f16[512]", arg44_1: "f16[512, 512]", arg45_1: "f16[512, 512]", arg46_1: "f16[512, 512]", arg47_1: "f16[512, 512]", arg48_1: "f16[512]", arg49_1: "f16[2048, 512]", arg50_1: "f16[512, 2048]", arg51_1: "f16[512]", arg52_1: "f16[512]", arg53_1: "f16[512, 512]", arg54_1: "f16[512, 512]", arg55_1: "f16[512, 512]", arg56_1: "f16[32, 8]", arg57_1: "f16[512, 512]", arg58_1: "f16[512]", arg59_1: "f16[512, 512]", arg60_1: "f16[512, 512]", arg61_1: "f16[512, 512]", arg62_1: "f16[512, 512]", arg63_1: "f16[512]", arg64_1: "f16[2048, 512]", arg65_1: "f16[512, 2048]", arg66_1: "f16[512]", arg67_1: "f16[512, 512]", arg68_1: "f16[512, 512]", arg69_1: "f16[512, 512]", arg70_1: "f16[512, 512]", arg71_1: "f16[512]", arg72_1: "f16[512, 512]", arg73_1: "f16[512, 512]", arg74_1: "f16[512, 512]", arg75_1: "f16[512, 512]", arg76_1: "f16[512]", arg77_1: "f16[2048, 512]", arg78_1: "f16[512, 2048]", arg79_1: "f16[512]", arg80_1: "f16[512, 512]", arg81_1: "f16[512, 512]", arg82_1: "f16[512, 512]", arg83_1: "f16[512, 512]", arg84_1: "f16[512]", arg85_1: "f16[512, 512]", arg86_1: "f16[512, 512]", arg87_1: "f16[512, 512]", arg88_1: "f16[512, 512]", arg89_1: "f16[512]", arg90_1: "f16[2048, 512]", arg91_1: "f16[512, 2048]", arg92_1: "f16[512]", arg93_1: "f16[512, 512]", arg94_1: "f16[512, 512]", arg95_1: "f16[512, 512]", arg96_1: "f16[512, 512]", arg97_1: "f16[512]", arg98_1: "f16[512, 512]", arg99_1: "f16[512, 512]", arg100_1: "f16[512, 512]", arg101_1: "f16[512, 512]", arg102_1: "f16[512]", arg103_1: "f16[2048, 512]", arg104_1: "f16[512, 2048]", arg105_1: "f16[512]", arg106_1: "f16[512, 512]", arg107_1: "f16[512, 512]", arg108_1: "f16[512, 512]", arg109_1: "f16[512, 512]", arg110_1: "f16[512]", arg111_1: "f16[512, 512]", arg112_1: "f16[512, 512]", arg113_1: "f16[512, 512]", arg114_1: "f16[512, 512]", arg115_1: "f16[512]", arg116_1: "f16[2048, 512]", arg117_1: "f16[512, 2048]", arg118_1: "f16[512]", arg119_1: "f16[512, 512]", arg120_1: "f16[512, 512]", arg121_1: "f16[512, 512]", arg122_1: "f16[512, 512]", arg123_1: "f16[512]", arg124_1: "f16[512, 512]", arg125_1: "f16[512, 512]", arg126_1: "f16[512, 512]", arg127_1: "f16[512, 512]", arg128_1: "f16[512]", arg129_1: "f16[2048, 512]", arg130_1: "f16[512, 2048]", arg131_1: "f16[512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        _tensor_constant0: "f16[]" = self._tensor_constant0;  _tensor_constant0 = None
        _tensor_constant1: "f16[]" = self._tensor_constant1;  _tensor_constant1 = None
        _tensor_constant2: "f16[]" = self._tensor_constant2;  _tensor_constant2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_2: "f16[1, 2048, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_161: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(embedding_2, torch.float32)
        pow_14: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_161, 2);  convert_element_type_161 = None
        mean_13: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_28: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(embedding_2, rsqrt_13);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_162: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_28, torch.float16);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_29: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg52_1, convert_element_type_162);  arg52_1 = convert_element_type_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_146: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_29, [2048, 512])
        permute_67: "f16[512, 512]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_36: "f16[2048, 512]" = torch.ops.aten.mm.default(view_146, permute_67);  view_146 = permute_67 = None
        view_147: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_36, [1, 2048, 512]);  mm_36 = None
        view_148: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_147, [1, 2048, -1, 64]);  view_147 = None
        permute_68: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_27: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_68, [1, 8, 2048, 64]);  permute_68 = None
        view_155: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_27, [8, 2048, 64]);  expand_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_149: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_29, [2048, 512])
        permute_69: "f16[512, 512]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_37: "f16[2048, 512]" = torch.ops.aten.mm.default(view_149, permute_69);  view_149 = permute_69 = None
        view_150: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_37, [1, 2048, 512]);  mm_37 = None
        view_151: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_150, [1, 2048, -1, 64]);  view_150 = None
        permute_70: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_73: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        expand_28: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_73, [1, 8, 64, 2048]);  permute_73 = None
        view_156: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_28, [8, 64, 2048]);  expand_28 = None
        bmm_12: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_155, view_156);  view_155 = view_156 = None
        view_157: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_12, [1, 8, 2048, 2048]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:228 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_15: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_16: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(iota_15, 0);  iota_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:227 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_14: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_15: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(iota_14, 1);  iota_14 = None
        add_43: "i64[2048, 1]" = torch.ops.aten.add.Tensor(unsqueeze_15, 0);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:229 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub_7: "i64[2048, 2048]" = torch.ops.aten.sub.Tensor(unsqueeze_16, add_43);  unsqueeze_16 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:203 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_default_31: "i64[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "i64[2048, 2048]" = torch.ops.aten.minimum.default(sub_7, full_default_31);  sub_7 = full_default_31 = None
        neg_12: "i64[2048, 2048]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_1: "b8[2048, 2048]" = torch.ops.aten.lt.Scalar(neg_12, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_171: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(neg_12, torch.float32)
        div_8: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_171, 16);  convert_element_type_171 = None
        log_1: "f32[2048, 2048]" = torch.ops.aten.log.default(div_8);  div_8 = None
        div_9: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_30: "f32[2048, 2048]" = torch.ops.aten.mul.Tensor(div_9, 16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_172: "i64[2048, 2048]" = torch.ops.prims.convert_element_type.default(mul_30, torch.int64);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_44: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_172, 16);  convert_element_type_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_32: "i64[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_2: "i64[2048, 2048]" = torch.ops.aten.minimum.default(add_44, full_default_32);  add_44 = full_default_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_16: "i64[2048, 2048]" = torch.ops.aten.where.self(lt_1, neg_12, minimum_2);  lt_1 = neg_12 = minimum_2 = None
        add_45: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(where_16, 0);  where_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_3: "f16[2048, 2048, 8]" = torch.ops.aten.embedding.default(arg56_1, add_45);  arg56_1 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_74: "f16[8, 2048, 2048]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_17: "f16[1, 8, 2048, 2048]" = torch.ops.aten.unsqueeze.default(permute_74, 0);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_9: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_39: "i64[2048]" = torch.ops.aten.add.Tensor(iota_9, 0);  iota_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(add_39, 0);  add_39 = None
        unsqueeze_10: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        unsqueeze_11: "i64[1, 1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_8: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_38: "i64[2048]" = torch.ops.aten.add.Tensor(iota_8, 0);  iota_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_6: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(add_38, 0);  add_38 = None
        unsqueeze_7: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 2048, 2048]" = torch.ops.aten.le.Tensor(unsqueeze_11, unsqueeze_8);  unsqueeze_11 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_25: "b8[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(le, [1, -1, 2048, 2048]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_27: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f16[1, 1, 2048, 2048]" = torch.ops.aten.where.self(expand_25, full_default_27, full_default_28);  expand_25 = full_default_27 = full_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_46: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(unsqueeze_17, where_14);  unsqueeze_17 = where_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_47: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_157, add_46);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_173: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_47, torch.float32);  add_47 = None
        amax_6: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_173, [-1], True)
        sub_8: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_173, amax_6);  convert_element_type_173 = amax_6 = None
        exp_6: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_7: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_174: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_10, torch.float16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_29: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_174, [1, 8, 2048, 2048]);  convert_element_type_174 = None
        view_160: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_29, [8, 2048, 2048]);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_152: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_29, [2048, 512]);  mul_29 = None
        permute_71: "f16[512, 512]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_38: "f16[2048, 512]" = torch.ops.aten.mm.default(view_152, permute_71);  view_152 = permute_71 = None
        view_153: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_38, [1, 2048, 512]);  mm_38 = None
        view_154: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_153, [1, 2048, -1, 64]);  view_153 = None
        permute_72: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_30: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_72, [1, 8, 2048, 64]);  permute_72 = None
        view_161: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_30, [8, 2048, 64]);  expand_30 = None
        bmm_13: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_160, view_161);  view_160 = view_161 = None
        view_162: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 8, 2048, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_75: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        clone_34: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_163: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_34, [1, 2048, -1]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_164: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_163, [2048, 512]);  view_163 = None
        permute_76: "f16[512, 512]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        mm_39: "f16[2048, 512]" = torch.ops.aten.mm.default(view_164, permute_76);  view_164 = permute_76 = None
        view_165: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_39, [1, 2048, 512]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_48: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(embedding_2, view_165);  embedding_2 = view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_179: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_12: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_48);  add_48 = None
        any_13: "b8[]" = torch.ops.aten.any.default(isinf_12);  isinf_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_34: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "f32[]" = torch.ops.aten.where.self(any_13, full_default_34, full_default_33);  any_13 = full_default_34 = full_default_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_13: "f32[]" = torch.ops.aten.neg.default(where_17)
        clamp_min_12: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_179, neg_13);  convert_element_type_179 = neg_13 = None
        clamp_max_12: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_12, where_17);  clamp_min_12 = where_17 = None
        convert_element_type_180: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_12, torch.float16);  clamp_max_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_181: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_180, torch.float32)
        pow_15: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_181, 2);  convert_element_type_181 = None
        mean_14: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_31: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_180, rsqrt_14);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_182: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_31, torch.float16);  mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_32: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg58_1, convert_element_type_182);  arg58_1 = convert_element_type_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_166: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_32, [2048, 512]);  mul_32 = None
        permute_77: "f16[512, 512]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_40: "f16[2048, 512]" = torch.ops.aten.mm.default(view_166, permute_77);  view_166 = permute_77 = None
        view_167: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_40, [1, 2048, 512]);  mm_40 = None
        view_168: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_167, [1, 2048, -1, 64]);  view_167 = None
        permute_78: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_31: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_78, [1, 8, 2048, 64]);  permute_78 = None
        view_175: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_31, [8, 2048, 64]);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "f16[1, 2048, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        pow_1: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2);  convert_element_type = None
        mean: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_2: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(embedding, rsqrt);  rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_1: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.float16);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg2_1, convert_element_type_1);  arg2_1 = convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_1: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_1, [2048, 512])
        permute: "f16[512, 512]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm: "f16[2048, 512]" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
        view_2: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm, [1, 2048, 512]);  mm = None
        view_3: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_2, [1, 2048, -1, 64]);  view_2 = None
        permute_1: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_1: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_1, [1, 8, 2048, 64]);  permute_1 = None
        view_10: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_1, [8, 2048, 64]);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_4: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_1, [2048, 512])
        permute_2: "f16[512, 512]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "f16[2048, 512]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_1, [1, 2048, 512]);  mm_1 = None
        view_6: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_5, [1, 2048, -1, 64]);  view_5 = None
        permute_3: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_6: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_2: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_6, [1, 8, 64, 2048]);  permute_6 = None
        view_11: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_2, [8, 64, 2048]);  expand_2 = None
        bmm: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
        view_12: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm, [1, 8, 2048, 2048]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:228 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_5: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_4: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(iota_5, 0);  iota_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:227 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_4: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(iota_4, 1);  iota_4 = None
        add_3: "i64[2048, 1]" = torch.ops.aten.add.Tensor(unsqueeze_3, 0);  unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:229 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub: "i64[2048, 2048]" = torch.ops.aten.sub.Tensor(unsqueeze_4, add_3);  unsqueeze_4 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:200 in _relative_position_bucket, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        gt: "b8[2048, 2048]" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type_10: "i64[2048, 2048]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
        mul_2: "i64[2048, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 16);  convert_element_type_10 = None
        add_4: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(mul_2, 0);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:201 in _relative_position_bucket, code: relative_position = torch.abs(relative_position)
        abs_1: "i64[2048, 2048]" = torch.ops.aten.abs.default(sub);  sub = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt: "b8[2048, 2048]" = torch.ops.aten.lt.Scalar(abs_1, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_11: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_11, 8);  convert_element_type_11 = None
        log: "f32[2048, 2048]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_3: "f32[2048, 2048]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_12: "i64[2048, 2048]" = torch.ops.prims.convert_element_type.default(mul_3, torch.int64);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_5: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_12, 8);  convert_element_type_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_2: "i64[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum: "i64[2048, 2048]" = torch.ops.aten.minimum.default(add_5, full_default_2);  add_5 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_1: "i64[2048, 2048]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_6: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(add_4, where_1);  add_4 = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_1: "f16[2048, 2048, 8]" = torch.ops.aten.embedding.default(arg6_1, add_6);  arg6_1 = add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "f16[8, 2048, 2048]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_5: "f16[1, 8, 2048, 2048]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[2048]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 2048, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(ge, [1, -1, 2048, 2048]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[1, 1, 2048, 2048]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_7: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(unsqueeze_5, where);  unsqueeze_5 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_8: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_13: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        amax: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_13, [-1], True)
        sub_1: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax);  convert_element_type_13 = amax = None
        exp: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_14: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_2, torch.float16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_3: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_14, [1, 8, 2048, 2048]);  convert_element_type_14 = None
        view_15: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_3, [8, 2048, 2048]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_7: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_1, [2048, 512]);  mul_1 = None
        permute_4: "f16[512, 512]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_2: "f16[2048, 512]" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
        view_8: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_2, [1, 2048, 512]);  mm_2 = None
        view_9: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_8, [1, 2048, -1, 64]);  view_8 = None
        permute_5: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_4: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_5, [1, 8, 2048, 64]);  permute_5 = None
        view_16: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_4, [8, 2048, 64]);  expand_4 = None
        bmm_1: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
        view_17: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 8, 2048, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        clone_2: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_18: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_2, [1, 2048, -1]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_19: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_18, [2048, 512]);  view_18 = None
        permute_9: "f16[512, 512]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_3: "f16[2048, 512]" = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
        view_20: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_3, [1, 2048, 512]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_9: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(embedding, view_20);  embedding = view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_19: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_9);  add_9 = None
        any_1: "b8[]" = torch.ops.aten.any.default(isinf);  isinf = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[]" = torch.ops.aten.where.self(any_1, full_default_4, full_default_3);  any_1 = full_default_4 = full_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg: "f32[]" = torch.ops.aten.neg.default(where_2)
        clamp_min: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_19, neg);  convert_element_type_19 = neg = None
        clamp_max: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min, where_2);  clamp_min = where_2 = None
        convert_element_type_20: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.float16);  clamp_max = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_21: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32)
        pow_2: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2);  convert_element_type_21 = None
        mean_1: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_4: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_20, rsqrt_1);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_22: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_4, torch.float16);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_5: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg8_1, convert_element_type_22);  arg8_1 = convert_element_type_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_21: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_5, [2048, 512]);  mul_5 = None
        permute_10: "f16[512, 2048]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_4: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_21, permute_10);  view_21 = permute_10 = None
        view_22: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_4, [1, 2048, 2048]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_22);  view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_23: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu, [2048, 2048]);  relu = None
        permute_11: "f16[2048, 512]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_5: "f16[2048, 512]" = torch.ops.aten.mm.default(view_23, permute_11);  view_23 = permute_11 = None
        view_24: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_5, [1, 2048, 512]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_11: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_20, view_24);  convert_element_type_20 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_27: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_1: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_11);  add_11 = None
        any_2: "b8[]" = torch.ops.aten.any.default(isinf_1);  isinf_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_6: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[]" = torch.ops.aten.where.self(any_2, full_default_6, full_default_5);  any_2 = full_default_6 = full_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_1: "f32[]" = torch.ops.aten.neg.default(where_3)
        clamp_min_1: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_27, neg_1);  convert_element_type_27 = neg_1 = None
        clamp_max_1: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_1, where_3);  clamp_min_1 = where_3 = None
        convert_element_type_28: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.float16);  clamp_max_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_29: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_28, torch.float32)
        pow_3: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_29, 2);  convert_element_type_29 = None
        mean_2: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_12: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_6: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_28, rsqrt_2);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_30: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_6, torch.float16);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_7: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg11_1, convert_element_type_30);  arg11_1 = convert_element_type_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_25: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_7, [2048, 512])
        permute_12: "f16[512, 512]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        mm_6: "f16[2048, 512]" = torch.ops.aten.mm.default(view_25, permute_12);  view_25 = permute_12 = None
        view_26: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_6, [1, 2048, 512]);  mm_6 = None
        view_27: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_26, [1, 2048, -1, 64]);  view_26 = None
        permute_13: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_5: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_13, [1, 8, 2048, 64]);  permute_13 = None
        view_34: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_5, [8, 2048, 64]);  expand_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_28: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_7, [2048, 512])
        permute_14: "f16[512, 512]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_7: "f16[2048, 512]" = torch.ops.aten.mm.default(view_28, permute_14);  view_28 = permute_14 = None
        view_29: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_7, [1, 2048, 512]);  mm_7 = None
        view_30: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_29, [1, 2048, -1, 64]);  view_29 = None
        permute_15: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_18: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_6: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_18, [1, 8, 64, 2048]);  permute_18 = None
        view_35: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_6, [8, 64, 2048]);  expand_6 = None
        bmm_2: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
        view_36: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_2, [1, 8, 2048, 2048]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_13: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_36, add_7);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_39: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        amax_1: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_39, [-1], True)
        sub_2: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_39, amax_1);  convert_element_type_39 = amax_1 = None
        exp_1: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_2: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_40: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_3, torch.float16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_7: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_40, [1, 8, 2048, 2048]);  convert_element_type_40 = None
        view_39: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_7, [8, 2048, 2048]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_31: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_7, [2048, 512]);  mul_7 = None
        permute_16: "f16[512, 512]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        mm_8: "f16[2048, 512]" = torch.ops.aten.mm.default(view_31, permute_16);  view_31 = permute_16 = None
        view_32: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_8, [1, 2048, 512]);  mm_8 = None
        view_33: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_32, [1, 2048, -1, 64]);  view_32 = None
        permute_17: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_8: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_17, [1, 8, 2048, 64]);  permute_17 = None
        view_40: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_8, [8, 2048, 64]);  expand_8 = None
        bmm_3: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_39, view_40);  view_39 = view_40 = None
        view_41: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 8, 2048, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_7: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_42: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_7, [1, 2048, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_43: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_42, [2048, 512]);  view_42 = None
        permute_20: "f16[512, 512]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_9: "f16[2048, 512]" = torch.ops.aten.mm.default(view_43, permute_20);  view_43 = permute_20 = None
        view_44: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_9, [1, 2048, 512]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_14: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_28, view_44);  convert_element_type_28 = view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_45: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_2: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_14);  add_14 = None
        any_3: "b8[]" = torch.ops.aten.any.default(isinf_2);  isinf_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_8: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[]" = torch.ops.aten.where.self(any_3, full_default_8, full_default_7);  any_3 = full_default_8 = full_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_2: "f32[]" = torch.ops.aten.neg.default(where_4)
        clamp_min_2: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_45, neg_2);  convert_element_type_45 = neg_2 = None
        clamp_max_2: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_2, where_4);  clamp_min_2 = where_4 = None
        convert_element_type_46: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.float16);  clamp_max_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_47: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_46, torch.float32)
        pow_4: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_47, 2);  convert_element_type_47 = None
        mean_3: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_15: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_8: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_46, rsqrt_3);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_48: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_8, torch.float16);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_9: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg16_1, convert_element_type_48);  arg16_1 = convert_element_type_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_45: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_9, [2048, 512]);  mul_9 = None
        permute_21: "f16[512, 2048]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        mm_10: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_45, permute_21);  view_45 = permute_21 = None
        view_46: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_10, [1, 2048, 2048]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_1: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_46);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_47: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu_1, [2048, 2048]);  relu_1 = None
        permute_22: "f16[2048, 512]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_11: "f16[2048, 512]" = torch.ops.aten.mm.default(view_47, permute_22);  view_47 = permute_22 = None
        view_48: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_11, [1, 2048, 512]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_16: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_46, view_48);  convert_element_type_46 = view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_53: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_3: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_16);  add_16 = None
        any_4: "b8[]" = torch.ops.aten.any.default(isinf_3);  isinf_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_10: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[]" = torch.ops.aten.where.self(any_4, full_default_10, full_default_9);  any_4 = full_default_10 = full_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_3: "f32[]" = torch.ops.aten.neg.default(where_5)
        clamp_min_3: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_53, neg_3);  convert_element_type_53 = neg_3 = None
        clamp_max_3: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_3, where_5);  clamp_min_3 = where_5 = None
        convert_element_type_54: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.float16);  clamp_max_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_55: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_54, torch.float32)
        pow_5: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_55, 2);  convert_element_type_55 = None
        mean_4: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_10: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_54, rsqrt_4);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_56: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_10, torch.float16);  mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_11: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg19_1, convert_element_type_56);  arg19_1 = convert_element_type_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_49: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_11, [2048, 512])
        permute_23: "f16[512, 512]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_12: "f16[2048, 512]" = torch.ops.aten.mm.default(view_49, permute_23);  view_49 = permute_23 = None
        view_50: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_12, [1, 2048, 512]);  mm_12 = None
        view_51: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_50, [1, 2048, -1, 64]);  view_50 = None
        permute_24: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_9: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_24, [1, 8, 2048, 64]);  permute_24 = None
        view_58: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_9, [8, 2048, 64]);  expand_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_52: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_11, [2048, 512])
        permute_25: "f16[512, 512]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        mm_13: "f16[2048, 512]" = torch.ops.aten.mm.default(view_52, permute_25);  view_52 = permute_25 = None
        view_53: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_13, [1, 2048, 512]);  mm_13 = None
        view_54: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_53, [1, 2048, -1, 64]);  view_53 = None
        permute_26: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_29: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        expand_10: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_29, [1, 8, 64, 2048]);  permute_29 = None
        view_59: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_10, [8, 64, 2048]);  expand_10 = None
        bmm_4: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
        view_60: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_4, [1, 8, 2048, 2048]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_18: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_60, add_7);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_65: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_18, torch.float32);  add_18 = None
        amax_2: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_65, [-1], True)
        sub_3: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_65, amax_2);  convert_element_type_65 = amax_2 = None
        exp_2: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_3: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_66: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_4, torch.float16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_11: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_66, [1, 8, 2048, 2048]);  convert_element_type_66 = None
        view_63: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_11, [8, 2048, 2048]);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_55: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_11, [2048, 512]);  mul_11 = None
        permute_27: "f16[512, 512]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_14: "f16[2048, 512]" = torch.ops.aten.mm.default(view_55, permute_27);  view_55 = permute_27 = None
        view_56: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_14, [1, 2048, 512]);  mm_14 = None
        view_57: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_56, [1, 2048, -1, 64]);  view_56 = None
        permute_28: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_12: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_28, [1, 8, 2048, 64]);  permute_28 = None
        view_64: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_12, [8, 2048, 64]);  expand_12 = None
        bmm_5: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_63, view_64);  view_63 = view_64 = None
        view_65: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 8, 2048, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        clone_12: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_66: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_12, [1, 2048, -1]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_67: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_66, [2048, 512]);  view_66 = None
        permute_31: "f16[512, 512]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_15: "f16[2048, 512]" = torch.ops.aten.mm.default(view_67, permute_31);  view_67 = permute_31 = None
        view_68: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_15, [1, 2048, 512]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_19: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_54, view_68);  convert_element_type_54 = view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_71: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_19, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_4: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_19);  add_19 = None
        any_5: "b8[]" = torch.ops.aten.any.default(isinf_4);  isinf_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_12: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[]" = torch.ops.aten.where.self(any_5, full_default_12, full_default_11);  any_5 = full_default_12 = full_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_4: "f32[]" = torch.ops.aten.neg.default(where_6)
        clamp_min_4: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_71, neg_4);  convert_element_type_71 = neg_4 = None
        clamp_max_4: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_4, where_6);  clamp_min_4 = where_6 = None
        convert_element_type_72: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_4, torch.float16);  clamp_max_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_73: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_72, torch.float32)
        pow_6: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 2);  convert_element_type_73 = None
        mean_5: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_20: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_12: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_72, rsqrt_5);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_74: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_12, torch.float16);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_13: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg24_1, convert_element_type_74);  arg24_1 = convert_element_type_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_69: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_13, [2048, 512]);  mul_13 = None
        permute_32: "f16[512, 2048]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_16: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_69, permute_32);  view_69 = permute_32 = None
        view_70: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_16, [1, 2048, 2048]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_2: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_70);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_71: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu_2, [2048, 2048]);  relu_2 = None
        permute_33: "f16[2048, 512]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        mm_17: "f16[2048, 512]" = torch.ops.aten.mm.default(view_71, permute_33);  view_71 = permute_33 = None
        view_72: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_17, [1, 2048, 512]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_21: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_72, view_72);  convert_element_type_72 = view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_79: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_21, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_5: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_21);  add_21 = None
        any_6: "b8[]" = torch.ops.aten.any.default(isinf_5);  isinf_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_14: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f32[]" = torch.ops.aten.where.self(any_6, full_default_14, full_default_13);  any_6 = full_default_14 = full_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_5: "f32[]" = torch.ops.aten.neg.default(where_7)
        clamp_min_5: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_79, neg_5);  convert_element_type_79 = neg_5 = None
        clamp_max_5: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_5, where_7);  clamp_min_5 = where_7 = None
        convert_element_type_80: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_5, torch.float16);  clamp_max_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_81: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_80, torch.float32)
        pow_7: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_81, 2);  convert_element_type_81 = None
        mean_6: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_22: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_14: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_80, rsqrt_6);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_82: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_14, torch.float16);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_15: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg27_1, convert_element_type_82);  arg27_1 = convert_element_type_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_73: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_15, [2048, 512])
        permute_34: "f16[512, 512]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_18: "f16[2048, 512]" = torch.ops.aten.mm.default(view_73, permute_34);  view_73 = permute_34 = None
        view_74: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_18, [1, 2048, 512]);  mm_18 = None
        view_75: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_74, [1, 2048, -1, 64]);  view_74 = None
        permute_35: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_13: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_35, [1, 8, 2048, 64]);  permute_35 = None
        view_82: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_13, [8, 2048, 64]);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_76: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_15, [2048, 512])
        permute_36: "f16[512, 512]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_19: "f16[2048, 512]" = torch.ops.aten.mm.default(view_76, permute_36);  view_76 = permute_36 = None
        view_77: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_19, [1, 2048, 512]);  mm_19 = None
        view_78: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_77, [1, 2048, -1, 64]);  view_77 = None
        permute_37: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_40: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        expand_14: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_40, [1, 8, 64, 2048]);  permute_40 = None
        view_83: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_14, [8, 64, 2048]);  expand_14 = None
        bmm_6: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
        view_84: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_6, [1, 8, 2048, 2048]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_23: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_84, add_7);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_91: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        amax_3: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_91, [-1], True)
        sub_4: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_91, amax_3);  convert_element_type_91 = amax_3 = None
        exp_3: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_4: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_92: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_5, torch.float16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_15: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_92, [1, 8, 2048, 2048]);  convert_element_type_92 = None
        view_87: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_15, [8, 2048, 2048]);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_79: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_15, [2048, 512]);  mul_15 = None
        permute_38: "f16[512, 512]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_20: "f16[2048, 512]" = torch.ops.aten.mm.default(view_79, permute_38);  view_79 = permute_38 = None
        view_80: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_20, [1, 2048, 512]);  mm_20 = None
        view_81: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_80, [1, 2048, -1, 64]);  view_80 = None
        permute_39: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_16: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_39, [1, 8, 2048, 64]);  permute_39 = None
        view_88: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_16, [8, 2048, 64]);  expand_16 = None
        bmm_7: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_87, view_88);  view_87 = view_88 = None
        view_89: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 8, 2048, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_41: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        clone_17: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_90: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_17, [1, 2048, -1]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_91: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_90, [2048, 512]);  view_90 = None
        permute_42: "f16[512, 512]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_21: "f16[2048, 512]" = torch.ops.aten.mm.default(view_91, permute_42);  view_91 = permute_42 = None
        view_92: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_21, [1, 2048, 512]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_24: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_80, view_92);  convert_element_type_80 = view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_97: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_6: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_24);  add_24 = None
        any_7: "b8[]" = torch.ops.aten.any.default(isinf_6);  isinf_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_16: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[]" = torch.ops.aten.where.self(any_7, full_default_16, full_default_15);  any_7 = full_default_16 = full_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_6: "f32[]" = torch.ops.aten.neg.default(where_8)
        clamp_min_6: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_97, neg_6);  convert_element_type_97 = neg_6 = None
        clamp_max_6: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_6, where_8);  clamp_min_6 = where_8 = None
        convert_element_type_98: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_6, torch.float16);  clamp_max_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_99: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_98, torch.float32)
        pow_8: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_99, 2);  convert_element_type_99 = None
        mean_7: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_25: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_16: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_98, rsqrt_7);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_100: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_16, torch.float16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_17: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg32_1, convert_element_type_100);  arg32_1 = convert_element_type_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_93: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_17, [2048, 512]);  mul_17 = None
        permute_43: "f16[512, 2048]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_22: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_93, permute_43);  view_93 = permute_43 = None
        view_94: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_22, [1, 2048, 2048]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_3: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_94);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_95: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu_3, [2048, 2048]);  relu_3 = None
        permute_44: "f16[2048, 512]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_23: "f16[2048, 512]" = torch.ops.aten.mm.default(view_95, permute_44);  view_95 = permute_44 = None
        view_96: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_23, [1, 2048, 512]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_26: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_98, view_96);  convert_element_type_98 = view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_105: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_7: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_26);  add_26 = None
        any_8: "b8[]" = torch.ops.aten.any.default(isinf_7);  isinf_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_18: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f32[]" = torch.ops.aten.where.self(any_8, full_default_18, full_default_17);  any_8 = full_default_18 = full_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_7: "f32[]" = torch.ops.aten.neg.default(where_9)
        clamp_min_7: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_105, neg_7);  convert_element_type_105 = neg_7 = None
        clamp_max_7: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_7, where_9);  clamp_min_7 = where_9 = None
        convert_element_type_106: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_7, torch.float16);  clamp_max_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_107: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_106, torch.float32)
        pow_9: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_107, 2);  convert_element_type_107 = None
        mean_8: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_27: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_18: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_106, rsqrt_8);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_108: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_18, torch.float16);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_19: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg35_1, convert_element_type_108);  arg35_1 = convert_element_type_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_97: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_19, [2048, 512])
        permute_45: "f16[512, 512]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_24: "f16[2048, 512]" = torch.ops.aten.mm.default(view_97, permute_45);  view_97 = permute_45 = None
        view_98: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_24, [1, 2048, 512]);  mm_24 = None
        view_99: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_98, [1, 2048, -1, 64]);  view_98 = None
        permute_46: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_17: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_46, [1, 8, 2048, 64]);  permute_46 = None
        view_106: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_17, [8, 2048, 64]);  expand_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_100: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_19, [2048, 512])
        permute_47: "f16[512, 512]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_25: "f16[2048, 512]" = torch.ops.aten.mm.default(view_100, permute_47);  view_100 = permute_47 = None
        view_101: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_25, [1, 2048, 512]);  mm_25 = None
        view_102: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_101, [1, 2048, -1, 64]);  view_101 = None
        permute_48: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_51: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_18: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_51, [1, 8, 64, 2048]);  permute_51 = None
        view_107: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_18, [8, 64, 2048]);  expand_18 = None
        bmm_8: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_106, view_107);  view_106 = view_107 = None
        view_108: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_8, [1, 8, 2048, 2048]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_28: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_108, add_7);  view_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_117: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_28, torch.float32);  add_28 = None
        amax_4: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_117, [-1], True)
        sub_5: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_117, amax_4);  convert_element_type_117 = amax_4 = None
        exp_4: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_5: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_118: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_6, torch.float16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_19: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_118, [1, 8, 2048, 2048]);  convert_element_type_118 = None
        view_111: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_19, [8, 2048, 2048]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_103: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_19, [2048, 512]);  mul_19 = None
        permute_49: "f16[512, 512]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_26: "f16[2048, 512]" = torch.ops.aten.mm.default(view_103, permute_49);  view_103 = permute_49 = None
        view_104: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_26, [1, 2048, 512]);  mm_26 = None
        view_105: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_104, [1, 2048, -1, 64]);  view_104 = None
        permute_50: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_20: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_50, [1, 8, 2048, 64]);  permute_50 = None
        view_112: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_20, [8, 2048, 64]);  expand_20 = None
        bmm_9: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_111, view_112);  view_111 = view_112 = None
        view_113: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 8, 2048, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_52: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        clone_22: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_114: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_22, [1, 2048, -1]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_115: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_114, [2048, 512]);  view_114 = None
        permute_53: "f16[512, 512]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_27: "f16[2048, 512]" = torch.ops.aten.mm.default(view_115, permute_53);  view_115 = permute_53 = None
        view_116: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_27, [1, 2048, 512]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_29: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_106, view_116);  convert_element_type_106 = view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_123: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_8: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_29);  add_29 = None
        any_9: "b8[]" = torch.ops.aten.any.default(isinf_8);  isinf_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_20: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f32[]" = torch.ops.aten.where.self(any_9, full_default_20, full_default_19);  any_9 = full_default_20 = full_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_8: "f32[]" = torch.ops.aten.neg.default(where_10)
        clamp_min_8: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_123, neg_8);  convert_element_type_123 = neg_8 = None
        clamp_max_8: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_8, where_10);  clamp_min_8 = where_10 = None
        convert_element_type_124: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_8, torch.float16);  clamp_max_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_125: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_124, torch.float32)
        pow_10: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_125, 2);  convert_element_type_125 = None
        mean_9: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_30: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_20: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_124, rsqrt_9);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_126: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_20, torch.float16);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_21: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg40_1, convert_element_type_126);  arg40_1 = convert_element_type_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_117: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_21, [2048, 512]);  mul_21 = None
        permute_54: "f16[512, 2048]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_28: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_117, permute_54);  view_117 = permute_54 = None
        view_118: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_28, [1, 2048, 2048]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_4: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_118);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_119: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu_4, [2048, 2048]);  relu_4 = None
        permute_55: "f16[2048, 512]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_29: "f16[2048, 512]" = torch.ops.aten.mm.default(view_119, permute_55);  view_119 = permute_55 = None
        view_120: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_29, [1, 2048, 512]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_31: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_124, view_120);  convert_element_type_124 = view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_131: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_9: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_31);  add_31 = None
        any_10: "b8[]" = torch.ops.aten.any.default(isinf_9);  isinf_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_22: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f32[]" = torch.ops.aten.where.self(any_10, full_default_22, full_default_21);  any_10 = full_default_22 = full_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_9: "f32[]" = torch.ops.aten.neg.default(where_11)
        clamp_min_9: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_131, neg_9);  convert_element_type_131 = neg_9 = None
        clamp_max_9: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_9, where_11);  clamp_min_9 = where_11 = None
        convert_element_type_132: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_9, torch.float16);  clamp_max_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_133: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_132, torch.float32)
        pow_11: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_133, 2);  convert_element_type_133 = None
        mean_10: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_32: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_22: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_132, rsqrt_10);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_134: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_22, torch.float16);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_23: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg43_1, convert_element_type_134);  arg43_1 = convert_element_type_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_121: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_23, [2048, 512])
        permute_56: "f16[512, 512]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        mm_30: "f16[2048, 512]" = torch.ops.aten.mm.default(view_121, permute_56);  view_121 = permute_56 = None
        view_122: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_30, [1, 2048, 512]);  mm_30 = None
        view_123: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_122, [1, 2048, -1, 64]);  view_122 = None
        permute_57: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_21: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_57, [1, 8, 2048, 64]);  permute_57 = None
        view_130: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_21, [8, 2048, 64]);  expand_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_124: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_23, [2048, 512])
        permute_58: "f16[512, 512]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_31: "f16[2048, 512]" = torch.ops.aten.mm.default(view_124, permute_58);  view_124 = permute_58 = None
        view_125: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_31, [1, 2048, 512]);  mm_31 = None
        view_126: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_125, [1, 2048, -1, 64]);  view_125 = None
        permute_59: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_62: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        expand_22: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_62, [1, 8, 64, 2048]);  permute_62 = None
        view_131: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_22, [8, 64, 2048]);  expand_22 = None
        bmm_10: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_130, view_131);  view_130 = view_131 = None
        view_132: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_10, [1, 8, 2048, 2048]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_33: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_132, add_7);  view_132 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_143: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        amax_5: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_143, [-1], True)
        sub_6: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_143, amax_5);  convert_element_type_143 = amax_5 = None
        exp_5: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_6: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_144: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_7, torch.float16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_23: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_144, [1, 8, 2048, 2048]);  convert_element_type_144 = None
        view_135: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_23, [8, 2048, 2048]);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_127: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_23, [2048, 512]);  mul_23 = None
        permute_60: "f16[512, 512]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_32: "f16[2048, 512]" = torch.ops.aten.mm.default(view_127, permute_60);  view_127 = permute_60 = None
        view_128: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_32, [1, 2048, 512]);  mm_32 = None
        view_129: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_128, [1, 2048, -1, 64]);  view_128 = None
        permute_61: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_24: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_61, [1, 8, 2048, 64]);  permute_61 = None
        view_136: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_24, [8, 2048, 64]);  expand_24 = None
        bmm_11: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_135, view_136);  view_135 = view_136 = None
        view_137: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 8, 2048, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        clone_27: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_138: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_27, [1, 2048, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_139: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_138, [2048, 512]);  view_138 = None
        permute_64: "f16[512, 512]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_33: "f16[2048, 512]" = torch.ops.aten.mm.default(view_139, permute_64);  view_139 = permute_64 = None
        view_140: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_33, [1, 2048, 512]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_34: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_132, view_140);  convert_element_type_132 = view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_149: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_10: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_34);  add_34 = None
        any_11: "b8[]" = torch.ops.aten.any.default(isinf_10);  isinf_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_24: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f32[]" = torch.ops.aten.where.self(any_11, full_default_24, full_default_23);  any_11 = full_default_24 = full_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_10: "f32[]" = torch.ops.aten.neg.default(where_12)
        clamp_min_10: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_149, neg_10);  convert_element_type_149 = neg_10 = None
        clamp_max_10: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_10, where_12);  clamp_min_10 = where_12 = None
        convert_element_type_150: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_10, torch.float16);  clamp_max_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_151: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_150, torch.float32)
        pow_12: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_151, 2);  convert_element_type_151 = None
        mean_11: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_24: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_150, rsqrt_11);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_152: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_24, torch.float16);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_25: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg48_1, convert_element_type_152);  arg48_1 = convert_element_type_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_141: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_25, [2048, 512]);  mul_25 = None
        permute_65: "f16[512, 2048]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_34: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_141, permute_65);  view_141 = permute_65 = None
        view_142: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_34, [1, 2048, 2048]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_5: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_142);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_143: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu_5, [2048, 2048]);  relu_5 = None
        permute_66: "f16[2048, 512]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_35: "f16[2048, 512]" = torch.ops.aten.mm.default(view_143, permute_66);  view_143 = permute_66 = None
        view_144: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_35, [1, 2048, 512]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_36: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_150, view_144);  convert_element_type_150 = view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_157: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_11: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_36);  add_36 = None
        any_12: "b8[]" = torch.ops.aten.any.default(isinf_11);  isinf_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_26: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f32[]" = torch.ops.aten.where.self(any_12, full_default_26, full_default_25);  any_12 = full_default_26 = full_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_11: "f32[]" = torch.ops.aten.neg.default(where_13)
        clamp_min_11: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_157, neg_11);  convert_element_type_157 = neg_11 = None
        clamp_max_11: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_11, where_13);  clamp_min_11 = where_13 = None
        convert_element_type_158: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_11, torch.float16);  clamp_max_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_159: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_158, torch.float32)
        pow_13: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_159, 2);  convert_element_type_159 = None
        mean_12: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_37: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_26: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_158, rsqrt_12);  convert_element_type_158 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_160: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_26, torch.float16);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_27: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg51_1, convert_element_type_160);  arg51_1 = convert_element_type_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_169: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_79: "f16[512, 512]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_41: "f16[2048, 512]" = torch.ops.aten.mm.default(view_169, permute_79);  view_169 = permute_79 = None
        view_170: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_41, [1, 2048, 512]);  mm_41 = None
        view_171: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_170, [1, 2048, -1, 64]);  view_170 = None
        permute_80: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_83: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        expand_32: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_83, [1, 8, 64, 2048]);  permute_83 = None
        view_176: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_32, [8, 64, 2048]);  expand_32 = None
        bmm_14: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_175, view_176);  view_175 = view_176 = None
        view_177: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_14, [1, 8, 2048, 2048]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:300 in forward, code: position_bias = torch.zeros(
        full_default_35: "f16[1, 8, 2048, 2048]" = torch.ops.aten.full.default([1, 8, 2048, 2048], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_12: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_40: "i64[2048]" = torch.ops.aten.add.Tensor(iota_12, 0);  iota_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_12: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(add_40, 0);  add_40 = None
        unsqueeze_13: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
        unsqueeze_14: "i64[1, 1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge_1: "b8[1, 1, 2048, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_14, 0);  unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_26: "b8[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(ge_1, [1, -1, 2048, 2048]);  ge_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_29: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "f16[1, 1, 2048, 2048]" = torch.ops.aten.where.self(expand_26, full_default_29, full_default_30);  expand_26 = full_default_29 = full_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_50: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(full_default_35, where_15);  full_default_35 = where_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_51: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_177, add_50);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_191: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32);  add_51 = None
        amax_7: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_191, [-1], True)
        sub_9: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_191, amax_7);  convert_element_type_191 = amax_7 = None
        exp_7: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_8: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_11: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_192: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_11, torch.float16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_33: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_192, [1, 8, 2048, 2048]);  convert_element_type_192 = None
        view_180: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_33, [8, 2048, 2048]);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_172: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_81: "f16[512, 512]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_42: "f16[2048, 512]" = torch.ops.aten.mm.default(view_172, permute_81);  view_172 = permute_81 = None
        view_173: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_42, [1, 2048, 512]);  mm_42 = None
        view_174: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_173, [1, 2048, -1, 64]);  view_173 = None
        permute_82: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_34: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_82, [1, 8, 2048, 64]);  permute_82 = None
        view_181: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_34, [8, 2048, 64]);  expand_34 = None
        bmm_15: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_180, view_181);  view_180 = view_181 = None
        view_182: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 8, 2048, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_37: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_183: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_37, [1, 2048, -1]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_184: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_183, [2048, 512]);  view_183 = None
        permute_85: "f16[512, 512]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        mm_43: "f16[2048, 512]" = torch.ops.aten.mm.default(view_184, permute_85);  view_184 = permute_85 = None
        view_185: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_43, [1, 2048, 512]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_52: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_180, view_185);  convert_element_type_180 = view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_197: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_52, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_13: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_52);  add_52 = None
        any_14: "b8[]" = torch.ops.aten.any.default(isinf_13);  isinf_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_37: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "f32[]" = torch.ops.aten.where.self(any_14, full_default_37, full_default_36);  any_14 = full_default_37 = full_default_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_14: "f32[]" = torch.ops.aten.neg.default(where_18)
        clamp_min_13: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_197, neg_14);  convert_element_type_197 = neg_14 = None
        clamp_max_13: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_13, where_18);  clamp_min_13 = where_18 = None
        convert_element_type_198: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_13, torch.float16);  clamp_max_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_199: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_198, torch.float32)
        pow_16: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_199, 2);  convert_element_type_199 = None
        mean_15: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_53: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_33: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_198, rsqrt_15);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_200: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_33, torch.float16);  mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_34: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg63_1, convert_element_type_200);  arg63_1 = convert_element_type_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_186: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_34, [2048, 512]);  mul_34 = None
        permute_86: "f16[512, 2048]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_44: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_186, permute_86);  view_186 = permute_86 = None
        view_187: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_44, [1, 2048, 2048]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_6: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_187);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_188: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu_6, [2048, 2048]);  relu_6 = None
        permute_87: "f16[2048, 512]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_45: "f16[2048, 512]" = torch.ops.aten.mm.default(view_188, permute_87);  view_188 = permute_87 = None
        view_189: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_45, [1, 2048, 512]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_54: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_198, view_189);  convert_element_type_198 = view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_205: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_54, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_14: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_54);  add_54 = None
        any_15: "b8[]" = torch.ops.aten.any.default(isinf_14);  isinf_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_39: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "f32[]" = torch.ops.aten.where.self(any_15, full_default_39, full_default_38);  any_15 = full_default_39 = full_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_15: "f32[]" = torch.ops.aten.neg.default(where_19)
        clamp_min_14: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_205, neg_15);  convert_element_type_205 = neg_15 = None
        clamp_max_14: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_14, where_19);  clamp_min_14 = where_19 = None
        convert_element_type_206: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_14, torch.float16);  clamp_max_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_207: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_206, torch.float32)
        pow_17: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_207, 2);  convert_element_type_207 = None
        mean_16: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_55: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_35: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_206, rsqrt_16);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_208: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_35, torch.float16);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_36: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg66_1, convert_element_type_208);  arg66_1 = convert_element_type_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_190: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_36, [2048, 512])
        permute_88: "f16[512, 512]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_46: "f16[2048, 512]" = torch.ops.aten.mm.default(view_190, permute_88);  view_190 = permute_88 = None
        view_191: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_46, [1, 2048, 512]);  mm_46 = None
        view_192: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_191, [1, 2048, -1, 64]);  view_191 = None
        permute_89: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_35: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_89, [1, 8, 2048, 64]);  permute_89 = None
        view_199: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_35, [8, 2048, 64]);  expand_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_193: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_36, [2048, 512])
        permute_90: "f16[512, 512]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_47: "f16[2048, 512]" = torch.ops.aten.mm.default(view_193, permute_90);  view_193 = permute_90 = None
        view_194: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_47, [1, 2048, 512]);  mm_47 = None
        view_195: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_194, [1, 2048, -1, 64]);  view_194 = None
        permute_91: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_94: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        expand_36: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_94, [1, 8, 64, 2048]);  permute_94 = None
        view_200: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_36, [8, 64, 2048]);  expand_36 = None
        bmm_16: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_199, view_200);  view_199 = view_200 = None
        view_201: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_16, [1, 8, 2048, 2048]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_56: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_201, add_46);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_217: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_56, torch.float32);  add_56 = None
        amax_8: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_217, [-1], True)
        sub_10: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_217, amax_8);  convert_element_type_217 = amax_8 = None
        exp_8: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_9: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_218: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_12, torch.float16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_37: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_218, [1, 8, 2048, 2048]);  convert_element_type_218 = None
        view_204: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_37, [8, 2048, 2048]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_196: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_36, [2048, 512]);  mul_36 = None
        permute_92: "f16[512, 512]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_48: "f16[2048, 512]" = torch.ops.aten.mm.default(view_196, permute_92);  view_196 = permute_92 = None
        view_197: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_48, [1, 2048, 512]);  mm_48 = None
        view_198: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_197, [1, 2048, -1, 64]);  view_197 = None
        permute_93: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_38: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_93, [1, 8, 2048, 64]);  permute_93 = None
        view_205: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_38, [8, 2048, 64]);  expand_38 = None
        bmm_17: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_204, view_205);  view_204 = view_205 = None
        view_206: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 8, 2048, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        clone_42: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_207: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_42, [1, 2048, -1]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_208: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_207, [2048, 512]);  view_207 = None
        permute_96: "f16[512, 512]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_49: "f16[2048, 512]" = torch.ops.aten.mm.default(view_208, permute_96);  view_208 = permute_96 = None
        view_209: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_49, [1, 2048, 512]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_57: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_206, view_209);  convert_element_type_206 = view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_223: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_57, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_15: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_57);  add_57 = None
        any_16: "b8[]" = torch.ops.aten.any.default(isinf_15);  isinf_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_41: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f32[]" = torch.ops.aten.where.self(any_16, full_default_41, full_default_40);  any_16 = full_default_41 = full_default_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_16: "f32[]" = torch.ops.aten.neg.default(where_20)
        clamp_min_15: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_223, neg_16);  convert_element_type_223 = neg_16 = None
        clamp_max_15: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_15, where_20);  clamp_min_15 = where_20 = None
        convert_element_type_224: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_15, torch.float16);  clamp_max_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_225: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_224, torch.float32)
        pow_18: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_225, 2);  convert_element_type_225 = None
        mean_17: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_58: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_37: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_224, rsqrt_17);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_226: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_37, torch.float16);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_38: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg71_1, convert_element_type_226);  arg71_1 = convert_element_type_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_210: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_38, [2048, 512]);  mul_38 = None
        permute_97: "f16[512, 512]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_50: "f16[2048, 512]" = torch.ops.aten.mm.default(view_210, permute_97);  view_210 = permute_97 = None
        view_211: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_50, [1, 2048, 512]);  mm_50 = None
        view_212: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_211, [1, 2048, -1, 64]);  view_211 = None
        permute_98: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_39: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_98, [1, 8, 2048, 64]);  permute_98 = None
        view_219: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_39, [8, 2048, 64]);  expand_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_213: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_99: "f16[512, 512]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_51: "f16[2048, 512]" = torch.ops.aten.mm.default(view_213, permute_99);  view_213 = permute_99 = None
        view_214: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_51, [1, 2048, 512]);  mm_51 = None
        view_215: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_214, [1, 2048, -1, 64]);  view_214 = None
        permute_100: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_103: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2]);  permute_100 = None
        expand_40: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_103, [1, 8, 64, 2048]);  permute_103 = None
        view_220: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_40, [8, 64, 2048]);  expand_40 = None
        bmm_18: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_219, view_220);  view_219 = view_220 = None
        view_221: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_18, [1, 8, 2048, 2048]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_59: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_221, add_50);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_235: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        amax_9: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_235, [-1], True)
        sub_11: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_235, amax_9);  convert_element_type_235 = amax_9 = None
        exp_9: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_10: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_236: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_13, torch.float16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_41: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_236, [1, 8, 2048, 2048]);  convert_element_type_236 = None
        view_224: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_41, [8, 2048, 2048]);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_216: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_101: "f16[512, 512]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_52: "f16[2048, 512]" = torch.ops.aten.mm.default(view_216, permute_101);  view_216 = permute_101 = None
        view_217: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_52, [1, 2048, 512]);  mm_52 = None
        view_218: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_217, [1, 2048, -1, 64]);  view_217 = None
        permute_102: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_42: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_102, [1, 8, 2048, 64]);  permute_102 = None
        view_225: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_42, [8, 2048, 64]);  expand_42 = None
        bmm_19: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_224, view_225);  view_224 = view_225 = None
        view_226: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 8, 2048, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_104: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_45: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_227: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_45, [1, 2048, -1]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_228: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_227, [2048, 512]);  view_227 = None
        permute_105: "f16[512, 512]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        mm_53: "f16[2048, 512]" = torch.ops.aten.mm.default(view_228, permute_105);  view_228 = permute_105 = None
        view_229: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_53, [1, 2048, 512]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_60: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_224, view_229);  convert_element_type_224 = view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_241: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_60, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_16: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_60);  add_60 = None
        any_17: "b8[]" = torch.ops.aten.any.default(isinf_16);  isinf_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_43: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: "f32[]" = torch.ops.aten.where.self(any_17, full_default_43, full_default_42);  any_17 = full_default_43 = full_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_17: "f32[]" = torch.ops.aten.neg.default(where_21)
        clamp_min_16: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_241, neg_17);  convert_element_type_241 = neg_17 = None
        clamp_max_16: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_16, where_21);  clamp_min_16 = where_21 = None
        convert_element_type_242: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_16, torch.float16);  clamp_max_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_243: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_242, torch.float32)
        pow_19: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_243, 2);  convert_element_type_243 = None
        mean_18: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_61: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_39: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_242, rsqrt_18);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_244: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_39, torch.float16);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_40: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg76_1, convert_element_type_244);  arg76_1 = convert_element_type_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_230: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_40, [2048, 512]);  mul_40 = None
        permute_106: "f16[512, 2048]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        mm_54: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_230, permute_106);  view_230 = permute_106 = None
        view_231: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_54, [1, 2048, 2048]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_7: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_231);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_232: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu_7, [2048, 2048]);  relu_7 = None
        permute_107: "f16[2048, 512]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_55: "f16[2048, 512]" = torch.ops.aten.mm.default(view_232, permute_107);  view_232 = permute_107 = None
        view_233: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_55, [1, 2048, 512]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_62: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_242, view_233);  convert_element_type_242 = view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_249: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_17: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_62);  add_62 = None
        any_18: "b8[]" = torch.ops.aten.any.default(isinf_17);  isinf_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_45: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f32[]" = torch.ops.aten.where.self(any_18, full_default_45, full_default_44);  any_18 = full_default_45 = full_default_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_18: "f32[]" = torch.ops.aten.neg.default(where_22)
        clamp_min_17: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_249, neg_18);  convert_element_type_249 = neg_18 = None
        clamp_max_17: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_17, where_22);  clamp_min_17 = where_22 = None
        convert_element_type_250: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_17, torch.float16);  clamp_max_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_251: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_250, torch.float32)
        pow_20: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_251, 2);  convert_element_type_251 = None
        mean_19: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_41: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_250, rsqrt_19);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_252: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_41, torch.float16);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_42: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg79_1, convert_element_type_252);  arg79_1 = convert_element_type_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_234: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_42, [2048, 512])
        permute_108: "f16[512, 512]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        mm_56: "f16[2048, 512]" = torch.ops.aten.mm.default(view_234, permute_108);  view_234 = permute_108 = None
        view_235: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_56, [1, 2048, 512]);  mm_56 = None
        view_236: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_235, [1, 2048, -1, 64]);  view_235 = None
        permute_109: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_43: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_109, [1, 8, 2048, 64]);  permute_109 = None
        view_243: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_43, [8, 2048, 64]);  expand_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_237: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_42, [2048, 512])
        permute_110: "f16[512, 512]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_57: "f16[2048, 512]" = torch.ops.aten.mm.default(view_237, permute_110);  view_237 = permute_110 = None
        view_238: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_57, [1, 2048, 512]);  mm_57 = None
        view_239: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_238, [1, 2048, -1, 64]);  view_238 = None
        permute_111: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_114: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_111, [0, 1, 3, 2]);  permute_111 = None
        expand_44: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_114, [1, 8, 64, 2048]);  permute_114 = None
        view_244: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_44, [8, 64, 2048]);  expand_44 = None
        bmm_20: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_243, view_244);  view_243 = view_244 = None
        view_245: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_20, [1, 8, 2048, 2048]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_64: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_245, add_46);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_261: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_64, torch.float32);  add_64 = None
        amax_10: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_261, [-1], True)
        sub_12: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_261, amax_10);  convert_element_type_261 = amax_10 = None
        exp_10: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_11: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_262: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_14, torch.float16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_45: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_262, [1, 8, 2048, 2048]);  convert_element_type_262 = None
        view_248: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_45, [8, 2048, 2048]);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_240: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_42, [2048, 512]);  mul_42 = None
        permute_112: "f16[512, 512]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_58: "f16[2048, 512]" = torch.ops.aten.mm.default(view_240, permute_112);  view_240 = permute_112 = None
        view_241: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_58, [1, 2048, 512]);  mm_58 = None
        view_242: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_241, [1, 2048, -1, 64]);  view_241 = None
        permute_113: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_46: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_113, [1, 8, 2048, 64]);  permute_113 = None
        view_249: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_46, [8, 2048, 64]);  expand_46 = None
        bmm_21: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_248, view_249);  view_248 = view_249 = None
        view_250: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 8, 2048, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_115: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        clone_50: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_251: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_50, [1, 2048, -1]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_252: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_251, [2048, 512]);  view_251 = None
        permute_116: "f16[512, 512]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_59: "f16[2048, 512]" = torch.ops.aten.mm.default(view_252, permute_116);  view_252 = permute_116 = None
        view_253: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_59, [1, 2048, 512]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_65: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_250, view_253);  convert_element_type_250 = view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_267: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_65, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_18: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_65);  add_65 = None
        any_19: "b8[]" = torch.ops.aten.any.default(isinf_18);  isinf_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_47: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "f32[]" = torch.ops.aten.where.self(any_19, full_default_47, full_default_46);  any_19 = full_default_47 = full_default_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_19: "f32[]" = torch.ops.aten.neg.default(where_23)
        clamp_min_18: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_267, neg_19);  convert_element_type_267 = neg_19 = None
        clamp_max_18: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_18, where_23);  clamp_min_18 = where_23 = None
        convert_element_type_268: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_18, torch.float16);  clamp_max_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_269: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_268, torch.float32)
        pow_21: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_269, 2);  convert_element_type_269 = None
        mean_20: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_66: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_43: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_268, rsqrt_20);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_270: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_43, torch.float16);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_44: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg84_1, convert_element_type_270);  arg84_1 = convert_element_type_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_254: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_44, [2048, 512]);  mul_44 = None
        permute_117: "f16[512, 512]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_60: "f16[2048, 512]" = torch.ops.aten.mm.default(view_254, permute_117);  view_254 = permute_117 = None
        view_255: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_60, [1, 2048, 512]);  mm_60 = None
        view_256: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_255, [1, 2048, -1, 64]);  view_255 = None
        permute_118: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_47: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_118, [1, 8, 2048, 64]);  permute_118 = None
        view_263: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_47, [8, 2048, 64]);  expand_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_257: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_119: "f16[512, 512]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_61: "f16[2048, 512]" = torch.ops.aten.mm.default(view_257, permute_119);  view_257 = permute_119 = None
        view_258: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_61, [1, 2048, 512]);  mm_61 = None
        view_259: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_258, [1, 2048, -1, 64]);  view_258 = None
        permute_120: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_123: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2]);  permute_120 = None
        expand_48: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_123, [1, 8, 64, 2048]);  permute_123 = None
        view_264: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_48, [8, 64, 2048]);  expand_48 = None
        bmm_22: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_263, view_264);  view_263 = view_264 = None
        view_265: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_22, [1, 8, 2048, 2048]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_67: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_265, add_50);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_279: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_67, torch.float32);  add_67 = None
        amax_11: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_279, [-1], True)
        sub_13: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_279, amax_11);  convert_element_type_279 = amax_11 = None
        exp_11: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_12: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_280: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_15, torch.float16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_49: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_280, [1, 8, 2048, 2048]);  convert_element_type_280 = None
        view_268: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_49, [8, 2048, 2048]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_260: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_121: "f16[512, 512]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_62: "f16[2048, 512]" = torch.ops.aten.mm.default(view_260, permute_121);  view_260 = permute_121 = None
        view_261: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_62, [1, 2048, 512]);  mm_62 = None
        view_262: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_261, [1, 2048, -1, 64]);  view_261 = None
        permute_122: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_50: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_122, [1, 8, 2048, 64]);  permute_122 = None
        view_269: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_50, [8, 2048, 64]);  expand_50 = None
        bmm_23: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_268, view_269);  view_268 = view_269 = None
        view_270: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 8, 2048, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_124: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        clone_53: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_271: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_53, [1, 2048, -1]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_272: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_271, [2048, 512]);  view_271 = None
        permute_125: "f16[512, 512]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        mm_63: "f16[2048, 512]" = torch.ops.aten.mm.default(view_272, permute_125);  view_272 = permute_125 = None
        view_273: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_63, [1, 2048, 512]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_68: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_268, view_273);  convert_element_type_268 = view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_285: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_68, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_19: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_68);  add_68 = None
        any_20: "b8[]" = torch.ops.aten.any.default(isinf_19);  isinf_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_49: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "f32[]" = torch.ops.aten.where.self(any_20, full_default_49, full_default_48);  any_20 = full_default_49 = full_default_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_20: "f32[]" = torch.ops.aten.neg.default(where_24)
        clamp_min_19: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_285, neg_20);  convert_element_type_285 = neg_20 = None
        clamp_max_19: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_19, where_24);  clamp_min_19 = where_24 = None
        convert_element_type_286: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_19, torch.float16);  clamp_max_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_287: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_286, torch.float32)
        pow_22: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_287, 2);  convert_element_type_287 = None
        mean_21: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_69: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_45: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_286, rsqrt_21);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_288: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_45, torch.float16);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_46: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg89_1, convert_element_type_288);  arg89_1 = convert_element_type_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_274: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_46, [2048, 512]);  mul_46 = None
        permute_126: "f16[512, 2048]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_64: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_274, permute_126);  view_274 = permute_126 = None
        view_275: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_64, [1, 2048, 2048]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_8: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_275);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_276: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu_8, [2048, 2048]);  relu_8 = None
        permute_127: "f16[2048, 512]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        mm_65: "f16[2048, 512]" = torch.ops.aten.mm.default(view_276, permute_127);  view_276 = permute_127 = None
        view_277: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_65, [1, 2048, 512]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_70: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_286, view_277);  convert_element_type_286 = view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_293: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_70, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_20: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_70);  add_70 = None
        any_21: "b8[]" = torch.ops.aten.any.default(isinf_20);  isinf_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_51: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_50: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[]" = torch.ops.aten.where.self(any_21, full_default_51, full_default_50);  any_21 = full_default_51 = full_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_21: "f32[]" = torch.ops.aten.neg.default(where_25)
        clamp_min_20: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_293, neg_21);  convert_element_type_293 = neg_21 = None
        clamp_max_20: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_20, where_25);  clamp_min_20 = where_25 = None
        convert_element_type_294: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_20, torch.float16);  clamp_max_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_295: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_294, torch.float32)
        pow_23: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_295, 2);  convert_element_type_295 = None
        mean_22: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_71: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_47: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_294, rsqrt_22);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_296: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_47, torch.float16);  mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_48: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg92_1, convert_element_type_296);  arg92_1 = convert_element_type_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_278: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_48, [2048, 512])
        permute_128: "f16[512, 512]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        mm_66: "f16[2048, 512]" = torch.ops.aten.mm.default(view_278, permute_128);  view_278 = permute_128 = None
        view_279: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_66, [1, 2048, 512]);  mm_66 = None
        view_280: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_279, [1, 2048, -1, 64]);  view_279 = None
        permute_129: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_51: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_129, [1, 8, 2048, 64]);  permute_129 = None
        view_287: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_51, [8, 2048, 64]);  expand_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_281: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_48, [2048, 512])
        permute_130: "f16[512, 512]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_67: "f16[2048, 512]" = torch.ops.aten.mm.default(view_281, permute_130);  view_281 = permute_130 = None
        view_282: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_67, [1, 2048, 512]);  mm_67 = None
        view_283: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_282, [1, 2048, -1, 64]);  view_282 = None
        permute_131: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_134: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2]);  permute_131 = None
        expand_52: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_134, [1, 8, 64, 2048]);  permute_134 = None
        view_288: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_52, [8, 64, 2048]);  expand_52 = None
        bmm_24: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_287, view_288);  view_287 = view_288 = None
        view_289: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_24, [1, 8, 2048, 2048]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_72: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_289, add_46);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_305: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_72, torch.float32);  add_72 = None
        amax_12: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_305, [-1], True)
        sub_14: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_305, amax_12);  convert_element_type_305 = amax_12 = None
        exp_12: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_13: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_306: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_16, torch.float16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_53: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_306, [1, 8, 2048, 2048]);  convert_element_type_306 = None
        view_292: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_53, [8, 2048, 2048]);  expand_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_284: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_48, [2048, 512]);  mul_48 = None
        permute_132: "f16[512, 512]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_68: "f16[2048, 512]" = torch.ops.aten.mm.default(view_284, permute_132);  view_284 = permute_132 = None
        view_285: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_68, [1, 2048, 512]);  mm_68 = None
        view_286: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_285, [1, 2048, -1, 64]);  view_285 = None
        permute_133: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_54: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_133, [1, 8, 2048, 64]);  permute_133 = None
        view_293: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_54, [8, 2048, 64]);  expand_54 = None
        bmm_25: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_292, view_293);  view_292 = view_293 = None
        view_294: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_25, [1, 8, 2048, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_135: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        clone_58: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_295: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_58, [1, 2048, -1]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_296: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_295, [2048, 512]);  view_295 = None
        permute_136: "f16[512, 512]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_69: "f16[2048, 512]" = torch.ops.aten.mm.default(view_296, permute_136);  view_296 = permute_136 = None
        view_297: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_69, [1, 2048, 512]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_73: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_294, view_297);  convert_element_type_294 = view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_311: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_73, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_21: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_73);  add_73 = None
        any_22: "b8[]" = torch.ops.aten.any.default(isinf_21);  isinf_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_53: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "f32[]" = torch.ops.aten.where.self(any_22, full_default_53, full_default_52);  any_22 = full_default_53 = full_default_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_22: "f32[]" = torch.ops.aten.neg.default(where_26)
        clamp_min_21: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_311, neg_22);  convert_element_type_311 = neg_22 = None
        clamp_max_21: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_21, where_26);  clamp_min_21 = where_26 = None
        convert_element_type_312: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_21, torch.float16);  clamp_max_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_313: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_312, torch.float32)
        pow_24: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_313, 2);  convert_element_type_313 = None
        mean_23: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_74: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_49: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_312, rsqrt_23);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_314: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_49, torch.float16);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_50: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg97_1, convert_element_type_314);  arg97_1 = convert_element_type_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_298: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_50, [2048, 512]);  mul_50 = None
        permute_137: "f16[512, 512]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        mm_70: "f16[2048, 512]" = torch.ops.aten.mm.default(view_298, permute_137);  view_298 = permute_137 = None
        view_299: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_70, [1, 2048, 512]);  mm_70 = None
        view_300: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_299, [1, 2048, -1, 64]);  view_299 = None
        permute_138: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_55: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_138, [1, 8, 2048, 64]);  permute_138 = None
        view_307: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_55, [8, 2048, 64]);  expand_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_301: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_139: "f16[512, 512]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_71: "f16[2048, 512]" = torch.ops.aten.mm.default(view_301, permute_139);  view_301 = permute_139 = None
        view_302: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_71, [1, 2048, 512]);  mm_71 = None
        view_303: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_302, [1, 2048, -1, 64]);  view_302 = None
        permute_140: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_143: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 2]);  permute_140 = None
        expand_56: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_143, [1, 8, 64, 2048]);  permute_143 = None
        view_308: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_56, [8, 64, 2048]);  expand_56 = None
        bmm_26: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_307, view_308);  view_307 = view_308 = None
        view_309: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_26, [1, 8, 2048, 2048]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_75: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_309, add_50);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_323: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_75, torch.float32);  add_75 = None
        amax_13: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_323, [-1], True)
        sub_15: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_323, amax_13);  convert_element_type_323 = amax_13 = None
        exp_13: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_14: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_324: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_17, torch.float16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_57: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_324, [1, 8, 2048, 2048]);  convert_element_type_324 = None
        view_312: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_57, [8, 2048, 2048]);  expand_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_304: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_141: "f16[512, 512]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_72: "f16[2048, 512]" = torch.ops.aten.mm.default(view_304, permute_141);  view_304 = permute_141 = None
        view_305: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_72, [1, 2048, 512]);  mm_72 = None
        view_306: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_305, [1, 2048, -1, 64]);  view_305 = None
        permute_142: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_58: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_142, [1, 8, 2048, 64]);  permute_142 = None
        view_313: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_58, [8, 2048, 64]);  expand_58 = None
        bmm_27: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_312, view_313);  view_312 = view_313 = None
        view_314: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_27, [1, 8, 2048, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_144: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
        clone_61: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_315: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_61, [1, 2048, -1]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_316: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_315, [2048, 512]);  view_315 = None
        permute_145: "f16[512, 512]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_73: "f16[2048, 512]" = torch.ops.aten.mm.default(view_316, permute_145);  view_316 = permute_145 = None
        view_317: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_73, [1, 2048, 512]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_76: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_312, view_317);  convert_element_type_312 = view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_329: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_76, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_22: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_76);  add_76 = None
        any_23: "b8[]" = torch.ops.aten.any.default(isinf_22);  isinf_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_55: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "f32[]" = torch.ops.aten.where.self(any_23, full_default_55, full_default_54);  any_23 = full_default_55 = full_default_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_23: "f32[]" = torch.ops.aten.neg.default(where_27)
        clamp_min_22: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_329, neg_23);  convert_element_type_329 = neg_23 = None
        clamp_max_22: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_22, where_27);  clamp_min_22 = where_27 = None
        convert_element_type_330: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_22, torch.float16);  clamp_max_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_331: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_330, torch.float32)
        pow_25: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_331, 2);  convert_element_type_331 = None
        mean_24: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_77: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_51: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_330, rsqrt_24);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_332: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_51, torch.float16);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_52: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg102_1, convert_element_type_332);  arg102_1 = convert_element_type_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_318: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_52, [2048, 512]);  mul_52 = None
        permute_146: "f16[512, 2048]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_74: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_318, permute_146);  view_318 = permute_146 = None
        view_319: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_74, [1, 2048, 2048]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_9: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_319);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_320: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu_9, [2048, 2048]);  relu_9 = None
        permute_147: "f16[2048, 512]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_75: "f16[2048, 512]" = torch.ops.aten.mm.default(view_320, permute_147);  view_320 = permute_147 = None
        view_321: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_75, [1, 2048, 512]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_78: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_330, view_321);  convert_element_type_330 = view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_337: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_78, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_23: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_78);  add_78 = None
        any_24: "b8[]" = torch.ops.aten.any.default(isinf_23);  isinf_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_57: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "f32[]" = torch.ops.aten.where.self(any_24, full_default_57, full_default_56);  any_24 = full_default_57 = full_default_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_24: "f32[]" = torch.ops.aten.neg.default(where_28)
        clamp_min_23: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_337, neg_24);  convert_element_type_337 = neg_24 = None
        clamp_max_23: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_23, where_28);  clamp_min_23 = where_28 = None
        convert_element_type_338: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_23, torch.float16);  clamp_max_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_339: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_338, torch.float32)
        pow_26: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_339, 2);  convert_element_type_339 = None
        mean_25: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_79: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_53: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_338, rsqrt_25);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_340: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_53, torch.float16);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_54: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg105_1, convert_element_type_340);  arg105_1 = convert_element_type_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_322: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_54, [2048, 512])
        permute_148: "f16[512, 512]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        mm_76: "f16[2048, 512]" = torch.ops.aten.mm.default(view_322, permute_148);  view_322 = permute_148 = None
        view_323: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_76, [1, 2048, 512]);  mm_76 = None
        view_324: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_323, [1, 2048, -1, 64]);  view_323 = None
        permute_149: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_59: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_149, [1, 8, 2048, 64]);  permute_149 = None
        view_331: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_59, [8, 2048, 64]);  expand_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_325: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_54, [2048, 512])
        permute_150: "f16[512, 512]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_77: "f16[2048, 512]" = torch.ops.aten.mm.default(view_325, permute_150);  view_325 = permute_150 = None
        view_326: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_77, [1, 2048, 512]);  mm_77 = None
        view_327: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_326, [1, 2048, -1, 64]);  view_326 = None
        permute_151: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_154: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_151, [0, 1, 3, 2]);  permute_151 = None
        expand_60: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_154, [1, 8, 64, 2048]);  permute_154 = None
        view_332: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_60, [8, 64, 2048]);  expand_60 = None
        bmm_28: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_331, view_332);  view_331 = view_332 = None
        view_333: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_28, [1, 8, 2048, 2048]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_80: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_333, add_46);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_349: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_80, torch.float32);  add_80 = None
        amax_14: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_349, [-1], True)
        sub_16: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_349, amax_14);  convert_element_type_349 = amax_14 = None
        exp_14: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_15: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_350: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_18, torch.float16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_61: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_350, [1, 8, 2048, 2048]);  convert_element_type_350 = None
        view_336: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_61, [8, 2048, 2048]);  expand_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_328: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_54, [2048, 512]);  mul_54 = None
        permute_152: "f16[512, 512]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_78: "f16[2048, 512]" = torch.ops.aten.mm.default(view_328, permute_152);  view_328 = permute_152 = None
        view_329: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_78, [1, 2048, 512]);  mm_78 = None
        view_330: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_329, [1, 2048, -1, 64]);  view_329 = None
        permute_153: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_62: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_153, [1, 8, 2048, 64]);  permute_153 = None
        view_337: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_62, [8, 2048, 64]);  expand_62 = None
        bmm_29: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_336, view_337);  view_336 = view_337 = None
        view_338: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_29, [1, 8, 2048, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        clone_66: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_339: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_66, [1, 2048, -1]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_340: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_339, [2048, 512]);  view_339 = None
        permute_156: "f16[512, 512]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_79: "f16[2048, 512]" = torch.ops.aten.mm.default(view_340, permute_156);  view_340 = permute_156 = None
        view_341: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_79, [1, 2048, 512]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_81: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_338, view_341);  convert_element_type_338 = view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_355: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_81, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_24: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_81);  add_81 = None
        any_25: "b8[]" = torch.ops.aten.any.default(isinf_24);  isinf_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_59: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_29: "f32[]" = torch.ops.aten.where.self(any_25, full_default_59, full_default_58);  any_25 = full_default_59 = full_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_25: "f32[]" = torch.ops.aten.neg.default(where_29)
        clamp_min_24: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_355, neg_25);  convert_element_type_355 = neg_25 = None
        clamp_max_24: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_24, where_29);  clamp_min_24 = where_29 = None
        convert_element_type_356: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_24, torch.float16);  clamp_max_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_357: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_356, torch.float32)
        pow_27: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_357, 2);  convert_element_type_357 = None
        mean_26: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_82: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_55: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_356, rsqrt_26);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_358: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_55, torch.float16);  mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_56: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg110_1, convert_element_type_358);  arg110_1 = convert_element_type_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_342: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_56, [2048, 512]);  mul_56 = None
        permute_157: "f16[512, 512]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        mm_80: "f16[2048, 512]" = torch.ops.aten.mm.default(view_342, permute_157);  view_342 = permute_157 = None
        view_343: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_80, [1, 2048, 512]);  mm_80 = None
        view_344: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_343, [1, 2048, -1, 64]);  view_343 = None
        permute_158: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_63: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_158, [1, 8, 2048, 64]);  permute_158 = None
        view_351: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_63, [8, 2048, 64]);  expand_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_345: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_159: "f16[512, 512]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_81: "f16[2048, 512]" = torch.ops.aten.mm.default(view_345, permute_159);  view_345 = permute_159 = None
        view_346: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_81, [1, 2048, 512]);  mm_81 = None
        view_347: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_346, [1, 2048, -1, 64]);  view_346 = None
        permute_160: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_163: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_160, [0, 1, 3, 2]);  permute_160 = None
        expand_64: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_163, [1, 8, 64, 2048]);  permute_163 = None
        view_352: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_64, [8, 64, 2048]);  expand_64 = None
        bmm_30: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_351, view_352);  view_351 = view_352 = None
        view_353: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_30, [1, 8, 2048, 2048]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_83: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_353, add_50);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_367: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32);  add_83 = None
        amax_15: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_367, [-1], True)
        sub_17: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_367, amax_15);  convert_element_type_367 = amax_15 = None
        exp_15: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_16: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_368: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_19, torch.float16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_65: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_368, [1, 8, 2048, 2048]);  convert_element_type_368 = None
        view_356: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_65, [8, 2048, 2048]);  expand_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_348: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_161: "f16[512, 512]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_82: "f16[2048, 512]" = torch.ops.aten.mm.default(view_348, permute_161);  view_348 = permute_161 = None
        view_349: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_82, [1, 2048, 512]);  mm_82 = None
        view_350: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_349, [1, 2048, -1, 64]);  view_349 = None
        permute_162: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_66: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_162, [1, 8, 2048, 64]);  permute_162 = None
        view_357: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_66, [8, 2048, 64]);  expand_66 = None
        bmm_31: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_356, view_357);  view_356 = view_357 = None
        view_358: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_31, [1, 8, 2048, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_164: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        clone_69: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_359: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_69, [1, 2048, -1]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_360: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_359, [2048, 512]);  view_359 = None
        permute_165: "f16[512, 512]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_83: "f16[2048, 512]" = torch.ops.aten.mm.default(view_360, permute_165);  view_360 = permute_165 = None
        view_361: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_83, [1, 2048, 512]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_84: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_356, view_361);  convert_element_type_356 = view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_373: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_84, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_25: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_84);  add_84 = None
        any_26: "b8[]" = torch.ops.aten.any.default(isinf_25);  isinf_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_61: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_30: "f32[]" = torch.ops.aten.where.self(any_26, full_default_61, full_default_60);  any_26 = full_default_61 = full_default_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_26: "f32[]" = torch.ops.aten.neg.default(where_30)
        clamp_min_25: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_373, neg_26);  convert_element_type_373 = neg_26 = None
        clamp_max_25: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_25, where_30);  clamp_min_25 = where_30 = None
        convert_element_type_374: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_25, torch.float16);  clamp_max_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_375: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_374, torch.float32)
        pow_28: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_375, 2);  convert_element_type_375 = None
        mean_27: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_85: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_57: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_374, rsqrt_27);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_376: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_57, torch.float16);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_58: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg115_1, convert_element_type_376);  arg115_1 = convert_element_type_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_362: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_58, [2048, 512]);  mul_58 = None
        permute_166: "f16[512, 2048]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        mm_84: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_362, permute_166);  view_362 = permute_166 = None
        view_363: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_84, [1, 2048, 2048]);  mm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_10: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_363);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_364: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu_10, [2048, 2048]);  relu_10 = None
        permute_167: "f16[2048, 512]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_85: "f16[2048, 512]" = torch.ops.aten.mm.default(view_364, permute_167);  view_364 = permute_167 = None
        view_365: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_85, [1, 2048, 512]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_86: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_374, view_365);  convert_element_type_374 = view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_381: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_86, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_26: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_86);  add_86 = None
        any_27: "b8[]" = torch.ops.aten.any.default(isinf_26);  isinf_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_63: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_62: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31: "f32[]" = torch.ops.aten.where.self(any_27, full_default_63, full_default_62);  any_27 = full_default_63 = full_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_27: "f32[]" = torch.ops.aten.neg.default(where_31)
        clamp_min_26: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_381, neg_27);  convert_element_type_381 = neg_27 = None
        clamp_max_26: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_26, where_31);  clamp_min_26 = where_31 = None
        convert_element_type_382: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_26, torch.float16);  clamp_max_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_383: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_382, torch.float32)
        pow_29: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_383, 2);  convert_element_type_383 = None
        mean_28: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_87: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_59: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_382, rsqrt_28);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_384: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_59, torch.float16);  mul_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_60: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg118_1, convert_element_type_384);  arg118_1 = convert_element_type_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_366: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_60, [2048, 512])
        permute_168: "f16[512, 512]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_86: "f16[2048, 512]" = torch.ops.aten.mm.default(view_366, permute_168);  view_366 = permute_168 = None
        view_367: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_86, [1, 2048, 512]);  mm_86 = None
        view_368: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_367, [1, 2048, -1, 64]);  view_367 = None
        permute_169: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_67: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_169, [1, 8, 2048, 64]);  permute_169 = None
        view_375: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_67, [8, 2048, 64]);  expand_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_369: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_60, [2048, 512])
        permute_170: "f16[512, 512]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        mm_87: "f16[2048, 512]" = torch.ops.aten.mm.default(view_369, permute_170);  view_369 = permute_170 = None
        view_370: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_87, [1, 2048, 512]);  mm_87 = None
        view_371: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_370, [1, 2048, -1, 64]);  view_370 = None
        permute_171: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_174: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_171, [0, 1, 3, 2]);  permute_171 = None
        expand_68: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_174, [1, 8, 64, 2048]);  permute_174 = None
        view_376: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_68, [8, 64, 2048]);  expand_68 = None
        bmm_32: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_375, view_376);  view_375 = view_376 = None
        view_377: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_32, [1, 8, 2048, 2048]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_88: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_377, add_46);  view_377 = add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_393: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_88, torch.float32);  add_88 = None
        amax_16: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_393, [-1], True)
        sub_18: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_393, amax_16);  convert_element_type_393 = amax_16 = None
        exp_16: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_17: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_394: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_20, torch.float16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_69: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_394, [1, 8, 2048, 2048]);  convert_element_type_394 = None
        view_380: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_69, [8, 2048, 2048]);  expand_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_372: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_60, [2048, 512]);  mul_60 = None
        permute_172: "f16[512, 512]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_88: "f16[2048, 512]" = torch.ops.aten.mm.default(view_372, permute_172);  view_372 = permute_172 = None
        view_373: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_88, [1, 2048, 512]);  mm_88 = None
        view_374: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_373, [1, 2048, -1, 64]);  view_373 = None
        permute_173: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_70: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_173, [1, 8, 2048, 64]);  permute_173 = None
        view_381: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_70, [8, 2048, 64]);  expand_70 = None
        bmm_33: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_380, view_381);  view_380 = view_381 = None
        view_382: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_33, [1, 8, 2048, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_175: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        clone_74: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_383: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_74, [1, 2048, -1]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_384: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_383, [2048, 512]);  view_383 = None
        permute_176: "f16[512, 512]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_89: "f16[2048, 512]" = torch.ops.aten.mm.default(view_384, permute_176);  view_384 = permute_176 = None
        view_385: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_89, [1, 2048, 512]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_89: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_382, view_385);  convert_element_type_382 = view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_399: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_89, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_27: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_89);  add_89 = None
        any_28: "b8[]" = torch.ops.aten.any.default(isinf_27);  isinf_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_65: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32: "f32[]" = torch.ops.aten.where.self(any_28, full_default_65, full_default_64);  any_28 = full_default_65 = full_default_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_28: "f32[]" = torch.ops.aten.neg.default(where_32)
        clamp_min_27: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_399, neg_28);  convert_element_type_399 = neg_28 = None
        clamp_max_27: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_27, where_32);  clamp_min_27 = where_32 = None
        convert_element_type_400: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_27, torch.float16);  clamp_max_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_401: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_400, torch.float32)
        pow_30: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_401, 2);  convert_element_type_401 = None
        mean_29: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_90: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_61: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_400, rsqrt_29);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_402: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_61, torch.float16);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_62: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg123_1, convert_element_type_402);  arg123_1 = convert_element_type_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_386: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_62, [2048, 512]);  mul_62 = None
        permute_177: "f16[512, 512]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        mm_90: "f16[2048, 512]" = torch.ops.aten.mm.default(view_386, permute_177);  view_386 = permute_177 = None
        view_387: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_90, [1, 2048, 512]);  mm_90 = None
        view_388: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_387, [1, 2048, -1, 64]);  view_387 = None
        permute_178: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_71: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_178, [1, 8, 2048, 64]);  permute_178 = None
        view_395: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_71, [8, 2048, 64]);  expand_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_389: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_179: "f16[512, 512]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        mm_91: "f16[2048, 512]" = torch.ops.aten.mm.default(view_389, permute_179);  view_389 = permute_179 = None
        view_390: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_91, [1, 2048, 512]);  mm_91 = None
        view_391: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_390, [1, 2048, -1, 64]);  view_390 = None
        permute_180: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_183: "f16[1, 8, 64, 2048]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2]);  permute_180 = None
        expand_72: "f16[1, 8, 64, 2048]" = torch.ops.aten.expand.default(permute_183, [1, 8, 64, 2048]);  permute_183 = None
        view_396: "f16[8, 64, 2048]" = torch.ops.aten.reshape.default(expand_72, [8, 64, 2048]);  expand_72 = None
        bmm_34: "f16[8, 2048, 2048]" = torch.ops.aten.bmm.default(view_395, view_396);  view_395 = view_396 = None
        view_397: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_34, [1, 8, 2048, 2048]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_91: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(view_397, add_50);  view_397 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_411: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_91, torch.float32);  add_91 = None
        amax_17: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_411, [-1], True)
        sub_19: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_411, amax_17);  convert_element_type_411 = amax_17 = None
        exp_17: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_18: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_412: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_21, torch.float16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_73: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_412, [1, 8, 2048, 2048]);  convert_element_type_412 = None
        view_400: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_73, [8, 2048, 2048]);  expand_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_392: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, [2048, 512])
        permute_181: "f16[512, 512]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_92: "f16[2048, 512]" = torch.ops.aten.mm.default(view_392, permute_181);  view_392 = permute_181 = None
        view_393: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_92, [1, 2048, 512]);  mm_92 = None
        view_394: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(view_393, [1, 2048, -1, 64]);  view_393 = None
        permute_182: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_74: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_182, [1, 8, 2048, 64]);  permute_182 = None
        view_401: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_74, [8, 2048, 64]);  expand_74 = None
        bmm_35: "f16[8, 2048, 64]" = torch.ops.aten.bmm.default(view_400, view_401);  view_400 = view_401 = None
        view_402: "f16[1, 8, 2048, 64]" = torch.ops.aten.reshape.default(bmm_35, [1, 8, 2048, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_184: "f16[1, 2048, 8, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_77: "f16[1, 2048, 8, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_403: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(clone_77, [1, 2048, -1]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_404: "f16[2048, 512]" = torch.ops.aten.reshape.default(view_403, [2048, 512]);  view_403 = None
        permute_185: "f16[512, 512]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_93: "f16[2048, 512]" = torch.ops.aten.mm.default(view_404, permute_185);  view_404 = permute_185 = None
        view_405: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_93, [1, 2048, 512]);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_92: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_400, view_405);  convert_element_type_400 = view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_417: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_92, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:460 in forward, code: torch.isinf(hidden_states).any(),
        isinf_28: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_92);  add_92 = None
        any_29: "b8[]" = torch.ops.aten.any.default(isinf_28);  isinf_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:459 in forward, code: clamp_value = torch.where(
        full_default_67: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_33: "f32[]" = torch.ops.aten.where.self(any_29, full_default_67, full_default_66);  any_29 = full_default_67 = full_default_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_29: "f32[]" = torch.ops.aten.neg.default(where_33)
        clamp_min_28: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_417, neg_29);  convert_element_type_417 = neg_29 = None
        clamp_max_28: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_28, where_33);  clamp_min_28 = where_33 = None
        convert_element_type_418: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_28, torch.float16);  clamp_max_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_419: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_418, torch.float32)
        pow_31: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_419, 2);  convert_element_type_419 = None
        mean_30: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_93: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_63: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_418, rsqrt_30);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_420: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_63, torch.float16);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_64: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg128_1, convert_element_type_420);  arg128_1 = convert_element_type_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_406: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_64, [2048, 512]);  mul_64 = None
        permute_186: "f16[512, 2048]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        mm_94: "f16[2048, 2048]" = torch.ops.aten.mm.default(view_406, permute_186);  view_406 = permute_186 = None
        view_407: "f16[1, 2048, 2048]" = torch.ops.aten.reshape.default(mm_94, [1, 2048, 2048]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_11: "f16[1, 2048, 2048]" = torch.ops.aten.relu.default(view_407);  view_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_408: "f16[2048, 2048]" = torch.ops.aten.reshape.default(relu_11, [2048, 2048]);  relu_11 = None
        permute_187: "f16[2048, 512]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        mm_95: "f16[2048, 512]" = torch.ops.aten.mm.default(view_408, permute_187);  view_408 = permute_187 = None
        view_409: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_95, [1, 2048, 512]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_94: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_418, view_409);  convert_element_type_418 = view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_425: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_94, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:475 in forward, code: torch.isinf(hidden_states).any(),
        isinf_29: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_94);  add_94 = None
        any_30: "b8[]" = torch.ops.aten.any.default(isinf_29);  isinf_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:474 in forward, code: clamp_value = torch.where(
        full_default_69: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_34: "f32[]" = torch.ops.aten.where.self(any_30, full_default_69, full_default_68);  any_30 = full_default_69 = full_default_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:479 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_30: "f32[]" = torch.ops.aten.neg.default(where_34)
        clamp_min_29: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_425, neg_30);  convert_element_type_425 = neg_30 = None
        clamp_max_29: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_29, where_34);  clamp_min_29 = where_34 = None
        convert_element_type_426: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_29, torch.float16);  clamp_max_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_427: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_426, torch.float32)
        pow_32: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_427, 2);  convert_element_type_427 = None
        mean_31: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_95: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_65: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_426, rsqrt_31);  convert_element_type_426 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_428: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_65, torch.float16);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_66: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg131_1, convert_element_type_428);  arg131_1 = convert_element_type_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_67: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(mul_66, 0.04419417382415922);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        view_410: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_67, [2048, 512]);  mul_67 = None
        permute_188: "f16[512, 32128]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_96: "f16[2048, 32128]" = torch.ops.aten.mm.default(view_410, permute_188);  view_410 = permute_188 = None
        view_411: "f16[1, 2048, 32128]" = torch.ops.aten.reshape.default(mm_96, [1, 2048, 32128]);  mm_96 = None
        return (view_411, mul_27)
