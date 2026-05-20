class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[8, 1024]", arg1_1: "f32[50257, 768]", arg2_1: "f32[1024, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[2304]", arg6_1: "f32[768, 2304]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[3072]", arg12_1: "f32[768, 3072]", arg13_1: "f32[768]", arg14_1: "f32[3072, 768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[2304]", arg18_1: "f32[768, 2304]", arg19_1: "f32[768]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[3072]", arg24_1: "f32[768, 3072]", arg25_1: "f32[768]", arg26_1: "f32[3072, 768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[2304]", arg30_1: "f32[768, 2304]", arg31_1: "f32[768]", arg32_1: "f32[768, 768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072]", arg36_1: "f32[768, 3072]", arg37_1: "f32[768]", arg38_1: "f32[3072, 768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[2304]", arg42_1: "f32[768, 2304]", arg43_1: "f32[768]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[3072, 768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[2304]", arg54_1: "f32[768, 2304]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[3072]", arg60_1: "f32[768, 3072]", arg61_1: "f32[768]", arg62_1: "f32[3072, 768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[2304]", arg66_1: "f32[768, 2304]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[3072]", arg72_1: "f32[768, 3072]", arg73_1: "f32[768]", arg74_1: "f32[3072, 768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[2304]", arg78_1: "f32[768, 2304]", arg79_1: "f32[768]", arg80_1: "f32[768, 768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[3072]", arg84_1: "f32[768, 3072]", arg85_1: "f32[768]", arg86_1: "f32[3072, 768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[2304]", arg90_1: "f32[768, 2304]", arg91_1: "f32[768]", arg92_1: "f32[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[3072]", arg96_1: "f32[768, 3072]", arg97_1: "f32[768]", arg98_1: "f32[3072, 768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[2304]", arg102_1: "f32[768, 2304]", arg103_1: "f32[768]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[3072]", arg108_1: "f32[768, 3072]", arg109_1: "f32[768]", arg110_1: "f32[3072, 768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[2304]", arg114_1: "f32[768, 2304]", arg115_1: "f32[768]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[3072]", arg120_1: "f32[768, 3072]", arg121_1: "f32[768]", arg122_1: "f32[3072, 768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[2304]", arg126_1: "f32[768, 2304]", arg127_1: "f32[768]", arg128_1: "f32[768, 768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[3072]", arg132_1: "f32[768, 3072]", arg133_1: "f32[768]", arg134_1: "f32[3072, 768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[2304]", arg138_1: "f32[768, 2304]", arg139_1: "f32[768]", arg140_1: "f32[768, 768]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[3072]", arg144_1: "f32[768, 3072]", arg145_1: "f32[768]", arg146_1: "f32[3072, 768]", arg147_1: "f32[768]", arg148_1: "f32[768]", arg149_1: "f32[2, 768]", arg150_1: "i64[8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:581 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:582 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        sub_2: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  getitem_1 = None
        add_4: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_5: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_5, [-1, 768]);  add_5 = None
        addmm: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg5_1, view_1, arg6_1);  arg5_1 = view_1 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_2: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm, [8, 1024, 2304]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_2, 768, 2);  view_2 = None
        getitem_2: "f32[8, 1024, 768]" = split[0]
        getitem_3: "f32[8, 1024, 768]" = split[1]
        getitem_4: "f32[8, 1024, 768]" = split[2];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_5: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, [8, 1024, -1, 64]);  getitem_2 = None
        permute_2: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_3: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_3, [8, 1024, -1, 64]);  getitem_3 = None
        permute: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_4: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [8, 1024, -1, 64]);  getitem_4 = None
        permute_1: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[1024]" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_7: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_8: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        unsqueeze_9: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[1024]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_4: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_5: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand: "i64[8, 1024]" = torch.ops.aten.expand.default(unsqueeze, [8, -1]);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[8, 1]" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub: "i64[8, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[8, 1025]" = torch.ops.aten.cat.default([sub, expand], -1);  sub = expand = None
        slice_3: "i64[8, 1024]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 1025)
        slice_2: "i64[8, 1024]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 1024);  cat = None
        sub_1: "i64[8, 1024]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[8, 1024]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[8, 1, 1024, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[8, 1, 1, 1024]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[8, 1, 1024, 1024]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[8, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(bitwise_and_1, [8, -1, 1024, 1024]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        expand_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where, [8, 12, 1024, 1024]);  where = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand_2, False, scale = 0.125);  permute_2 = permute = permute_1 = expand_2 = None
        getitem_5: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_3, [8, 1024, -1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_6, [-1, 768]);  view_6 = None
        addmm_1: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg7_1, view_7, arg8_1);  arg7_1 = view_7 = arg8_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_8: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_1, [8, 1024, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_6: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_8, add_1);  view_8 = add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_9: "f32[8, 1024, 1]" = var_mean_1[0]
        getitem_10: "f32[8, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_3: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_6, getitem_10);  getitem_10 = None
        add_7: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
        add_8: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_9: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_8, [-1, 768]);  add_8 = None
        addmm_2: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg11_1, view_9, arg12_1);  arg11_1 = view_9 = arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_10: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_2, [8, 1024, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        pow_1: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_5: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_10, mul_5);  view_10 = mul_5 = None
        mul_6: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_10: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_10);  mul_4 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_11: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg13_1, view_11, arg14_1);  arg13_1 = view_11 = arg14_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_12: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_3, [8, 1024, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_11: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_6, view_12);  add_6 = view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_11: "f32[8, 1024, 1]" = var_mean_2[0]
        getitem_12: "f32[8, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_4: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_12);  getitem_12 = None
        add_12: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
        add_13: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_13: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_13, [-1, 768]);  add_13 = None
        addmm_4: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg17_1, view_13, arg18_1);  arg17_1 = view_13 = arg18_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_14: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_4, [8, 1024, 2304]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_14, 768, 2);  view_14 = None
        getitem_13: "f32[8, 1024, 768]" = split_1[0]
        getitem_14: "f32[8, 1024, 768]" = split_1[1]
        getitem_15: "f32[8, 1024, 768]" = split_1[2];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_17: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_13, [8, 1024, -1, 64]);  getitem_13 = None
        permute_6: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_15: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_14, [8, 1024, -1, 64]);  getitem_14 = None
        permute_4: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_16: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_15, [8, 1024, -1, 64]);  getitem_15 = None
        permute_5: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        expand_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_1, [8, 12, 1024, 1024]);  where_1 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_3, False, scale = 0.125);  permute_6 = permute_4 = permute_5 = expand_3 = None
        getitem_16: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_7, [8, 1024, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_18, [-1, 768]);  view_18 = None
        addmm_5: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg19_1, view_19, arg20_1);  arg19_1 = view_19 = arg20_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_20: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_5, [8, 1024, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_14: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_20, add_11);  view_20 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 1024, 1]" = var_mean_3[0]
        getitem_21: "f32[8, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_21);  getitem_21 = None
        add_15: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_11: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_16: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_21: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_16, [-1, 768]);  add_16 = None
        addmm_6: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg23_1, view_21, arg24_1);  arg23_1 = view_21 = arg24_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_22: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_6, [8, 1024, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_2: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_13: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_22, mul_13);  view_22 = mul_13 = None
        mul_14: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_18: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_18);  mul_12 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_23: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg25_1, view_23, arg26_1);  arg25_1 = view_23 = arg26_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_24: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_7, [8, 1024, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_19: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_14, view_24);  add_14 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 1024, 1]" = var_mean_4[0]
        getitem_23: "f32[8, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_23);  getitem_23 = None
        add_20: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_16: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg27_1);  mul_16 = arg27_1 = None
        add_21: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_17, arg28_1);  mul_17 = arg28_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_25: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_21, [-1, 768]);  add_21 = None
        addmm_8: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg29_1, view_25, arg30_1);  arg29_1 = view_25 = arg30_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_26: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_8, [8, 1024, 2304]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_26, 768, 2);  view_26 = None
        getitem_24: "f32[8, 1024, 768]" = split_2[0]
        getitem_25: "f32[8, 1024, 768]" = split_2[1]
        getitem_26: "f32[8, 1024, 768]" = split_2[2];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_29: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_24, [8, 1024, -1, 64]);  getitem_24 = None
        permute_10: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_27: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_25, [8, 1024, -1, 64]);  getitem_25 = None
        permute_8: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_28: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_26, [8, 1024, -1, 64]);  getitem_26 = None
        permute_9: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_6: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        expand_4: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_2, [8, 12, 1024, 1024]);  where_2 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_4, False, scale = 0.125);  permute_10 = permute_8 = permute_9 = expand_4 = None
        getitem_27: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_11, [8, 1024, -1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_30, [-1, 768]);  view_30 = None
        addmm_9: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg31_1, view_31, arg32_1);  arg31_1 = view_31 = arg32_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_32: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_9, [8, 1024, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_22: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_32, add_19);  view_32 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_31: "f32[8, 1024, 1]" = var_mean_5[0]
        getitem_32: "f32[8, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_7: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_32);  getitem_32 = None
        add_23: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_18: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_19: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg33_1);  mul_18 = arg33_1 = None
        add_24: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_19, arg34_1);  mul_19 = arg34_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_33: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_24, [-1, 768]);  add_24 = None
        addmm_10: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg35_1, view_33, arg36_1);  arg35_1 = view_33 = arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_34: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 1024, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        pow_3: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_21: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_34, mul_21);  view_34 = mul_21 = None
        mul_22: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_26: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_26);  mul_20 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_35: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg37_1, view_35, arg38_1);  arg37_1 = view_35 = arg38_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_36: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_11, [8, 1024, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_27: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_22, view_36);  add_22 = view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_33: "f32[8, 1024, 1]" = var_mean_6[0]
        getitem_34: "f32[8, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_8: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_34);  getitem_34 = None
        add_28: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_24: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_25: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg39_1);  mul_24 = arg39_1 = None
        add_29: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_25, arg40_1);  mul_25 = arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_37: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_29, [-1, 768]);  add_29 = None
        addmm_12: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg41_1, view_37, arg42_1);  arg41_1 = view_37 = arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_38: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_12, [8, 1024, 2304]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_38, 768, 2);  view_38 = None
        getitem_35: "f32[8, 1024, 768]" = split_3[0]
        getitem_36: "f32[8, 1024, 768]" = split_3[1]
        getitem_37: "f32[8, 1024, 768]" = split_3[2];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_41: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_35, [8, 1024, -1, 64]);  getitem_35 = None
        permute_14: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_39: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_36, [8, 1024, -1, 64]);  getitem_36 = None
        permute_12: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_40: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_37, [8, 1024, -1, 64]);  getitem_37 = None
        permute_13: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_8: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        expand_5: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_3, [8, 12, 1024, 1024]);  where_3 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_5, False, scale = 0.125);  permute_14 = permute_12 = permute_13 = expand_5 = None
        getitem_38: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_15, [8, 1024, -1]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_42, [-1, 768]);  view_42 = None
        addmm_13: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg43_1, view_43, arg44_1);  arg43_1 = view_43 = arg44_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_44: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_13, [8, 1024, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_30: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_44, add_27);  view_44 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 1024, 1]" = var_mean_7[0]
        getitem_43: "f32[8, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_9: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_43);  getitem_43 = None
        add_31: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_26: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = rsqrt_7 = None
        mul_27: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg45_1);  mul_26 = arg45_1 = None
        add_32: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_27, arg46_1);  mul_27 = arg46_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_45: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_32, [-1, 768]);  add_32 = None
        addmm_14: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg47_1, view_45, arg48_1);  arg47_1 = view_45 = arg48_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_46: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_14, [8, 1024, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        pow_4: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_29: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_33: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_46, mul_29);  view_46 = mul_29 = None
        mul_30: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_3: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_34: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_34);  mul_28 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_47: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg49_1, view_47, arg50_1);  arg49_1 = view_47 = arg50_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_48: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_15, [8, 1024, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_35: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_30, view_48);  add_30 = view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 1024, 1]" = var_mean_8[0]
        getitem_45: "f32[8, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_10: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_45);  getitem_45 = None
        add_36: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_32: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = rsqrt_8 = None
        mul_33: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg51_1);  mul_32 = arg51_1 = None
        add_37: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_33, arg52_1);  mul_33 = arg52_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_49: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_37, [-1, 768]);  add_37 = None
        addmm_16: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg53_1, view_49, arg54_1);  arg53_1 = view_49 = arg54_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_50: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_16, [8, 1024, 2304]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_50, 768, 2);  view_50 = None
        getitem_46: "f32[8, 1024, 768]" = split_4[0]
        getitem_47: "f32[8, 1024, 768]" = split_4[1]
        getitem_48: "f32[8, 1024, 768]" = split_4[2];  split_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_53: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_46, [8, 1024, -1, 64]);  getitem_46 = None
        permute_18: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_51: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_47, [8, 1024, -1, 64]);  getitem_47 = None
        permute_16: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_52: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_48, [8, 1024, -1, 64]);  getitem_48 = None
        permute_17: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_10: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        expand_6: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_4, [8, 12, 1024, 1024]);  where_4 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_6, False, scale = 0.125);  permute_18 = permute_16 = permute_17 = expand_6 = None
        getitem_49: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_19, [8, 1024, -1]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_54, [-1, 768]);  view_54 = None
        addmm_17: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg55_1, view_55, arg56_1);  arg55_1 = view_55 = arg56_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_56: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_17, [8, 1024, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_38: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_56, add_35);  view_56 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_53: "f32[8, 1024, 1]" = var_mean_9[0]
        getitem_54: "f32[8, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_11: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_54);  getitem_54 = None
        add_39: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_34: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = rsqrt_9 = None
        mul_35: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg57_1);  mul_34 = arg57_1 = None
        add_40: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_35, arg58_1);  mul_35 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_57: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_40, [-1, 768]);  add_40 = None
        addmm_18: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg59_1, view_57, arg60_1);  arg59_1 = view_57 = arg60_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_58: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_18, [8, 1024, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        pow_5: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_37: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_58, mul_37);  view_58 = mul_37 = None
        mul_38: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_42: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_42);  mul_36 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_59: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg61_1, view_59, arg62_1);  arg61_1 = view_59 = arg62_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_60: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_19, [8, 1024, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_43: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_38, view_60);  add_38 = view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_55: "f32[8, 1024, 1]" = var_mean_10[0]
        getitem_56: "f32[8, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_12: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_56);  getitem_56 = None
        add_44: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_40: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = rsqrt_10 = None
        mul_41: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg63_1);  mul_40 = arg63_1 = None
        add_45: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_41, arg64_1);  mul_41 = arg64_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_61: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_45, [-1, 768]);  add_45 = None
        addmm_20: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg65_1, view_61, arg66_1);  arg65_1 = view_61 = arg66_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_62: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_20, [8, 1024, 2304]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_62, 768, 2);  view_62 = None
        getitem_57: "f32[8, 1024, 768]" = split_5[0]
        getitem_58: "f32[8, 1024, 768]" = split_5[1]
        getitem_59: "f32[8, 1024, 768]" = split_5[2];  split_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_65: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_57, [8, 1024, -1, 64]);  getitem_57 = None
        permute_22: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_63: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_58, [8, 1024, -1, 64]);  getitem_58 = None
        permute_20: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_64: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_59, [8, 1024, -1, 64]);  getitem_59 = None
        permute_21: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_12: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        expand_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_5, [8, 12, 1024, 1024]);  where_5 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_7, False, scale = 0.125);  permute_22 = permute_20 = permute_21 = expand_7 = None
        getitem_60: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_23, [8, 1024, -1]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_66, [-1, 768]);  view_66 = None
        addmm_21: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg67_1, view_67, arg68_1);  arg67_1 = view_67 = arg68_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_68: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_21, [8, 1024, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_46: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_68, add_43);  view_68 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_64: "f32[8, 1024, 1]" = var_mean_11[0]
        getitem_65: "f32[8, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_13: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_65);  getitem_65 = None
        add_47: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_42: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = rsqrt_11 = None
        mul_43: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg69_1);  mul_42 = arg69_1 = None
        add_48: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_43, arg70_1);  mul_43 = arg70_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_69: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_48, [-1, 768]);  add_48 = None
        addmm_22: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg71_1, view_69, arg72_1);  arg71_1 = view_69 = arg72_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_70: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 1024, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        pow_6: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_45: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_49: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_70, mul_45);  view_70 = mul_45 = None
        mul_46: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_5: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_50: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_50);  mul_44 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_71: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg73_1, view_71, arg74_1);  arg73_1 = view_71 = arg74_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_72: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_23, [8, 1024, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_51: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_46, view_72);  add_46 = view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_66: "f32[8, 1024, 1]" = var_mean_12[0]
        getitem_67: "f32[8, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_14: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_67);  getitem_67 = None
        add_52: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_48: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = rsqrt_12 = None
        mul_49: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_53: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_73: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_53, [-1, 768]);  add_53 = None
        addmm_24: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg77_1, view_73, arg78_1);  arg77_1 = view_73 = arg78_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_74: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_24, [8, 1024, 2304]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_6 = torch.ops.aten.split.Tensor(view_74, 768, 2);  view_74 = None
        getitem_68: "f32[8, 1024, 768]" = split_6[0]
        getitem_69: "f32[8, 1024, 768]" = split_6[1]
        getitem_70: "f32[8, 1024, 768]" = split_6[2];  split_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_77: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_68, [8, 1024, -1, 64]);  getitem_68 = None
        permute_26: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_75: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_69, [8, 1024, -1, 64]);  getitem_69 = None
        permute_24: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_76: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_70, [8, 1024, -1, 64]);  getitem_70 = None
        permute_25: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_14: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        expand_8: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_6, [8, 12, 1024, 1024]);  where_6 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_26, permute_24, permute_25, expand_8, False, scale = 0.125);  permute_26 = permute_24 = permute_25 = expand_8 = None
        getitem_71: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_78: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_27, [8, 1024, -1]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_79: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_78, [-1, 768]);  view_78 = None
        addmm_25: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg79_1, view_79, arg80_1);  arg79_1 = view_79 = arg80_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_80: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_25, [8, 1024, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_54: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_80, add_51);  view_80 = add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
        getitem_75: "f32[8, 1024, 1]" = var_mean_13[0]
        getitem_76: "f32[8, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_15: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_54, getitem_76);  getitem_76 = None
        add_55: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_50: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = rsqrt_13 = None
        mul_51: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg81_1);  mul_50 = arg81_1 = None
        add_56: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_51, arg82_1);  mul_51 = arg82_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_81: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_56, [-1, 768]);  add_56 = None
        addmm_26: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg83_1, view_81, arg84_1);  arg83_1 = view_81 = arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_82: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_26, [8, 1024, 3072]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
        pow_7: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 3.0)
        mul_53: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_57: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_82, mul_53);  view_82 = mul_53 = None
        mul_54: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_6: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_58: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_58);  mul_52 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_83: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_55, [-1, 3072]);  mul_55 = None
        addmm_27: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg85_1, view_83, arg86_1);  arg85_1 = view_83 = arg86_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_84: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_27, [8, 1024, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_59: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_54, view_84);  add_54 = view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_77: "f32[8, 1024, 1]" = var_mean_14[0]
        getitem_78: "f32[8, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_16: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_78);  getitem_78 = None
        add_60: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_56: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = rsqrt_14 = None
        mul_57: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg87_1);  mul_56 = arg87_1 = None
        add_61: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_57, arg88_1);  mul_57 = arg88_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_85: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_61, [-1, 768]);  add_61 = None
        addmm_28: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg89_1, view_85, arg90_1);  arg89_1 = view_85 = arg90_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_86: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_28, [8, 1024, 2304]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_7 = torch.ops.aten.split.Tensor(view_86, 768, 2);  view_86 = None
        getitem_79: "f32[8, 1024, 768]" = split_7[0]
        getitem_80: "f32[8, 1024, 768]" = split_7[1]
        getitem_81: "f32[8, 1024, 768]" = split_7[2];  split_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_89: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_79, [8, 1024, -1, 64]);  getitem_79 = None
        permute_30: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_87: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_80, [8, 1024, -1, 64]);  getitem_80 = None
        permute_28: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_88: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_81, [8, 1024, -1, 64]);  getitem_81 = None
        permute_29: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_16: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        expand_9: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_7, [8, 12, 1024, 1024]);  where_7 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_30, permute_28, permute_29, expand_9, False, scale = 0.125);  permute_30 = permute_28 = permute_29 = expand_9 = None
        getitem_82: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_90: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_31, [8, 1024, -1]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_91: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_90, [-1, 768]);  view_90 = None
        addmm_29: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg91_1, view_91, arg92_1);  arg91_1 = view_91 = arg92_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_92: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_29, [8, 1024, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_62: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_92, add_59);  view_92 = add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
        getitem_86: "f32[8, 1024, 1]" = var_mean_15[0]
        getitem_87: "f32[8, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_17: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_62, getitem_87);  getitem_87 = None
        add_63: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_58: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_15);  sub_17 = rsqrt_15 = None
        mul_59: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg93_1);  mul_58 = arg93_1 = None
        add_64: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_59, arg94_1);  mul_59 = arg94_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_93: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_64, [-1, 768]);  add_64 = None
        addmm_30: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg95_1, view_93, arg96_1);  arg95_1 = view_93 = arg96_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_94: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_30, [8, 1024, 3072]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
        pow_8: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_94, 3.0)
        mul_61: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_65: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_94, mul_61);  view_94 = mul_61 = None
        mul_62: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_7: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_66: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_66);  mul_60 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_95: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_63, [-1, 3072]);  mul_63 = None
        addmm_31: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg97_1, view_95, arg98_1);  arg97_1 = view_95 = arg98_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_96: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_31, [8, 1024, 768]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_67: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_62, view_96);  add_62 = view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_88: "f32[8, 1024, 1]" = var_mean_16[0]
        getitem_89: "f32[8, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_18: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_89);  getitem_89 = None
        add_68: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_64: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_16);  sub_18 = rsqrt_16 = None
        mul_65: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg99_1);  mul_64 = arg99_1 = None
        add_69: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_65, arg100_1);  mul_65 = arg100_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_97: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_69, [-1, 768]);  add_69 = None
        addmm_32: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg101_1, view_97, arg102_1);  arg101_1 = view_97 = arg102_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_98: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_32, [8, 1024, 2304]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_8 = torch.ops.aten.split.Tensor(view_98, 768, 2);  view_98 = None
        getitem_90: "f32[8, 1024, 768]" = split_8[0]
        getitem_91: "f32[8, 1024, 768]" = split_8[1]
        getitem_92: "f32[8, 1024, 768]" = split_8[2];  split_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_101: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_90, [8, 1024, -1, 64]);  getitem_90 = None
        permute_34: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_99: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_91, [8, 1024, -1, 64]);  getitem_91 = None
        permute_32: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_100: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_92, [8, 1024, -1, 64]);  getitem_92 = None
        permute_33: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_18: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_18, full_default_17);  full_default_18 = full_default_17 = None
        expand_10: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_8, [8, 12, 1024, 1024]);  where_8 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, permute_32, permute_33, expand_10, False, scale = 0.125);  permute_34 = permute_32 = permute_33 = expand_10 = None
        getitem_93: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_102: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_35, [8, 1024, -1]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_103: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_102, [-1, 768]);  view_102 = None
        addmm_33: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg103_1, view_103, arg104_1);  arg103_1 = view_103 = arg104_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_104: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_33, [8, 1024, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_70: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_104, add_67);  view_104 = add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_97: "f32[8, 1024, 1]" = var_mean_17[0]
        getitem_98: "f32[8, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_19: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_98);  getitem_98 = None
        add_71: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_66: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_17);  sub_19 = rsqrt_17 = None
        mul_67: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg105_1);  mul_66 = arg105_1 = None
        add_72: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_67, arg106_1);  mul_67 = arg106_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_105: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_72, [-1, 768]);  add_72 = None
        addmm_34: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg107_1, view_105, arg108_1);  arg107_1 = view_105 = arg108_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_106: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 1024, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        pow_9: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 3.0)
        mul_69: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_73: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_106, mul_69);  view_106 = mul_69 = None
        mul_70: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_8: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_74: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_74);  mul_68 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_107: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_71, [-1, 3072]);  mul_71 = None
        addmm_35: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg109_1, view_107, arg110_1);  arg109_1 = view_107 = arg110_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_108: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_35, [8, 1024, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_75: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_70, view_108);  add_70 = view_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_99: "f32[8, 1024, 1]" = var_mean_18[0]
        getitem_100: "f32[8, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_20: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_100);  getitem_100 = None
        add_76: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_72: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_18);  sub_20 = rsqrt_18 = None
        mul_73: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg111_1);  mul_72 = arg111_1 = None
        add_77: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_73, arg112_1);  mul_73 = arg112_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_109: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_77, [-1, 768]);  add_77 = None
        addmm_36: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg113_1, view_109, arg114_1);  arg113_1 = view_109 = arg114_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_110: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_36, [8, 1024, 2304]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_9 = torch.ops.aten.split.Tensor(view_110, 768, 2);  view_110 = None
        getitem_101: "f32[8, 1024, 768]" = split_9[0]
        getitem_102: "f32[8, 1024, 768]" = split_9[1]
        getitem_103: "f32[8, 1024, 768]" = split_9[2];  split_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_113: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_101, [8, 1024, -1, 64]);  getitem_101 = None
        permute_38: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_111: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_102, [8, 1024, -1, 64]);  getitem_102 = None
        permute_36: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_112: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_103, [8, 1024, -1, 64]);  getitem_103 = None
        permute_37: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_20: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_20, full_default_19);  full_default_20 = full_default_19 = None
        expand_11: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_9, [8, 12, 1024, 1024]);  where_9 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_38, permute_36, permute_37, expand_11, False, scale = 0.125);  permute_38 = permute_36 = permute_37 = expand_11 = None
        getitem_104: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_114: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_39, [8, 1024, -1]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_115: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_114, [-1, 768]);  view_114 = None
        addmm_37: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg115_1, view_115, arg116_1);  arg115_1 = view_115 = arg116_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_116: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_37, [8, 1024, 768]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_78: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_116, add_75);  view_116 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_108: "f32[8, 1024, 1]" = var_mean_19[0]
        getitem_109: "f32[8, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_21: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_78, getitem_109);  getitem_109 = None
        add_79: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_74: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_19);  sub_21 = rsqrt_19 = None
        mul_75: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg117_1);  mul_74 = arg117_1 = None
        add_80: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_75, arg118_1);  mul_75 = arg118_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_117: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_80, [-1, 768]);  add_80 = None
        addmm_38: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg119_1, view_117, arg120_1);  arg119_1 = view_117 = arg120_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_118: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_38, [8, 1024, 3072]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        pow_10: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_118, 3.0)
        mul_77: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_81: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_118, mul_77);  view_118 = mul_77 = None
        mul_78: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_9: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_82: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_82);  mul_76 = add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_119: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_79, [-1, 3072]);  mul_79 = None
        addmm_39: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg121_1, view_119, arg122_1);  arg121_1 = view_119 = arg122_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_120: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_39, [8, 1024, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_83: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_78, view_120);  add_78 = view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_110: "f32[8, 1024, 1]" = var_mean_20[0]
        getitem_111: "f32[8, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_22: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_111);  getitem_111 = None
        add_84: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_80: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_20);  sub_22 = rsqrt_20 = None
        mul_81: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg123_1);  mul_80 = arg123_1 = None
        add_85: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_81, arg124_1);  mul_81 = arg124_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_121: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_85, [-1, 768]);  add_85 = None
        addmm_40: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg125_1, view_121, arg126_1);  arg125_1 = view_121 = arg126_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_122: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_40, [8, 1024, 2304]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_10 = torch.ops.aten.split.Tensor(view_122, 768, 2);  view_122 = None
        getitem_112: "f32[8, 1024, 768]" = split_10[0]
        getitem_113: "f32[8, 1024, 768]" = split_10[1]
        getitem_114: "f32[8, 1024, 768]" = split_10[2];  split_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_125: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_112, [8, 1024, -1, 64]);  getitem_112 = None
        permute_42: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_123: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_113, [8, 1024, -1, 64]);  getitem_113 = None
        permute_40: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_124: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_114, [8, 1024, -1, 64]);  getitem_114 = None
        permute_41: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_22: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        expand_12: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_10, [8, 12, 1024, 1024]);  where_10 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_40, permute_41, expand_12, False, scale = 0.125);  permute_42 = permute_40 = permute_41 = expand_12 = None
        getitem_115: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_126: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_43, [8, 1024, -1]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_127: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_126, [-1, 768]);  view_126 = None
        addmm_41: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg127_1, view_127, arg128_1);  arg127_1 = view_127 = arg128_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_128: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_41, [8, 1024, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_86: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_128, add_83);  view_128 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
        getitem_119: "f32[8, 1024, 1]" = var_mean_21[0]
        getitem_120: "f32[8, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_23: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_86, getitem_120);  getitem_120 = None
        add_87: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_82: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_21);  sub_23 = rsqrt_21 = None
        mul_83: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg129_1);  mul_82 = arg129_1 = None
        add_88: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_83, arg130_1);  mul_83 = arg130_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_129: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_88, [-1, 768]);  add_88 = None
        addmm_42: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg131_1, view_129, arg132_1);  arg131_1 = view_129 = arg132_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_130: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_42, [8, 1024, 3072]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.5)
        pow_11: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_130, 3.0)
        mul_85: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_89: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_130, mul_85);  view_130 = mul_85 = None
        mul_86: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_10: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_90: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_90);  mul_84 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_131: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_87, [-1, 3072]);  mul_87 = None
        addmm_43: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg133_1, view_131, arg134_1);  arg133_1 = view_131 = arg134_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_132: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_43, [8, 1024, 768]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_91: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_86, view_132);  add_86 = view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_121: "f32[8, 1024, 1]" = var_mean_22[0]
        getitem_122: "f32[8, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_24: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_122);  getitem_122 = None
        add_92: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_88: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_22);  sub_24 = rsqrt_22 = None
        mul_89: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg135_1);  mul_88 = arg135_1 = None
        add_93: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_89, arg136_1);  mul_89 = arg136_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_133: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_93, [-1, 768]);  add_93 = None
        addmm_44: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg137_1, view_133, arg138_1);  arg137_1 = view_133 = arg138_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_134: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_44, [8, 1024, 2304]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_11 = torch.ops.aten.split.Tensor(view_134, 768, 2);  view_134 = None
        getitem_123: "f32[8, 1024, 768]" = split_11[0]
        getitem_124: "f32[8, 1024, 768]" = split_11[1]
        getitem_125: "f32[8, 1024, 768]" = split_11[2];  split_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_137: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_123, [8, 1024, -1, 64]);  getitem_123 = None
        permute_46: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_135: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_124, [8, 1024, -1, 64]);  getitem_124 = None
        permute_44: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_136: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_125, [8, 1024, -1, 64]);  getitem_125 = None
        permute_45: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_24: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default_24, full_default_23);  expand_1 = full_default_24 = full_default_23 = None
        expand_13: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_11, [8, 12, 1024, 1024]);  where_11 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_46, permute_44, permute_45, expand_13, False, scale = 0.125);  permute_46 = permute_44 = permute_45 = expand_13 = None
        getitem_126: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_138: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_47, [8, 1024, -1]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_139: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_138, [-1, 768]);  view_138 = None
        addmm_45: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg139_1, view_139, arg140_1);  arg139_1 = view_139 = arg140_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_140: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_45, [8, 1024, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_94: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_140, add_91);  view_140 = add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
        getitem_130: "f32[8, 1024, 1]" = var_mean_23[0]
        getitem_131: "f32[8, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_25: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_94, getitem_131);  getitem_131 = None
        add_95: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_90: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_23);  sub_25 = rsqrt_23 = None
        mul_91: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg141_1);  mul_90 = arg141_1 = None
        add_96: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_91, arg142_1);  mul_91 = arg142_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_141: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_96, [-1, 768]);  add_96 = None
        addmm_46: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg143_1, view_141, arg144_1);  arg143_1 = view_141 = arg144_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_142: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_46, [8, 1024, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        pow_12: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 3.0)
        mul_93: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_97: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_142, mul_93);  view_142 = mul_93 = None
        mul_94: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_11: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_98: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_98);  mul_92 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_143: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_95, [-1, 3072]);  mul_95 = None
        addmm_47: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg145_1, view_143, arg146_1);  arg145_1 = view_143 = arg146_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_144: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_47, [8, 1024, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_99: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_94, view_144);  add_94 = view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_132: "f32[8, 1024, 1]" = var_mean_24[0]
        getitem_133: "f32[8, 1024, 1]" = var_mean_24[1];  var_mean_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:963 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        ne_3: "b8[8]" = torch.ops.aten.ne.Scalar(arg150_1, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        sub_26: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_133);  add_99 = getitem_133 = None
        add_100: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_96: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_24);  sub_26 = rsqrt_24 = None
        mul_97: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg147_1);  mul_96 = arg147_1 = None
        add_101: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_97, arg148_1);  mul_97 = arg148_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:920 in forward, code: logits = self.score(hidden_states)
        view_146: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_101, [8192, 768]);  add_101 = None
        permute_48: "f32[768, 2]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm: "f32[8192, 2]" = torch.ops.aten.mm.default(view_146, permute_48);  view_146 = permute_48 = None
        view_147: "f32[8, 1024, 2]" = torch.ops.aten.reshape.default(mm, [8, 1024, 2]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:943 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        iota_6: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:934 in forward, code: token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
        iota_5: "i32[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:933 in forward, code: non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
        ne_1: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg0_1, 0);  arg0_1 = None
        convert_element_type: "i32[8, 1024]" = torch.ops.prims.convert_element_type.default(ne_1, torch.int32);  ne_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:935 in forward, code: last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        mul_98: "i32[8, 1024]" = torch.ops.aten.mul.Tensor(iota_5, convert_element_type);  iota_5 = convert_element_type = None
        argmax: "i64[8]" = torch.ops.aten.argmax.default(mul_98, -1);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:943 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        index_2: "f32[8, 2]" = torch.ops.aten.index.Tensor(view_147, [iota_6, argmax]);  view_147 = iota_6 = argmax = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:963 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        amax: "f32[8, 1]" = torch.ops.aten.amax.default(index_2, [1], True)
        sub_27: "f32[8, 2]" = torch.ops.aten.sub.Tensor(index_2, amax);  amax = None
        exp: "f32[8, 2]" = torch.ops.aten.exp.default(sub_27)
        sum_1: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_28: "f32[8, 2]" = torch.ops.aten.sub.Tensor(sub_27, log);  sub_27 = log = None
        ne_2: "b8[8]" = torch.ops.aten.ne.Scalar(arg150_1, -100)
        full_default_25: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[8]" = torch.ops.aten.where.self(ne_2, arg150_1, full_default_25);  ne_2 = full_default_25 = None
        unsqueeze_10: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather: "f32[8, 1]" = torch.ops.aten.gather.default(sub_28, 1, unsqueeze_10);  sub_28 = unsqueeze_10 = None
        squeeze: "f32[8]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_26: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f32[8]" = torch.ops.aten.where.self(ne_3, neg, full_default_26);  ne_3 = neg = full_default_26 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        ne_4: "b8[8]" = torch.ops.aten.ne.Scalar(arg150_1, -100);  arg150_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_4);  ne_4 = None
        convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_1);  sum_3 = convert_element_type_1 = None
        return (div, index_2)
