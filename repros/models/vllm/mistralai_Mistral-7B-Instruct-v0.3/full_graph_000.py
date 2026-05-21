class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 512]", arg1_1: "bf16[32768, 4096]", arg2_1: "f32[64]", arg3_1: "bf16[4096]", arg4_1: "bf16[4096, 4096]", arg5_1: "bf16[1024, 4096]", arg6_1: "bf16[1024, 4096]", arg7_1: "bf16[4096, 4096]", arg8_1: "bf16[4096]", arg9_1: "bf16[14336, 4096]", arg10_1: "bf16[14336, 4096]", arg11_1: "bf16[4096, 14336]", arg12_1: "bf16[4096]", arg13_1: "bf16[4096, 4096]", arg14_1: "bf16[1024, 4096]", arg15_1: "bf16[1024, 4096]", arg16_1: "bf16[4096, 4096]", arg17_1: "bf16[4096]", arg18_1: "bf16[14336, 4096]", arg19_1: "bf16[14336, 4096]", arg20_1: "bf16[4096, 14336]", arg21_1: "bf16[4096]", arg22_1: "bf16[4096, 4096]", arg23_1: "bf16[1024, 4096]", arg24_1: "bf16[1024, 4096]", arg25_1: "bf16[4096, 4096]", arg26_1: "bf16[4096]", arg27_1: "bf16[14336, 4096]", arg28_1: "bf16[14336, 4096]", arg29_1: "bf16[4096, 14336]", arg30_1: "bf16[4096]", arg31_1: "bf16[4096, 4096]", arg32_1: "bf16[1024, 4096]", arg33_1: "bf16[1024, 4096]", arg34_1: "bf16[4096, 4096]", arg35_1: "bf16[4096]", arg36_1: "bf16[14336, 4096]", arg37_1: "bf16[14336, 4096]", arg38_1: "bf16[4096, 14336]", arg39_1: "bf16[4096]", arg40_1: "bf16[4096, 4096]", arg41_1: "bf16[1024, 4096]", arg42_1: "bf16[1024, 4096]", arg43_1: "bf16[4096, 4096]", arg44_1: "bf16[4096]", arg45_1: "bf16[14336, 4096]", arg46_1: "bf16[14336, 4096]", arg47_1: "bf16[4096, 14336]", arg48_1: "bf16[4096]", arg49_1: "bf16[4096, 4096]", arg50_1: "bf16[1024, 4096]", arg51_1: "bf16[1024, 4096]", arg52_1: "bf16[4096, 4096]", arg53_1: "bf16[4096]", arg54_1: "bf16[14336, 4096]", arg55_1: "bf16[14336, 4096]", arg56_1: "bf16[4096, 14336]", arg57_1: "bf16[4096]", arg58_1: "bf16[4096, 4096]", arg59_1: "bf16[1024, 4096]", arg60_1: "bf16[1024, 4096]", arg61_1: "bf16[4096, 4096]", arg62_1: "bf16[4096]", arg63_1: "bf16[14336, 4096]", arg64_1: "bf16[14336, 4096]", arg65_1: "bf16[4096, 14336]", arg66_1: "bf16[4096]", arg67_1: "bf16[4096, 4096]", arg68_1: "bf16[1024, 4096]", arg69_1: "bf16[1024, 4096]", arg70_1: "bf16[4096, 4096]", arg71_1: "bf16[4096]", arg72_1: "bf16[14336, 4096]", arg73_1: "bf16[14336, 4096]", arg74_1: "bf16[4096, 14336]", arg75_1: "bf16[4096]", arg76_1: "bf16[32768, 4096]", arg77_1: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:362 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "bf16[4, 512, 4096]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_3: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 2)
        mean: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_3: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_3: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt);  convert_element_type_3 = rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_4: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_4: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg3_1, convert_element_type_4);  arg3_1 = convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_4: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_4, [2048, 4096])
        permute_1: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
        view_5: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm, [4, 512, 4096]);  mm = None
        view_6: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_5, [4, 512, -1, 128]);  view_5 = None
        permute_2: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:314 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        expand_2: "f32[1, 64, 1]" = torch.ops.aten.expand.default(unsqueeze_11, [1, -1, 1]);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:319 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_3: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_2, [1, 64, 1]);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:369 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:370 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:315 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_12: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        convert_element_type: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_12, torch.float32);  unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:319 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_4: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type, [1, 1, 512]);  convert_element_type = None
        mul: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_3, expand_4);  expand_3 = expand_4 = None
        permute: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:320 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_13: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_5: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_13, [1, 512, 2, 64]);  unsqueeze_13 = None
        clone: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_3: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(clone, [1, 512, 128]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:321 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 512, 128]" = torch.ops.aten.cos.default(view_3)
        mul_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:324 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_1: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_14: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_5: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_2, unsqueeze_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_5: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_2, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_5);  slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_4: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_2, 3, 0, 64);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_1: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg, slice_4], -1);  neg = slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:322 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 512, 128]" = torch.ops.aten.sin.default(view_3);  view_3 = None
        mul_2: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:324 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_15: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_6: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_15);  cat_1 = None
        add_4: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_7: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_4, [2048, 4096])
        permute_3: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_7, permute_3);  view_7 = permute_3 = None
        view_8: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_1, [4, 512, 1024]);  mm_1 = None
        view_9: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_8, [4, 512, -1, 128]);  view_8 = None
        permute_4: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_7: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_14);  unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_7: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_4, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_7);  slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_6: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_4, 3, 0, 64);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_2: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_1, slice_6], -1);  neg_1 = slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_8: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_15);  cat_2 = unsqueeze_15 = None
        add_5: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_16: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_5, 2);  add_5 = None
        expand_6: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_16, [4, 8, 4, 512, 128]);  unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_2: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_13: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_2, [4, 32, 512, 128]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_10: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_4, [2048, 4096]);  mul_4 = None
        permute_5: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_2: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
        view_11: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 1024]);  mm_2 = None
        view_12: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_11, [4, 512, -1, 128]);  view_11 = None
        permute_6: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_17: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_6, 2);  permute_6 = None
        expand_7: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_17, [4, 8, 4, 512, 128]);  unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_3: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_14: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_3, [4, 32, 512, 128]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[512]" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_7: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_8: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        unsqueeze_9: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[512]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_4: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_5: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand: "i64[4, 512]" = torch.ops.aten.expand.default(unsqueeze, [4, -1]);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[4, 1]" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub: "i64[4, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[4, 513]" = torch.ops.aten.cat.default([sub, expand], -1);  sub = expand = None
        slice_3: "i64[4, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 513)
        slice_2: "i64[4, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 512);  cat = None
        sub_1: "i64[4, 512]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[4, 512]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[4, 512]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[4, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[4, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[4, 1, 512, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[4, 1, 1, 512]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[4, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[4, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_1, [4, -1, 512, 512]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_4, view_13, view_14, where, False, scale = 0.08838834764831845);  add_4 = view_13 = view_14 = where = None
        getitem: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_7, [4, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_16: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_15, [2048, 4096]);  view_15 = None
        permute_8: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_3: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
        view_17: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 4096]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_6: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(embedding, view_17);  embedding = view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_13: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_2: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_13, 2)
        mean_1: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_7: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_9: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_13, rsqrt_1);  convert_element_type_13 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_14: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        mul_10: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg8_1, convert_element_type_14);  arg8_1 = convert_element_type_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_18: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_10, [2048, 4096])
        permute_9: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_4: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_18, permute_9);  view_18 = permute_9 = None
        view_19: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_4, [4, 512, 14336]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_17: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        neg_2: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_17)
        exp: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_8: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_17, add_8);  convert_element_type_17 = add_8 = None
        convert_element_type_18: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_20: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_10, [2048, 4096]);  mul_10 = None
        permute_10: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_5: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_20, permute_10);  view_20 = permute_10 = None
        view_21: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_5, [4, 512, 14336]);  mm_5 = None
        mul_11: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_18, view_21);  convert_element_type_18 = view_21 = None
        view_22: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_11, [2048, 14336]);  mul_11 = None
        permute_11: "bf16[14336, 4096]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_6: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
        view_23: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_6, [4, 512, 4096]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_9: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_6, view_23);  add_6 = view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_23: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_3: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 2)
        mean_2: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-05);  mean_2 = None
        rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_12: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_23, rsqrt_2);  convert_element_type_23 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_24: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        mul_13: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg12_1, convert_element_type_24);  arg12_1 = convert_element_type_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_24: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_13, [2048, 4096])
        permute_12: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_7: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_24, permute_12);  view_24 = permute_12 = None
        view_25: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_7, [4, 512, 4096]);  mm_7 = None
        view_26: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_25, [4, 512, -1, 128]);  view_25 = None
        permute_13: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_18: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_14: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_13, unsqueeze_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_9: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_13, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_3: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_9);  slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_8: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_13, 3, 0, 64);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_3: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg_3, slice_8], -1);  neg_3 = slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_19: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_15: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_19);  cat_3 = None
        add_11: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_27: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_13, [2048, 4096])
        permute_14: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        mm_8: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_27, permute_14);  view_27 = permute_14 = None
        view_28: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_8, [4, 512, 1024]);  mm_8 = None
        view_29: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_28, [4, 512, -1, 128]);  view_28 = None
        permute_15: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_16: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_15, unsqueeze_18);  unsqueeze_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_11: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_15, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_4: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_11);  slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_10: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_15, 3, 0, 64);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_4: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_4, slice_10], -1);  neg_4 = slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_17: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_19);  cat_4 = unsqueeze_19 = None
        add_12: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_20: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_12, 2);  add_12 = None
        expand_8: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_20, [4, 8, 4, 512, 128]);  unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_4: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_33: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_4, [4, 32, 512, 128]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_30: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_13, [2048, 4096]);  mul_13 = None
        permute_16: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_9: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_30, permute_16);  view_30 = permute_16 = None
        view_31: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_9, [4, 512, 1024]);  mm_9 = None
        view_32: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_31, [4, 512, -1, 128]);  view_31 = None
        permute_17: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_21: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_17, 2);  permute_17 = None
        expand_9: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_21, [4, 8, 4, 512, 128]);  unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_5: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_34: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_5, [4, 32, 512, 128]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_4: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_11, view_33, view_34, where_1, False, scale = 0.08838834764831845);  add_11 = view_33 = view_34 = where_1 = None
        getitem_9: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_35: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_18, [4, 512, -1]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_36: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_35, [2048, 4096]);  view_35 = None
        permute_19: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_10: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_36, permute_19);  view_36 = permute_19 = None
        view_37: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_10, [4, 512, 4096]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_13: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_9, view_37);  add_9 = view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_33: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_4: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_33, 2)
        mean_3: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_14: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-05);  mean_3 = None
        rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_18: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_33, rsqrt_3);  convert_element_type_33 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_34: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        mul_19: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg17_1, convert_element_type_34);  arg17_1 = convert_element_type_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_38: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_19, [2048, 4096])
        permute_20: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_11: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_38, permute_20);  view_38 = permute_20 = None
        view_39: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_11, [4, 512, 14336]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_37: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        neg_5: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_37)
        exp_1: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_15: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_37, add_15);  convert_element_type_37 = add_15 = None
        convert_element_type_38: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_40: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_19, [2048, 4096]);  mul_19 = None
        permute_21: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_12: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_40, permute_21);  view_40 = permute_21 = None
        view_41: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_12, [4, 512, 14336]);  mm_12 = None
        mul_20: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_38, view_41);  convert_element_type_38 = view_41 = None
        view_42: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_20, [2048, 14336]);  mul_20 = None
        permute_22: "bf16[14336, 4096]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_13: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_42, permute_22);  view_42 = permute_22 = None
        view_43: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_13, [4, 512, 4096]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_16: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_13, view_43);  add_13 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_43: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_5: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_43, 2)
        mean_4: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-05);  mean_4 = None
        rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_21: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_43, rsqrt_4);  convert_element_type_43 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_44: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        mul_22: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg21_1, convert_element_type_44);  arg21_1 = convert_element_type_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_44: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_22, [2048, 4096])
        permute_23: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_14: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_44, permute_23);  view_44 = permute_23 = None
        view_45: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_14, [4, 512, 4096]);  mm_14 = None
        view_46: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_45, [4, 512, -1, 128]);  view_45 = None
        permute_24: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_22: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_23: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_24, unsqueeze_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_13: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_24, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_6: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_13);  slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_12: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_24, 3, 0, 64);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_5: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg_6, slice_12], -1);  neg_6 = slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_23: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_24: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_23);  cat_5 = None
        add_18: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_47: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_22, [2048, 4096])
        permute_25: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_15: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_47, permute_25);  view_47 = permute_25 = None
        view_48: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_15, [4, 512, 1024]);  mm_15 = None
        view_49: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_48, [4, 512, -1, 128]);  view_48 = None
        permute_26: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_25: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_26, unsqueeze_22);  unsqueeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_15: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_26, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_7: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_15);  slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_14: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_26, 3, 0, 64);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_6: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_7, slice_14], -1);  neg_7 = slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_26: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_23);  cat_6 = unsqueeze_23 = None
        add_19: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_24: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_19, 2);  add_19 = None
        expand_10: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_24, [4, 8, 4, 512, 128]);  unsqueeze_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_6: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_53: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_6, [4, 32, 512, 128]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_50: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_22, [2048, 4096]);  mul_22 = None
        permute_27: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_16: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_50, permute_27);  view_50 = permute_27 = None
        view_51: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_16, [4, 512, 1024]);  mm_16 = None
        view_52: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_51, [4, 512, -1, 128]);  view_51 = None
        permute_28: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_25: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_28, 2);  permute_28 = None
        expand_11: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_25, [4, 8, 4, 512, 128]);  unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_7: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_54: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_7, [4, 32, 512, 128]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_6: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_18, view_53, view_54, where_2, False, scale = 0.08838834764831845);  add_18 = view_53 = view_54 = where_2 = None
        getitem_18: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_55: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_29, [4, 512, -1]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_56: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_55, [2048, 4096]);  view_55 = None
        permute_30: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_17: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_56, permute_30);  view_56 = permute_30 = None
        view_57: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_17, [4, 512, 4096]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_20: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_16, view_57);  add_16 = view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_53: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_6: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_53, 2)
        mean_5: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_21: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-05);  mean_5 = None
        rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_27: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_53, rsqrt_5);  convert_element_type_53 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_54: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        mul_28: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg26_1, convert_element_type_54);  arg26_1 = convert_element_type_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_58: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_28, [2048, 4096])
        permute_31: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_18: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_58, permute_31);  view_58 = permute_31 = None
        view_59: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_18, [4, 512, 14336]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_57: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        neg_8: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_57)
        exp_2: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_22: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_57, add_22);  convert_element_type_57 = add_22 = None
        convert_element_type_58: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_60: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_28, [2048, 4096]);  mul_28 = None
        permute_32: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_19: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_60, permute_32);  view_60 = permute_32 = None
        view_61: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_19, [4, 512, 14336]);  mm_19 = None
        mul_29: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_58, view_61);  convert_element_type_58 = view_61 = None
        view_62: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_29, [2048, 14336]);  mul_29 = None
        permute_33: "bf16[14336, 4096]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_20: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_62, permute_33);  view_62 = permute_33 = None
        view_63: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_20, [4, 512, 4096]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_23: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_20, view_63);  add_20 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_63: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_7: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_63, 2)
        mean_6: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_24: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-05);  mean_6 = None
        rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_30: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_63, rsqrt_6);  convert_element_type_63 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_64: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        mul_31: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg30_1, convert_element_type_64);  arg30_1 = convert_element_type_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_64: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_31, [2048, 4096])
        permute_34: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_21: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_64, permute_34);  view_64 = permute_34 = None
        view_65: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_21, [4, 512, 4096]);  mm_21 = None
        view_66: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_65, [4, 512, -1, 128]);  view_65 = None
        permute_35: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_26: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_32: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_35, unsqueeze_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_17: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_35, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_9: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_17);  slice_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_16: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_35, 3, 0, 64);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_7: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg_9, slice_16], -1);  neg_9 = slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_27: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_33: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_27);  cat_7 = None
        add_25: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_67: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_31, [2048, 4096])
        permute_36: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        mm_22: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_67, permute_36);  view_67 = permute_36 = None
        view_68: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_22, [4, 512, 1024]);  mm_22 = None
        view_69: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_68, [4, 512, -1, 128]);  view_68 = None
        permute_37: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_34: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_37, unsqueeze_26);  unsqueeze_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_19: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_37, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_10: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_19);  slice_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_18: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_37, 3, 0, 64);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_8: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_10, slice_18], -1);  neg_10 = slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_35: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_27);  cat_8 = unsqueeze_27 = None
        add_26: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_28: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_26, 2);  add_26 = None
        expand_12: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_28, [4, 8, 4, 512, 128]);  unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_8: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_73: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_8, [4, 32, 512, 128]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_70: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_31, [2048, 4096]);  mul_31 = None
        permute_38: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_23: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_70, permute_38);  view_70 = permute_38 = None
        view_71: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_23, [4, 512, 1024]);  mm_23 = None
        view_72: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_71, [4, 512, -1, 128]);  view_71 = None
        permute_39: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_29: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_39, 2);  permute_39 = None
        expand_13: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_29, [4, 8, 4, 512, 128]);  unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_9: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_74: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_9, [4, 32, 512, 128]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_8: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_25, view_73, view_74, where_3, False, scale = 0.08838834764831845);  add_25 = view_73 = view_74 = where_3 = None
        getitem_27: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_75: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_40, [4, 512, -1]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_76: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_75, [2048, 4096]);  view_75 = None
        permute_41: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_24: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_76, permute_41);  view_76 = permute_41 = None
        view_77: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_24, [4, 512, 4096]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_27: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_23, view_77);  add_23 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_73: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_8: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 2)
        mean_7: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_28: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-05);  mean_7 = None
        rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_36: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_73, rsqrt_7);  convert_element_type_73 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_74: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None
        mul_37: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg35_1, convert_element_type_74);  arg35_1 = convert_element_type_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_78: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_37, [2048, 4096])
        permute_42: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_25: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_78, permute_42);  view_78 = permute_42 = None
        view_79: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_25, [4, 512, 14336]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_77: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        neg_11: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_77)
        exp_3: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_29: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_77, add_29);  convert_element_type_77 = add_29 = None
        convert_element_type_78: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_80: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_37, [2048, 4096]);  mul_37 = None
        permute_43: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_26: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_80, permute_43);  view_80 = permute_43 = None
        view_81: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_26, [4, 512, 14336]);  mm_26 = None
        mul_38: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_78, view_81);  convert_element_type_78 = view_81 = None
        view_82: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_38, [2048, 14336]);  mul_38 = None
        permute_44: "bf16[14336, 4096]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_27: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_82, permute_44);  view_82 = permute_44 = None
        view_83: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_27, [4, 512, 4096]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_30: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_27, view_83);  add_27 = view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_83: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_9: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_83, 2)
        mean_8: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_31: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-05);  mean_8 = None
        rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_39: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_83, rsqrt_8);  convert_element_type_83 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_84: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        mul_40: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg39_1, convert_element_type_84);  arg39_1 = convert_element_type_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_84: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_40, [2048, 4096])
        permute_45: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_28: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_84, permute_45);  view_84 = permute_45 = None
        view_85: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_28, [4, 512, 4096]);  mm_28 = None
        view_86: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_85, [4, 512, -1, 128]);  view_85 = None
        permute_46: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_30: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_41: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_46, unsqueeze_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_21: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_46, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_12: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_21);  slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_20: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_46, 3, 0, 64);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_9: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg_12, slice_20], -1);  neg_12 = slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_31: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_42: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_31);  cat_9 = None
        add_32: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_87: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_40, [2048, 4096])
        permute_47: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_29: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_87, permute_47);  view_87 = permute_47 = None
        view_88: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_29, [4, 512, 1024]);  mm_29 = None
        view_89: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_88, [4, 512, -1, 128]);  view_88 = None
        permute_48: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_43: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_48, unsqueeze_30);  unsqueeze_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_23: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_48, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_13: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_23);  slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_22: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_48, 3, 0, 64);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_10: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_13, slice_22], -1);  neg_13 = slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_44: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_31);  cat_10 = unsqueeze_31 = None
        add_33: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_32: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_33, 2);  add_33 = None
        expand_14: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_32, [4, 8, 4, 512, 128]);  unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_10: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_93: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_10, [4, 32, 512, 128]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_90: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_40, [2048, 4096]);  mul_40 = None
        permute_49: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_30: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_90, permute_49);  view_90 = permute_49 = None
        view_91: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_30, [4, 512, 1024]);  mm_30 = None
        view_92: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_91, [4, 512, -1, 128]);  view_91 = None
        permute_50: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_33: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_50, 2);  permute_50 = None
        expand_15: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_33, [4, 8, 4, 512, 128]);  unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_11: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_94: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_11, [4, 32, 512, 128]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_10: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_32, view_93, view_94, where_4, False, scale = 0.08838834764831845);  add_32 = view_93 = view_94 = where_4 = None
        getitem_36: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_95: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_51, [4, 512, -1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_96: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_95, [2048, 4096]);  view_95 = None
        permute_52: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        mm_31: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_96, permute_52);  view_96 = permute_52 = None
        view_97: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_31, [4, 512, 4096]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_34: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_30, view_97);  add_30 = view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_93: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_10: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 2)
        mean_9: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-05);  mean_9 = None
        rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_45: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_93, rsqrt_9);  convert_element_type_93 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_94: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        mul_46: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg44_1, convert_element_type_94);  arg44_1 = convert_element_type_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_98: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_46, [2048, 4096])
        permute_53: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_32: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_98, permute_53);  view_98 = permute_53 = None
        view_99: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_32, [4, 512, 14336]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_97: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        neg_14: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_97)
        exp_4: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_36: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_97, add_36);  convert_element_type_97 = add_36 = None
        convert_element_type_98: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_100: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_46, [2048, 4096]);  mul_46 = None
        permute_54: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_33: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_100, permute_54);  view_100 = permute_54 = None
        view_101: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_33, [4, 512, 14336]);  mm_33 = None
        mul_47: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_98, view_101);  convert_element_type_98 = view_101 = None
        view_102: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_47, [2048, 14336]);  mul_47 = None
        permute_55: "bf16[14336, 4096]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_34: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_102, permute_55);  view_102 = permute_55 = None
        view_103: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_34, [4, 512, 4096]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_37: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_34, view_103);  add_34 = view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_103: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_11: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_103, 2)
        mean_10: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_38: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-05);  mean_10 = None
        rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_48: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_103, rsqrt_10);  convert_element_type_103 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_104: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        mul_49: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg48_1, convert_element_type_104);  arg48_1 = convert_element_type_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_104: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_49, [2048, 4096])
        permute_56: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_35: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_104, permute_56);  view_104 = permute_56 = None
        view_105: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_35, [4, 512, 4096]);  mm_35 = None
        view_106: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_105, [4, 512, -1, 128]);  view_105 = None
        permute_57: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_34: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_50: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_57, unsqueeze_34)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_25: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_57, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_15: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_25);  slice_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_24: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_57, 3, 0, 64);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_11: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg_15, slice_24], -1);  neg_15 = slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_35: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_51: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_11, unsqueeze_35);  cat_11 = None
        add_39: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_107: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_49, [2048, 4096])
        permute_58: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_36: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_107, permute_58);  view_107 = permute_58 = None
        view_108: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_36, [4, 512, 1024]);  mm_36 = None
        view_109: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_108, [4, 512, -1, 128]);  view_108 = None
        permute_59: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_52: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_59, unsqueeze_34);  unsqueeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_27: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_59, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_16: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_27);  slice_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_26: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_59, 3, 0, 64);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_12: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_16, slice_26], -1);  neg_16 = slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_53: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_12, unsqueeze_35);  cat_12 = unsqueeze_35 = None
        add_40: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_36: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_40, 2);  add_40 = None
        expand_16: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_36, [4, 8, 4, 512, 128]);  unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_12: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_113: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_12, [4, 32, 512, 128]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_110: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_49, [2048, 4096]);  mul_49 = None
        permute_60: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_37: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_110, permute_60);  view_110 = permute_60 = None
        view_111: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_37, [4, 512, 1024]);  mm_37 = None
        view_112: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_111, [4, 512, -1, 128]);  view_111 = None
        permute_61: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_37: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_61, 2);  permute_61 = None
        expand_17: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_37, [4, 8, 4, 512, 128]);  unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_13: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_114: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_13, [4, 32, 512, 128]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_12: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_39, view_113, view_114, where_5, False, scale = 0.08838834764831845);  add_39 = view_113 = view_114 = where_5 = None
        getitem_45: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_115: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_62, [4, 512, -1]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_116: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_115, [2048, 4096]);  view_115 = None
        permute_63: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_38: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_116, permute_63);  view_116 = permute_63 = None
        view_117: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_38, [4, 512, 4096]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_41: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_37, view_117);  add_37 = view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_113: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_12: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_113, 2)
        mean_11: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-05);  mean_11 = None
        rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_54: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_113, rsqrt_11);  convert_element_type_113 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_114: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        mul_55: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg53_1, convert_element_type_114);  arg53_1 = convert_element_type_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_118: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_55, [2048, 4096])
        permute_64: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_39: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_118, permute_64);  view_118 = permute_64 = None
        view_119: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_39, [4, 512, 14336]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_117: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        neg_17: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_117)
        exp_5: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_43: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_117, add_43);  convert_element_type_117 = add_43 = None
        convert_element_type_118: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_120: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_55, [2048, 4096]);  mul_55 = None
        permute_65: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_40: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_120, permute_65);  view_120 = permute_65 = None
        view_121: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_40, [4, 512, 14336]);  mm_40 = None
        mul_56: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_118, view_121);  convert_element_type_118 = view_121 = None
        view_122: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_56, [2048, 14336]);  mul_56 = None
        permute_66: "bf16[14336, 4096]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_41: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_122, permute_66);  view_122 = permute_66 = None
        view_123: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_41, [4, 512, 4096]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_44: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_41, view_123);  add_41 = view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_123: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_13: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_123, 2)
        mean_12: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_45: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-05);  mean_12 = None
        rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_57: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_123, rsqrt_12);  convert_element_type_123 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_124: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        mul_58: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg57_1, convert_element_type_124);  arg57_1 = convert_element_type_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_124: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_58, [2048, 4096])
        permute_67: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_42: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_124, permute_67);  view_124 = permute_67 = None
        view_125: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_42, [4, 512, 4096]);  mm_42 = None
        view_126: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_125, [4, 512, -1, 128]);  view_125 = None
        permute_68: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_38: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_59: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_68, unsqueeze_38)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_29: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_68, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_18: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_29);  slice_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_28: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_68, 3, 0, 64);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_13: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg_18, slice_28], -1);  neg_18 = slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_39: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_60: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_39);  cat_13 = None
        add_46: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_127: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_58, [2048, 4096])
        permute_69: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_43: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_127, permute_69);  view_127 = permute_69 = None
        view_128: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_43, [4, 512, 1024]);  mm_43 = None
        view_129: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_128, [4, 512, -1, 128]);  view_128 = None
        permute_70: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_61: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_70, unsqueeze_38);  unsqueeze_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_31: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_70, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_19: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_31);  slice_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_30: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_70, 3, 0, 64);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_14: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_19, slice_30], -1);  neg_19 = slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_62: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_14, unsqueeze_39);  cat_14 = unsqueeze_39 = None
        add_47: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_40: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_47, 2);  add_47 = None
        expand_18: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_40, [4, 8, 4, 512, 128]);  unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_14: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_133: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_14, [4, 32, 512, 128]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_130: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_58, [2048, 4096]);  mul_58 = None
        permute_71: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_44: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_130, permute_71);  view_130 = permute_71 = None
        view_131: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_44, [4, 512, 1024]);  mm_44 = None
        view_132: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_131, [4, 512, -1, 128]);  view_131 = None
        permute_72: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_41: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_72, 2);  permute_72 = None
        expand_19: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_41, [4, 8, 4, 512, 128]);  unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_15: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_134: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_15, [4, 32, 512, 128]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_14: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_46, view_133, view_134, where_6, False, scale = 0.08838834764831845);  add_46 = view_133 = view_134 = where_6 = None
        getitem_54: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_135: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_73, [4, 512, -1]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_136: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_135, [2048, 4096]);  view_135 = None
        permute_74: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_45: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_136, permute_74);  view_136 = permute_74 = None
        view_137: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_45, [4, 512, 4096]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_48: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_44, view_137);  add_44 = view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_133: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_14: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_133, 2)
        mean_13: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-05);  mean_13 = None
        rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_63: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_133, rsqrt_13);  convert_element_type_133 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_134: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None
        mul_64: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg62_1, convert_element_type_134);  arg62_1 = convert_element_type_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_138: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_64, [2048, 4096])
        permute_75: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_46: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_138, permute_75);  view_138 = permute_75 = None
        view_139: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_46, [4, 512, 14336]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_137: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None
        neg_20: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_137)
        exp_6: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_50: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_137, add_50);  convert_element_type_137 = add_50 = None
        convert_element_type_138: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_140: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_64, [2048, 4096]);  mul_64 = None
        permute_76: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_47: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_140, permute_76);  view_140 = permute_76 = None
        view_141: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_47, [4, 512, 14336]);  mm_47 = None
        mul_65: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_138, view_141);  convert_element_type_138 = view_141 = None
        view_142: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_65, [2048, 14336]);  mul_65 = None
        permute_77: "bf16[14336, 4096]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_48: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_142, permute_77);  view_142 = permute_77 = None
        view_143: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_48, [4, 512, 4096]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_51: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_48, view_143);  add_48 = view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_143: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_15: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_143, 2)
        mean_14: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_52: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-05);  mean_14 = None
        rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_66: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_143, rsqrt_14);  convert_element_type_143 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_144: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_66, torch.bfloat16);  mul_66 = None
        mul_67: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg66_1, convert_element_type_144);  arg66_1 = convert_element_type_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_144: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_67, [2048, 4096])
        permute_78: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_49: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_144, permute_78);  view_144 = permute_78 = None
        view_145: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_49, [4, 512, 4096]);  mm_49 = None
        view_146: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_145, [4, 512, -1, 128]);  view_145 = None
        permute_79: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_42: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_68: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_79, unsqueeze_42)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_33: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_79, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_21: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_33);  slice_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_32: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_79, 3, 0, 64);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_15: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg_21, slice_32], -1);  neg_21 = slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_43: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_69: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_15, unsqueeze_43);  cat_15 = None
        add_53: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_147: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_67, [2048, 4096])
        permute_80: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_50: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_147, permute_80);  view_147 = permute_80 = None
        view_148: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_50, [4, 512, 1024]);  mm_50 = None
        view_149: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_148, [4, 512, -1, 128]);  view_148 = None
        permute_81: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_70: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_81, unsqueeze_42);  unsqueeze_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_35: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_81, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_22: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_35);  slice_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_34: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_81, 3, 0, 64);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_16: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_22, slice_34], -1);  neg_22 = slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_71: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_16, unsqueeze_43);  cat_16 = unsqueeze_43 = None
        add_54: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_70, mul_71);  mul_70 = mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_44: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_54, 2);  add_54 = None
        expand_20: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_44, [4, 8, 4, 512, 128]);  unsqueeze_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_16: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_153: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_16, [4, 32, 512, 128]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_150: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_67, [2048, 4096]);  mul_67 = None
        permute_82: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_51: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_150, permute_82);  view_150 = permute_82 = None
        view_151: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_51, [4, 512, 1024]);  mm_51 = None
        view_152: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_151, [4, 512, -1, 128]);  view_151 = None
        permute_83: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_45: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_83, 2);  permute_83 = None
        expand_21: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_45, [4, 8, 4, 512, 128]);  unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_17: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_154: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_17, [4, 32, 512, 128]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_16: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_16, full_default_15);  expand_1 = full_default_16 = full_default_15 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_53, view_153, view_154, where_7, False, scale = 0.08838834764831845);  add_53 = view_153 = view_154 = where_7 = None
        getitem_63: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[4, 513]" = torch.ops.aten.constant_pad_nd.default(arg77_1, [0, 1], -100.0);  arg77_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_36: "i64[4, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_18: "i64[4, 512]" = torch.ops.aten.clone.default(slice_36, memory_format = torch.contiguous_format);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_167: "i64[2048]" = torch.ops.aten.reshape.default(clone_18, [-1]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        ne_2: "b8[2048]" = torch.ops.aten.ne.Scalar(view_167, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_155: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_84, [4, 512, -1]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_156: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_155, [2048, 4096]);  view_155 = None
        permute_85: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_52: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_156, permute_85);  view_156 = permute_85 = None
        view_157: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_52, [4, 512, 4096]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_55: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_51, view_157);  add_51 = view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_153: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_55, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_16: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 2)
        mean_15: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_56: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-05);  mean_15 = None
        rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_72: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_153, rsqrt_15);  convert_element_type_153 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_154: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        mul_73: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg71_1, convert_element_type_154);  arg71_1 = convert_element_type_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_158: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_73, [2048, 4096])
        permute_86: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_53: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_158, permute_86);  view_158 = permute_86 = None
        view_159: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_53, [4, 512, 14336]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_157: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        neg_23: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_157)
        exp_7: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_57: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_157, add_57);  convert_element_type_157 = add_57 = None
        convert_element_type_158: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_160: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_73, [2048, 4096]);  mul_73 = None
        permute_87: "bf16[4096, 14336]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_54: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_160, permute_87);  view_160 = permute_87 = None
        view_161: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_54, [4, 512, 14336]);  mm_54 = None
        mul_74: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_158, view_161);  convert_element_type_158 = view_161 = None
        view_162: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_74, [2048, 14336]);  mul_74 = None
        permute_88: "bf16[14336, 4096]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_55: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_162, permute_88);  view_162 = permute_88 = None
        view_163: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_55, [4, 512, 4096]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_58: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_55, view_163);  add_55 = view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_163: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32);  add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_17: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_163, 2)
        mean_16: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_59: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-05);  mean_16 = None
        rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_75: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_163, rsqrt_16);  convert_element_type_163 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_164: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_75, torch.bfloat16);  mul_75 = None
        mul_76: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(arg75_1, convert_element_type_164);  arg75_1 = convert_element_type_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:460 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_164: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(mul_76, [2048, 4096]);  mul_76 = None
        permute_89: "bf16[4096, 32768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_56: "bf16[2048, 32768]" = torch.ops.aten.mm.default(view_164, permute_89);  view_164 = permute_89 = None
        view_165: "bf16[4, 512, 32768]" = torch.ops.aten.reshape.default(mm_56, [4, 512, 32768]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_167: "f32[4, 512, 32768]" = torch.ops.prims.convert_element_type.default(view_165, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_166: "f32[2048, 32768]" = torch.ops.aten.reshape.default(convert_element_type_167, [-1, 32768]);  convert_element_type_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax: "f32[2048, 1]" = torch.ops.aten.amax.default(view_166, [1], True)
        sub_2: "f32[2048, 32768]" = torch.ops.aten.sub.Tensor(view_166, amax);  view_166 = amax = None
        exp_8: "f32[2048, 32768]" = torch.ops.aten.exp.default(sub_2)
        sum_1: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [1], True);  exp_8 = None
        log: "f32[2048, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_3: "f32[2048, 32768]" = torch.ops.aten.sub.Tensor(sub_2, log);  sub_2 = log = None
        ne_1: "b8[2048]" = torch.ops.aten.ne.Scalar(view_167, -100)
        full_default_17: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "i64[2048]" = torch.ops.aten.where.self(ne_1, view_167, full_default_17);  ne_1 = full_default_17 = None
        unsqueeze_46: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(where_8, 1);  where_8 = None
        gather: "f32[2048, 1]" = torch.ops.aten.gather.default(sub_3, 1, unsqueeze_46);  sub_3 = unsqueeze_46 = None
        squeeze: "f32[2048]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_24: "f32[2048]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_18: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f32[2048]" = torch.ops.aten.where.self(ne_2, neg_24, full_default_18);  ne_2 = neg_24 = full_default_18 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_9);  where_9 = None
        ne_3: "b8[2048]" = torch.ops.aten.ne.Scalar(view_167, -100);  view_167 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
        convert_element_type_168: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        div_8: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_168);  sum_3 = convert_element_type_168 = None
        return (div_8, view_165)
