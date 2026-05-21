class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 512]", arg1_1: "bf16[128256, 2048]", arg2_1: "f32[32]", arg3_1: "bf16[2048]", arg4_1: "bf16[2048, 2048]", arg5_1: "bf16[512, 2048]", arg6_1: "bf16[512, 2048]", arg7_1: "bf16[2048, 2048]", arg8_1: "bf16[2048]", arg9_1: "bf16[8192, 2048]", arg10_1: "bf16[8192, 2048]", arg11_1: "bf16[2048, 8192]", arg12_1: "bf16[2048]", arg13_1: "bf16[2048, 2048]", arg14_1: "bf16[512, 2048]", arg15_1: "bf16[512, 2048]", arg16_1: "bf16[2048, 2048]", arg17_1: "bf16[2048]", arg18_1: "bf16[8192, 2048]", arg19_1: "bf16[8192, 2048]", arg20_1: "bf16[2048, 8192]", arg21_1: "bf16[2048]", arg22_1: "bf16[2048, 2048]", arg23_1: "bf16[512, 2048]", arg24_1: "bf16[512, 2048]", arg25_1: "bf16[2048, 2048]", arg26_1: "bf16[2048]", arg27_1: "bf16[8192, 2048]", arg28_1: "bf16[8192, 2048]", arg29_1: "bf16[2048, 8192]", arg30_1: "bf16[2048]", arg31_1: "bf16[2048, 2048]", arg32_1: "bf16[512, 2048]", arg33_1: "bf16[512, 2048]", arg34_1: "bf16[2048, 2048]", arg35_1: "bf16[2048]", arg36_1: "bf16[8192, 2048]", arg37_1: "bf16[8192, 2048]", arg38_1: "bf16[2048, 8192]", arg39_1: "bf16[2048]", arg40_1: "bf16[2048, 2048]", arg41_1: "bf16[512, 2048]", arg42_1: "bf16[512, 2048]", arg43_1: "bf16[2048, 2048]", arg44_1: "bf16[2048]", arg45_1: "bf16[8192, 2048]", arg46_1: "bf16[8192, 2048]", arg47_1: "bf16[2048, 8192]", arg48_1: "bf16[2048]", arg49_1: "bf16[2048, 2048]", arg50_1: "bf16[512, 2048]", arg51_1: "bf16[512, 2048]", arg52_1: "bf16[2048, 2048]", arg53_1: "bf16[2048]", arg54_1: "bf16[8192, 2048]", arg55_1: "bf16[8192, 2048]", arg56_1: "bf16[2048, 8192]", arg57_1: "bf16[2048]", arg58_1: "bf16[2048, 2048]", arg59_1: "bf16[512, 2048]", arg60_1: "bf16[512, 2048]", arg61_1: "bf16[2048, 2048]", arg62_1: "bf16[2048]", arg63_1: "bf16[8192, 2048]", arg64_1: "bf16[8192, 2048]", arg65_1: "bf16[2048, 8192]", arg66_1: "bf16[2048]", arg67_1: "bf16[2048, 2048]", arg68_1: "bf16[512, 2048]", arg69_1: "bf16[512, 2048]", arg70_1: "bf16[2048, 2048]", arg71_1: "bf16[2048]", arg72_1: "bf16[8192, 2048]", arg73_1: "bf16[8192, 2048]", arg74_1: "bf16[2048, 8192]", arg75_1: "bf16[2048]", arg76_1: "bf16[2048, 2048]", arg77_1: "bf16[512, 2048]", arg78_1: "bf16[512, 2048]", arg79_1: "bf16[2048, 2048]", arg80_1: "bf16[2048]", arg81_1: "bf16[8192, 2048]", arg82_1: "bf16[8192, 2048]", arg83_1: "bf16[2048, 8192]", arg84_1: "bf16[2048]", arg85_1: "bf16[2048, 2048]", arg86_1: "bf16[512, 2048]", arg87_1: "bf16[512, 2048]", arg88_1: "bf16[2048, 2048]", arg89_1: "bf16[2048]", arg90_1: "bf16[8192, 2048]", arg91_1: "bf16[8192, 2048]", arg92_1: "bf16[2048, 8192]", arg93_1: "bf16[2048]", arg94_1: "bf16[2048, 2048]", arg95_1: "bf16[512, 2048]", arg96_1: "bf16[512, 2048]", arg97_1: "bf16[2048, 2048]", arg98_1: "bf16[2048]", arg99_1: "bf16[8192, 2048]", arg100_1: "bf16[8192, 2048]", arg101_1: "bf16[2048, 8192]", arg102_1: "bf16[2048]", arg103_1: "bf16[2048, 2048]", arg104_1: "bf16[512, 2048]", arg105_1: "bf16[512, 2048]", arg106_1: "bf16[2048, 2048]", arg107_1: "bf16[2048]", arg108_1: "bf16[8192, 2048]", arg109_1: "bf16[8192, 2048]", arg110_1: "bf16[2048, 8192]", arg111_1: "bf16[2048]", arg112_1: "bf16[2048, 2048]", arg113_1: "bf16[512, 2048]", arg114_1: "bf16[512, 2048]", arg115_1: "bf16[2048, 2048]", arg116_1: "bf16[2048]", arg117_1: "bf16[8192, 2048]", arg118_1: "bf16[8192, 2048]", arg119_1: "bf16[2048, 8192]", arg120_1: "bf16[2048]", arg121_1: "bf16[2048, 2048]", arg122_1: "bf16[512, 2048]", arg123_1: "bf16[512, 2048]", arg124_1: "bf16[2048, 2048]", arg125_1: "bf16[2048]", arg126_1: "bf16[8192, 2048]", arg127_1: "bf16[8192, 2048]", arg128_1: "bf16[2048, 8192]", arg129_1: "bf16[2048]", arg130_1: "bf16[2048, 2048]", arg131_1: "bf16[512, 2048]", arg132_1: "bf16[512, 2048]", arg133_1: "bf16[2048, 2048]", arg134_1: "bf16[2048]", arg135_1: "bf16[8192, 2048]", arg136_1: "bf16[8192, 2048]", arg137_1: "bf16[2048, 8192]", arg138_1: "bf16[2048]", arg139_1: "bf16[2048, 2048]", arg140_1: "bf16[512, 2048]", arg141_1: "bf16[512, 2048]", arg142_1: "bf16[2048, 2048]", arg143_1: "bf16[2048]", arg144_1: "bf16[8192, 2048]", arg145_1: "bf16[8192, 2048]", arg146_1: "bf16[2048, 8192]", arg147_1: "bf16[2048]", arg148_1: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:389 in forward, code: inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        embedding: "bf16[4, 512, 2048]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_3: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 2)
        mean: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_3: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_3: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt);  convert_element_type_3 = rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_4: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_4: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg3_1, convert_element_type_4);  arg3_1 = convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_4: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_4, [2048, 2048])
        permute_1: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
        view_5: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm, [4, 512, 2048]);  mm = None
        view_6: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_5, [4, 512, -1, 64]);  view_5 = None
        permute_2: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:125 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_10: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_11: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        expand_2: "f32[1, 32, 1]" = torch.ops.aten.expand.default(unsqueeze_11, [1, -1, 1]);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:130 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_3: "f32[1, 32, 1]" = torch.ops.aten.expand.default(expand_2, [1, 32, 1]);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:396 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:397 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:126 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_12: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        convert_element_type: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_12, torch.float32);  unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:130 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_4: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type, [1, 1, 512]);  convert_element_type = None
        mul: "f32[1, 32, 512]" = torch.ops.aten.mul.Tensor(expand_3, expand_4);  expand_3 = expand_4 = None
        permute: "f32[1, 512, 32]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:131 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_13: "f32[1, 512, 1, 32]" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_5: "f32[1, 512, 2, 32]" = torch.ops.aten.expand.default(unsqueeze_13, [1, 512, 2, 32]);  unsqueeze_13 = None
        clone: "f32[1, 512, 2, 32]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_3: "f32[1, 512, 64]" = torch.ops.aten.reshape.default(clone, [1, 512, 64]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:132 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 512, 64]" = torch.ops.aten.cos.default(view_3)
        mul_1: "f32[1, 512, 64]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:135 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_1: "bf16[1, 512, 64]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_14: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_5: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_2, unsqueeze_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_5: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_2, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_5);  slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_4: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_2, 3, 0, 32);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_1: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg, slice_4], -1);  neg = slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:133 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 512, 64]" = torch.ops.aten.sin.default(view_3);  view_3 = None
        mul_2: "f32[1, 512, 64]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:135 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "bf16[1, 512, 64]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_15: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_6: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_15);  cat_1 = None
        add_4: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_7: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_4, [2048, 2048])
        permute_3: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_7, permute_3);  view_7 = permute_3 = None
        view_8: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_1, [4, 512, 512]);  mm_1 = None
        view_9: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_8, [4, 512, -1, 64]);  view_8 = None
        permute_4: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_7: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_14);  unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_7: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_4, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_7);  slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_6: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_4, 3, 0, 32);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_2: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_1, slice_6], -1);  neg_1 = slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_8: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_15);  cat_2 = unsqueeze_15 = None
        add_5: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_16: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_5, 2);  add_5 = None
        expand_6: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_16, [4, 8, 4, 512, 64]);  unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_2: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_13: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_2, [4, 32, 512, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_10: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_4, [2048, 2048]);  mul_4 = None
        permute_5: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_2: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
        view_11: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 512]);  mm_2 = None
        view_12: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_11, [4, 512, -1, 64]);  view_11 = None
        permute_6: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_17: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_6, 2);  permute_6 = None
        expand_7: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_17, [4, 8, 4, 512, 64]);  unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_3: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_14: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_3, [4, 32, 512, 64]);  clone_3 = None

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
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_4, view_13, view_14, where, False, scale = 0.125);  add_4 = view_13 = view_14 = where = None
        getitem: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_7, [4, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_16: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_15, [2048, 2048]);  view_15 = None
        permute_8: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_3: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
        view_17: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 2048]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_6: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(embedding, view_17);  embedding = view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_13: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_2: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_13, 2)
        mean_1: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_7: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_9: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_13, rsqrt_1);  convert_element_type_13 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_14: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        mul_10: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg8_1, convert_element_type_14);  arg8_1 = convert_element_type_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_18: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_10, [2048, 2048])
        permute_9: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_4: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_18, permute_9);  view_18 = permute_9 = None
        view_19: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_4, [4, 512, 8192]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_17: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        neg_2: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_17)
        exp: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_8: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_17, add_8);  convert_element_type_17 = add_8 = None
        convert_element_type_18: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_20: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_10, [2048, 2048]);  mul_10 = None
        permute_10: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_5: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_20, permute_10);  view_20 = permute_10 = None
        view_21: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_5, [4, 512, 8192]);  mm_5 = None
        mul_11: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_18, view_21);  convert_element_type_18 = view_21 = None
        view_22: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_11, [2048, 8192]);  mul_11 = None
        permute_11: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_6: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
        view_23: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_6, [4, 512, 2048]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_9: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_6, view_23);  add_6 = view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_23: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_3: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 2)
        mean_2: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-05);  mean_2 = None
        rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_12: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_23, rsqrt_2);  convert_element_type_23 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_24: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        mul_13: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg12_1, convert_element_type_24);  arg12_1 = convert_element_type_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_24: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_13, [2048, 2048])
        permute_12: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_7: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_24, permute_12);  view_24 = permute_12 = None
        view_25: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_7, [4, 512, 2048]);  mm_7 = None
        view_26: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_25, [4, 512, -1, 64]);  view_25 = None
        permute_13: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_18: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_14: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_13, unsqueeze_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_9: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_13, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_3: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_9);  slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_8: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_13, 3, 0, 32);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_3: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_3, slice_8], -1);  neg_3 = slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_19: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_15: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_19);  cat_3 = None
        add_11: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_27: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_13, [2048, 2048])
        permute_14: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        mm_8: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_27, permute_14);  view_27 = permute_14 = None
        view_28: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_8, [4, 512, 512]);  mm_8 = None
        view_29: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_28, [4, 512, -1, 64]);  view_28 = None
        permute_15: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_16: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_15, unsqueeze_18);  unsqueeze_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_11: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_15, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_4: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_11);  slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_10: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_15, 3, 0, 32);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_4: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_4, slice_10], -1);  neg_4 = slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_17: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_19);  cat_4 = unsqueeze_19 = None
        add_12: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_20: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_12, 2);  add_12 = None
        expand_8: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_20, [4, 8, 4, 512, 64]);  unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_4: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_33: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_4, [4, 32, 512, 64]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_30: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_13, [2048, 2048]);  mul_13 = None
        permute_16: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_9: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_30, permute_16);  view_30 = permute_16 = None
        view_31: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_9, [4, 512, 512]);  mm_9 = None
        view_32: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_31, [4, 512, -1, 64]);  view_31 = None
        permute_17: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_21: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_17, 2);  permute_17 = None
        expand_9: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_21, [4, 8, 4, 512, 64]);  unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_5: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_34: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_5, [4, 32, 512, 64]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_4: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_11, view_33, view_34, where_1, False, scale = 0.125);  add_11 = view_33 = view_34 = where_1 = None
        getitem_9: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_35: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_18, [4, 512, -1]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_36: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_35, [2048, 2048]);  view_35 = None
        permute_19: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_10: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_36, permute_19);  view_36 = permute_19 = None
        view_37: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_10, [4, 512, 2048]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_13: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_9, view_37);  add_9 = view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_33: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_4: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_33, 2)
        mean_3: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_14: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-05);  mean_3 = None
        rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_18: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_33, rsqrt_3);  convert_element_type_33 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_34: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        mul_19: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg17_1, convert_element_type_34);  arg17_1 = convert_element_type_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_38: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_19, [2048, 2048])
        permute_20: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_11: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_38, permute_20);  view_38 = permute_20 = None
        view_39: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_11, [4, 512, 8192]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_37: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        neg_5: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_37)
        exp_1: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_15: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_37, add_15);  convert_element_type_37 = add_15 = None
        convert_element_type_38: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_40: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_19, [2048, 2048]);  mul_19 = None
        permute_21: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_12: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_40, permute_21);  view_40 = permute_21 = None
        view_41: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_12, [4, 512, 8192]);  mm_12 = None
        mul_20: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_38, view_41);  convert_element_type_38 = view_41 = None
        view_42: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_20, [2048, 8192]);  mul_20 = None
        permute_22: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_13: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_42, permute_22);  view_42 = permute_22 = None
        view_43: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_13, [4, 512, 2048]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_16: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_13, view_43);  add_13 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_43: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_5: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_43, 2)
        mean_4: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-05);  mean_4 = None
        rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_21: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_43, rsqrt_4);  convert_element_type_43 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_44: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        mul_22: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg21_1, convert_element_type_44);  arg21_1 = convert_element_type_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_44: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_22, [2048, 2048])
        permute_23: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_14: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_44, permute_23);  view_44 = permute_23 = None
        view_45: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_14, [4, 512, 2048]);  mm_14 = None
        view_46: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_45, [4, 512, -1, 64]);  view_45 = None
        permute_24: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_22: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_23: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_24, unsqueeze_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_13: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_24, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_6: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_13);  slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_12: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_24, 3, 0, 32);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_5: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_6, slice_12], -1);  neg_6 = slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_23: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_24: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_23);  cat_5 = None
        add_18: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_47: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_22, [2048, 2048])
        permute_25: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_15: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_47, permute_25);  view_47 = permute_25 = None
        view_48: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_15, [4, 512, 512]);  mm_15 = None
        view_49: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_48, [4, 512, -1, 64]);  view_48 = None
        permute_26: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_25: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_26, unsqueeze_22);  unsqueeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_15: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_26, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_7: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_15);  slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_14: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_26, 3, 0, 32);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_6: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_7, slice_14], -1);  neg_7 = slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_26: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_23);  cat_6 = unsqueeze_23 = None
        add_19: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_24: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_19, 2);  add_19 = None
        expand_10: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_24, [4, 8, 4, 512, 64]);  unsqueeze_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_6: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_53: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_6, [4, 32, 512, 64]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_50: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_22, [2048, 2048]);  mul_22 = None
        permute_27: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_16: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_50, permute_27);  view_50 = permute_27 = None
        view_51: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_16, [4, 512, 512]);  mm_16 = None
        view_52: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_51, [4, 512, -1, 64]);  view_51 = None
        permute_28: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_25: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_28, 2);  permute_28 = None
        expand_11: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_25, [4, 8, 4, 512, 64]);  unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_7: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_54: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_7, [4, 32, 512, 64]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_6: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_18, view_53, view_54, where_2, False, scale = 0.125);  add_18 = view_53 = view_54 = where_2 = None
        getitem_18: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_55: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_29, [4, 512, -1]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_56: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_55, [2048, 2048]);  view_55 = None
        permute_30: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_17: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_56, permute_30);  view_56 = permute_30 = None
        view_57: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_17, [4, 512, 2048]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_20: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_16, view_57);  add_16 = view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_53: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_6: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_53, 2)
        mean_5: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_21: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-05);  mean_5 = None
        rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_27: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_53, rsqrt_5);  convert_element_type_53 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_54: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        mul_28: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg26_1, convert_element_type_54);  arg26_1 = convert_element_type_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_58: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_28, [2048, 2048])
        permute_31: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_18: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_58, permute_31);  view_58 = permute_31 = None
        view_59: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_18, [4, 512, 8192]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_57: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        neg_8: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_57)
        exp_2: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_22: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_57, add_22);  convert_element_type_57 = add_22 = None
        convert_element_type_58: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_60: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_28, [2048, 2048]);  mul_28 = None
        permute_32: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_19: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_60, permute_32);  view_60 = permute_32 = None
        view_61: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_19, [4, 512, 8192]);  mm_19 = None
        mul_29: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_58, view_61);  convert_element_type_58 = view_61 = None
        view_62: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_29, [2048, 8192]);  mul_29 = None
        permute_33: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_20: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_62, permute_33);  view_62 = permute_33 = None
        view_63: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_20, [4, 512, 2048]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_23: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_20, view_63);  add_20 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_63: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_7: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_63, 2)
        mean_6: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_24: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-05);  mean_6 = None
        rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_30: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_63, rsqrt_6);  convert_element_type_63 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_64: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        mul_31: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg30_1, convert_element_type_64);  arg30_1 = convert_element_type_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_64: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_31, [2048, 2048])
        permute_34: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_21: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_64, permute_34);  view_64 = permute_34 = None
        view_65: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_21, [4, 512, 2048]);  mm_21 = None
        view_66: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_65, [4, 512, -1, 64]);  view_65 = None
        permute_35: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_26: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_32: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_35, unsqueeze_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_17: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_35, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_9: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_17);  slice_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_16: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_35, 3, 0, 32);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_7: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_9, slice_16], -1);  neg_9 = slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_27: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_33: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_27);  cat_7 = None
        add_25: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_67: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_31, [2048, 2048])
        permute_36: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        mm_22: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_67, permute_36);  view_67 = permute_36 = None
        view_68: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_22, [4, 512, 512]);  mm_22 = None
        view_69: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_68, [4, 512, -1, 64]);  view_68 = None
        permute_37: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_34: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_37, unsqueeze_26);  unsqueeze_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_19: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_37, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_10: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_19);  slice_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_18: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_37, 3, 0, 32);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_8: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_10, slice_18], -1);  neg_10 = slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_35: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_27);  cat_8 = unsqueeze_27 = None
        add_26: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_28: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_26, 2);  add_26 = None
        expand_12: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_28, [4, 8, 4, 512, 64]);  unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_8: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_73: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_8, [4, 32, 512, 64]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_70: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_31, [2048, 2048]);  mul_31 = None
        permute_38: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_23: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_70, permute_38);  view_70 = permute_38 = None
        view_71: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_23, [4, 512, 512]);  mm_23 = None
        view_72: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_71, [4, 512, -1, 64]);  view_71 = None
        permute_39: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_29: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_39, 2);  permute_39 = None
        expand_13: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_29, [4, 8, 4, 512, 64]);  unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_9: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_74: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_9, [4, 32, 512, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_8: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_25, view_73, view_74, where_3, False, scale = 0.125);  add_25 = view_73 = view_74 = where_3 = None
        getitem_27: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_75: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_40, [4, 512, -1]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_76: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_75, [2048, 2048]);  view_75 = None
        permute_41: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_24: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_76, permute_41);  view_76 = permute_41 = None
        view_77: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_24, [4, 512, 2048]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_27: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_23, view_77);  add_23 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_73: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_8: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 2)
        mean_7: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_28: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-05);  mean_7 = None
        rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_36: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_73, rsqrt_7);  convert_element_type_73 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_74: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None
        mul_37: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg35_1, convert_element_type_74);  arg35_1 = convert_element_type_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_78: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_37, [2048, 2048])
        permute_42: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_25: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_78, permute_42);  view_78 = permute_42 = None
        view_79: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_25, [4, 512, 8192]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_77: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        neg_11: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_77)
        exp_3: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_29: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_77, add_29);  convert_element_type_77 = add_29 = None
        convert_element_type_78: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_80: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_37, [2048, 2048]);  mul_37 = None
        permute_43: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_26: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_80, permute_43);  view_80 = permute_43 = None
        view_81: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_26, [4, 512, 8192]);  mm_26 = None
        mul_38: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_78, view_81);  convert_element_type_78 = view_81 = None
        view_82: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_38, [2048, 8192]);  mul_38 = None
        permute_44: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_27: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_82, permute_44);  view_82 = permute_44 = None
        view_83: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_27, [4, 512, 2048]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_30: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_27, view_83);  add_27 = view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_83: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_9: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_83, 2)
        mean_8: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_31: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-05);  mean_8 = None
        rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_39: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_83, rsqrt_8);  convert_element_type_83 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_84: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        mul_40: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg39_1, convert_element_type_84);  arg39_1 = convert_element_type_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_84: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_40, [2048, 2048])
        permute_45: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_28: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_84, permute_45);  view_84 = permute_45 = None
        view_85: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_28, [4, 512, 2048]);  mm_28 = None
        view_86: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_85, [4, 512, -1, 64]);  view_85 = None
        permute_46: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_30: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_41: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_46, unsqueeze_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_21: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_46, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_12: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_21);  slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_20: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_46, 3, 0, 32);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_9: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_12, slice_20], -1);  neg_12 = slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_31: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_42: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_31);  cat_9 = None
        add_32: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_87: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_40, [2048, 2048])
        permute_47: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_29: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_87, permute_47);  view_87 = permute_47 = None
        view_88: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_29, [4, 512, 512]);  mm_29 = None
        view_89: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_88, [4, 512, -1, 64]);  view_88 = None
        permute_48: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_43: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_48, unsqueeze_30);  unsqueeze_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_23: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_48, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_13: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_23);  slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_22: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_48, 3, 0, 32);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_10: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_13, slice_22], -1);  neg_13 = slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_44: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_31);  cat_10 = unsqueeze_31 = None
        add_33: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_32: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_33, 2);  add_33 = None
        expand_14: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_32, [4, 8, 4, 512, 64]);  unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_10: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_93: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_10, [4, 32, 512, 64]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_90: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_40, [2048, 2048]);  mul_40 = None
        permute_49: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_30: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_90, permute_49);  view_90 = permute_49 = None
        view_91: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_30, [4, 512, 512]);  mm_30 = None
        view_92: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_91, [4, 512, -1, 64]);  view_91 = None
        permute_50: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_33: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_50, 2);  permute_50 = None
        expand_15: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_33, [4, 8, 4, 512, 64]);  unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_11: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_94: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_11, [4, 32, 512, 64]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_10: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_32, view_93, view_94, where_4, False, scale = 0.125);  add_32 = view_93 = view_94 = where_4 = None
        getitem_36: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_95: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_51, [4, 512, -1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_96: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_95, [2048, 2048]);  view_95 = None
        permute_52: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        mm_31: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_96, permute_52);  view_96 = permute_52 = None
        view_97: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_31, [4, 512, 2048]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_34: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_30, view_97);  add_30 = view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_93: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_10: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 2)
        mean_9: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-05);  mean_9 = None
        rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_45: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_93, rsqrt_9);  convert_element_type_93 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_94: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        mul_46: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg44_1, convert_element_type_94);  arg44_1 = convert_element_type_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_98: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_46, [2048, 2048])
        permute_53: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_32: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_98, permute_53);  view_98 = permute_53 = None
        view_99: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_32, [4, 512, 8192]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_97: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        neg_14: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_97)
        exp_4: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_36: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_97, add_36);  convert_element_type_97 = add_36 = None
        convert_element_type_98: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_100: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_46, [2048, 2048]);  mul_46 = None
        permute_54: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_33: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_100, permute_54);  view_100 = permute_54 = None
        view_101: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_33, [4, 512, 8192]);  mm_33 = None
        mul_47: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_98, view_101);  convert_element_type_98 = view_101 = None
        view_102: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_47, [2048, 8192]);  mul_47 = None
        permute_55: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_34: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_102, permute_55);  view_102 = permute_55 = None
        view_103: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_34, [4, 512, 2048]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_37: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_34, view_103);  add_34 = view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_103: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_11: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_103, 2)
        mean_10: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_38: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-05);  mean_10 = None
        rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_48: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_103, rsqrt_10);  convert_element_type_103 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_104: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        mul_49: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg48_1, convert_element_type_104);  arg48_1 = convert_element_type_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_104: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_49, [2048, 2048])
        permute_56: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_35: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_104, permute_56);  view_104 = permute_56 = None
        view_105: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_35, [4, 512, 2048]);  mm_35 = None
        view_106: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_105, [4, 512, -1, 64]);  view_105 = None
        permute_57: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_34: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_50: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_57, unsqueeze_34)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_25: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_57, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_15: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_25);  slice_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_24: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_57, 3, 0, 32);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_11: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_15, slice_24], -1);  neg_15 = slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_35: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_51: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_11, unsqueeze_35);  cat_11 = None
        add_39: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_107: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_49, [2048, 2048])
        permute_58: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_36: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_107, permute_58);  view_107 = permute_58 = None
        view_108: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_36, [4, 512, 512]);  mm_36 = None
        view_109: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_108, [4, 512, -1, 64]);  view_108 = None
        permute_59: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_52: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_59, unsqueeze_34);  unsqueeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_27: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_59, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_16: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_27);  slice_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_26: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_59, 3, 0, 32);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_12: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_16, slice_26], -1);  neg_16 = slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_53: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_12, unsqueeze_35);  cat_12 = unsqueeze_35 = None
        add_40: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_36: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_40, 2);  add_40 = None
        expand_16: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_36, [4, 8, 4, 512, 64]);  unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_12: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_113: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_12, [4, 32, 512, 64]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_110: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_49, [2048, 2048]);  mul_49 = None
        permute_60: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_37: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_110, permute_60);  view_110 = permute_60 = None
        view_111: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_37, [4, 512, 512]);  mm_37 = None
        view_112: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_111, [4, 512, -1, 64]);  view_111 = None
        permute_61: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_37: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_61, 2);  permute_61 = None
        expand_17: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_37, [4, 8, 4, 512, 64]);  unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_13: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_114: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_13, [4, 32, 512, 64]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_12: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_39, view_113, view_114, where_5, False, scale = 0.125);  add_39 = view_113 = view_114 = where_5 = None
        getitem_45: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_115: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_62, [4, 512, -1]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_116: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_115, [2048, 2048]);  view_115 = None
        permute_63: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_38: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_116, permute_63);  view_116 = permute_63 = None
        view_117: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_38, [4, 512, 2048]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_41: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_37, view_117);  add_37 = view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_113: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_12: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_113, 2)
        mean_11: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-05);  mean_11 = None
        rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_54: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_113, rsqrt_11);  convert_element_type_113 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_114: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        mul_55: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg53_1, convert_element_type_114);  arg53_1 = convert_element_type_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_118: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_55, [2048, 2048])
        permute_64: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_39: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_118, permute_64);  view_118 = permute_64 = None
        view_119: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_39, [4, 512, 8192]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_117: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        neg_17: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_117)
        exp_5: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_43: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_117, add_43);  convert_element_type_117 = add_43 = None
        convert_element_type_118: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_120: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_55, [2048, 2048]);  mul_55 = None
        permute_65: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_40: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_120, permute_65);  view_120 = permute_65 = None
        view_121: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_40, [4, 512, 8192]);  mm_40 = None
        mul_56: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_118, view_121);  convert_element_type_118 = view_121 = None
        view_122: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_56, [2048, 8192]);  mul_56 = None
        permute_66: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_41: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_122, permute_66);  view_122 = permute_66 = None
        view_123: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_41, [4, 512, 2048]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_44: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_41, view_123);  add_41 = view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_123: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_13: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_123, 2)
        mean_12: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_45: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-05);  mean_12 = None
        rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_57: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_123, rsqrt_12);  convert_element_type_123 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_124: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        mul_58: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg57_1, convert_element_type_124);  arg57_1 = convert_element_type_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_124: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_58, [2048, 2048])
        permute_67: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_42: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_124, permute_67);  view_124 = permute_67 = None
        view_125: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_42, [4, 512, 2048]);  mm_42 = None
        view_126: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_125, [4, 512, -1, 64]);  view_125 = None
        permute_68: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_38: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_59: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_68, unsqueeze_38)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_29: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_68, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_18: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_29);  slice_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_28: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_68, 3, 0, 32);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_13: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_18, slice_28], -1);  neg_18 = slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_39: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_60: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_39);  cat_13 = None
        add_46: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_127: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_58, [2048, 2048])
        permute_69: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_43: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_127, permute_69);  view_127 = permute_69 = None
        view_128: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_43, [4, 512, 512]);  mm_43 = None
        view_129: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_128, [4, 512, -1, 64]);  view_128 = None
        permute_70: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_61: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_70, unsqueeze_38);  unsqueeze_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_31: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_70, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_19: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_31);  slice_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_30: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_70, 3, 0, 32);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_14: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_19, slice_30], -1);  neg_19 = slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_62: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_14, unsqueeze_39);  cat_14 = unsqueeze_39 = None
        add_47: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_40: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_47, 2);  add_47 = None
        expand_18: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_40, [4, 8, 4, 512, 64]);  unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_14: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_133: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_14, [4, 32, 512, 64]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_130: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_58, [2048, 2048]);  mul_58 = None
        permute_71: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_44: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_130, permute_71);  view_130 = permute_71 = None
        view_131: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_44, [4, 512, 512]);  mm_44 = None
        view_132: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_131, [4, 512, -1, 64]);  view_131 = None
        permute_72: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_41: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_72, 2);  permute_72 = None
        expand_19: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_41, [4, 8, 4, 512, 64]);  unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_15: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_134: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_15, [4, 32, 512, 64]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_14: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_46, view_133, view_134, where_6, False, scale = 0.125);  add_46 = view_133 = view_134 = where_6 = None
        getitem_54: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_135: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_73, [4, 512, -1]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_136: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_135, [2048, 2048]);  view_135 = None
        permute_74: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_45: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_136, permute_74);  view_136 = permute_74 = None
        view_137: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_45, [4, 512, 2048]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_48: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_44, view_137);  add_44 = view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_133: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_14: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_133, 2)
        mean_13: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-05);  mean_13 = None
        rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_63: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_133, rsqrt_13);  convert_element_type_133 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_134: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None
        mul_64: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg62_1, convert_element_type_134);  arg62_1 = convert_element_type_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_138: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_64, [2048, 2048])
        permute_75: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_46: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_138, permute_75);  view_138 = permute_75 = None
        view_139: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_46, [4, 512, 8192]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_137: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None
        neg_20: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_137)
        exp_6: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_50: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_137, add_50);  convert_element_type_137 = add_50 = None
        convert_element_type_138: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_140: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_64, [2048, 2048]);  mul_64 = None
        permute_76: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_47: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_140, permute_76);  view_140 = permute_76 = None
        view_141: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_47, [4, 512, 8192]);  mm_47 = None
        mul_65: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_138, view_141);  convert_element_type_138 = view_141 = None
        view_142: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_65, [2048, 8192]);  mul_65 = None
        permute_77: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_48: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_142, permute_77);  view_142 = permute_77 = None
        view_143: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_48, [4, 512, 2048]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_51: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_48, view_143);  add_48 = view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_143: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_15: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_143, 2)
        mean_14: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_52: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-05);  mean_14 = None
        rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_66: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_143, rsqrt_14);  convert_element_type_143 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_144: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_66, torch.bfloat16);  mul_66 = None
        mul_67: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg66_1, convert_element_type_144);  arg66_1 = convert_element_type_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_144: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_67, [2048, 2048])
        permute_78: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_49: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_144, permute_78);  view_144 = permute_78 = None
        view_145: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_49, [4, 512, 2048]);  mm_49 = None
        view_146: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_145, [4, 512, -1, 64]);  view_145 = None
        permute_79: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_42: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_68: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_79, unsqueeze_42)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_33: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_79, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_21: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_33);  slice_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_32: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_79, 3, 0, 32);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_15: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_21, slice_32], -1);  neg_21 = slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_43: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_69: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_15, unsqueeze_43);  cat_15 = None
        add_53: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_147: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_67, [2048, 2048])
        permute_80: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_50: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_147, permute_80);  view_147 = permute_80 = None
        view_148: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_50, [4, 512, 512]);  mm_50 = None
        view_149: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_148, [4, 512, -1, 64]);  view_148 = None
        permute_81: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_70: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_81, unsqueeze_42);  unsqueeze_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_35: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_81, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_22: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_35);  slice_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_34: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_81, 3, 0, 32);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_16: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_22, slice_34], -1);  neg_22 = slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_71: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_16, unsqueeze_43);  cat_16 = unsqueeze_43 = None
        add_54: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_70, mul_71);  mul_70 = mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_44: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_54, 2);  add_54 = None
        expand_20: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_44, [4, 8, 4, 512, 64]);  unsqueeze_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_16: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_153: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_16, [4, 32, 512, 64]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_150: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_67, [2048, 2048]);  mul_67 = None
        permute_82: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_51: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_150, permute_82);  view_150 = permute_82 = None
        view_151: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_51, [4, 512, 512]);  mm_51 = None
        view_152: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_151, [4, 512, -1, 64]);  view_151 = None
        permute_83: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_45: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_83, 2);  permute_83 = None
        expand_21: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_45, [4, 8, 4, 512, 64]);  unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_17: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_154: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_17, [4, 32, 512, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_16: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_53, view_153, view_154, where_7, False, scale = 0.125);  add_53 = view_153 = view_154 = where_7 = None
        getitem_63: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_155: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_84, [4, 512, -1]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_156: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_155, [2048, 2048]);  view_155 = None
        permute_85: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_52: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_156, permute_85);  view_156 = permute_85 = None
        view_157: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_52, [4, 512, 2048]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_55: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_51, view_157);  add_51 = view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_153: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_55, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_16: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 2)
        mean_15: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_56: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-05);  mean_15 = None
        rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_72: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_153, rsqrt_15);  convert_element_type_153 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_154: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        mul_73: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg71_1, convert_element_type_154);  arg71_1 = convert_element_type_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_158: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_73, [2048, 2048])
        permute_86: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_53: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_158, permute_86);  view_158 = permute_86 = None
        view_159: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_53, [4, 512, 8192]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_157: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        neg_23: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_157)
        exp_7: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_57: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_157, add_57);  convert_element_type_157 = add_57 = None
        convert_element_type_158: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_160: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_73, [2048, 2048]);  mul_73 = None
        permute_87: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_54: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_160, permute_87);  view_160 = permute_87 = None
        view_161: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_54, [4, 512, 8192]);  mm_54 = None
        mul_74: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_158, view_161);  convert_element_type_158 = view_161 = None
        view_162: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_74, [2048, 8192]);  mul_74 = None
        permute_88: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_55: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_162, permute_88);  view_162 = permute_88 = None
        view_163: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_55, [4, 512, 2048]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_58: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_55, view_163);  add_55 = view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_163: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_17: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_163, 2)
        mean_16: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_59: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-05);  mean_16 = None
        rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_75: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_163, rsqrt_16);  convert_element_type_163 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_164: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_75, torch.bfloat16);  mul_75 = None
        mul_76: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg75_1, convert_element_type_164);  arg75_1 = convert_element_type_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_164: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_76, [2048, 2048])
        permute_89: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_56: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_164, permute_89);  view_164 = permute_89 = None
        view_165: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_56, [4, 512, 2048]);  mm_56 = None
        view_166: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_165, [4, 512, -1, 64]);  view_165 = None
        permute_90: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_46: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_77: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_90, unsqueeze_46)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_37: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_90, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_24: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_37);  slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_36: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_90, 3, 0, 32);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_17: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_24, slice_36], -1);  neg_24 = slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_47: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_78: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_17, unsqueeze_47);  cat_17 = None
        add_60: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_167: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_76, [2048, 2048])
        permute_91: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        mm_57: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_167, permute_91);  view_167 = permute_91 = None
        view_168: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_57, [4, 512, 512]);  mm_57 = None
        view_169: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_168, [4, 512, -1, 64]);  view_168 = None
        permute_92: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_79: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_92, unsqueeze_46);  unsqueeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_39: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_92, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_25: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_39);  slice_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_38: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_92, 3, 0, 32);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_18: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_25, slice_38], -1);  neg_25 = slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_80: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_18, unsqueeze_47);  cat_18 = unsqueeze_47 = None
        add_61: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_79, mul_80);  mul_79 = mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_48: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_61, 2);  add_61 = None
        expand_22: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_48, [4, 8, 4, 512, 64]);  unsqueeze_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_18: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_173: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_18, [4, 32, 512, 64]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_170: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_76, [2048, 2048]);  mul_76 = None
        permute_93: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_58: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_170, permute_93);  view_170 = permute_93 = None
        view_171: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_58, [4, 512, 512]);  mm_58 = None
        view_172: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_171, [4, 512, -1, 64]);  view_171 = None
        permute_94: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_49: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_94, 2);  permute_94 = None
        expand_23: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_49, [4, 8, 4, 512, 64]);  unsqueeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_19: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_174: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_19, [4, 32, 512, 64]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_18: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_18, full_default_17);  full_default_18 = full_default_17 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_60, view_173, view_174, where_8, False, scale = 0.125);  add_60 = view_173 = view_174 = where_8 = None
        getitem_72: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_175: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_95, [4, 512, -1]);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_176: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_175, [2048, 2048]);  view_175 = None
        permute_96: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_59: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_176, permute_96);  view_176 = permute_96 = None
        view_177: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_59, [4, 512, 2048]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_62: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_58, view_177);  add_58 = view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_173: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_18: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_173, 2)
        mean_17: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-05);  mean_17 = None
        rsqrt_17: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_81: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_173, rsqrt_17);  convert_element_type_173 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_174: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None
        mul_82: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg80_1, convert_element_type_174);  arg80_1 = convert_element_type_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_178: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_82, [2048, 2048])
        permute_97: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_60: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_178, permute_97);  view_178 = permute_97 = None
        view_179: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_60, [4, 512, 8192]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_177: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_179, torch.float32);  view_179 = None
        neg_26: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_177)
        exp_8: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_64: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_177, add_64);  convert_element_type_177 = add_64 = None
        convert_element_type_178: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_180: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_82, [2048, 2048]);  mul_82 = None
        permute_98: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_61: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_180, permute_98);  view_180 = permute_98 = None
        view_181: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_61, [4, 512, 8192]);  mm_61 = None
        mul_83: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_178, view_181);  convert_element_type_178 = view_181 = None
        view_182: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_83, [2048, 8192]);  mul_83 = None
        permute_99: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_62: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_182, permute_99);  view_182 = permute_99 = None
        view_183: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_62, [4, 512, 2048]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_65: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_62, view_183);  add_62 = view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_183: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_65, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_19: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_183, 2)
        mean_18: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_66: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-05);  mean_18 = None
        rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_84: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_183, rsqrt_18);  convert_element_type_183 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_184: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None
        mul_85: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg84_1, convert_element_type_184);  arg84_1 = convert_element_type_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_184: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_85, [2048, 2048])
        permute_100: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_63: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_184, permute_100);  view_184 = permute_100 = None
        view_185: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_63, [4, 512, 2048]);  mm_63 = None
        view_186: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_185, [4, 512, -1, 64]);  view_185 = None
        permute_101: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_50: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_86: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_101, unsqueeze_50)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_41: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_101, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_27: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_41);  slice_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_40: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_101, 3, 0, 32);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_19: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_27, slice_40], -1);  neg_27 = slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_51: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_87: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_19, unsqueeze_51);  cat_19 = None
        add_67: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_86, mul_87);  mul_86 = mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_187: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_85, [2048, 2048])
        permute_102: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_64: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_187, permute_102);  view_187 = permute_102 = None
        view_188: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_64, [4, 512, 512]);  mm_64 = None
        view_189: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_188, [4, 512, -1, 64]);  view_188 = None
        permute_103: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_88: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_103, unsqueeze_50);  unsqueeze_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_43: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_103, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_28: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_43);  slice_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_42: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_103, 3, 0, 32);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_20: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_28, slice_42], -1);  neg_28 = slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_89: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_20, unsqueeze_51);  cat_20 = unsqueeze_51 = None
        add_68: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_52: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_68, 2);  add_68 = None
        expand_24: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_52, [4, 8, 4, 512, 64]);  unsqueeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_20: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_193: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_20, [4, 32, 512, 64]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_190: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_85, [2048, 2048]);  mul_85 = None
        permute_104: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_65: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_190, permute_104);  view_190 = permute_104 = None
        view_191: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_65, [4, 512, 512]);  mm_65 = None
        view_192: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_191, [4, 512, -1, 64]);  view_191 = None
        permute_105: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_53: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_105, 2);  permute_105 = None
        expand_25: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_53, [4, 8, 4, 512, 64]);  unsqueeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_21: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_194: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_21, [4, 32, 512, 64]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_20: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_20, full_default_19);  full_default_20 = full_default_19 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_67, view_193, view_194, where_9, False, scale = 0.125);  add_67 = view_193 = view_194 = where_9 = None
        getitem_81: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3]);  getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_195: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_106, [4, 512, -1]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_196: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_195, [2048, 2048]);  view_195 = None
        permute_107: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        mm_66: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_196, permute_107);  view_196 = permute_107 = None
        view_197: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_66, [4, 512, 2048]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_69: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_65, view_197);  add_65 = view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_193: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_69, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_20: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_193, 2)
        mean_19: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_70: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-05);  mean_19 = None
        rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_90: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_193, rsqrt_19);  convert_element_type_193 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_194: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None
        mul_91: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg89_1, convert_element_type_194);  arg89_1 = convert_element_type_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_198: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_91, [2048, 2048])
        permute_108: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_67: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_198, permute_108);  view_198 = permute_108 = None
        view_199: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_67, [4, 512, 8192]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_197: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_199, torch.float32);  view_199 = None
        neg_29: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_197)
        exp_9: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_71: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_197, add_71);  convert_element_type_197 = add_71 = None
        convert_element_type_198: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_200: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_91, [2048, 2048]);  mul_91 = None
        permute_109: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        mm_68: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_200, permute_109);  view_200 = permute_109 = None
        view_201: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_68, [4, 512, 8192]);  mm_68 = None
        mul_92: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_198, view_201);  convert_element_type_198 = view_201 = None
        view_202: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_92, [2048, 8192]);  mul_92 = None
        permute_110: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        mm_69: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_202, permute_110);  view_202 = permute_110 = None
        view_203: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_69, [4, 512, 2048]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_72: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_69, view_203);  add_69 = view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_203: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_72, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_21: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_203, 2)
        mean_20: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_73: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-05);  mean_20 = None
        rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_93: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_203, rsqrt_20);  convert_element_type_203 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_204: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        mul_94: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg93_1, convert_element_type_204);  arg93_1 = convert_element_type_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_204: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_94, [2048, 2048])
        permute_111: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_70: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_204, permute_111);  view_204 = permute_111 = None
        view_205: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_70, [4, 512, 2048]);  mm_70 = None
        view_206: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_205, [4, 512, -1, 64]);  view_205 = None
        permute_112: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_54: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_95: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_112, unsqueeze_54)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_45: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_112, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_30: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_45);  slice_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_44: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_112, 3, 0, 32);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_21: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_30, slice_44], -1);  neg_30 = slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_55: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_96: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_21, unsqueeze_55);  cat_21 = None
        add_74: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_207: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_94, [2048, 2048])
        permute_113: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_71: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_207, permute_113);  view_207 = permute_113 = None
        view_208: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_71, [4, 512, 512]);  mm_71 = None
        view_209: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_208, [4, 512, -1, 64]);  view_208 = None
        permute_114: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_97: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_114, unsqueeze_54);  unsqueeze_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_47: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_114, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_31: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_47);  slice_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_46: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_114, 3, 0, 32);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_22: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_31, slice_46], -1);  neg_31 = slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_98: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_22, unsqueeze_55);  cat_22 = unsqueeze_55 = None
        add_75: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_56: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_75, 2);  add_75 = None
        expand_26: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_56, [4, 8, 4, 512, 64]);  unsqueeze_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_22: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_213: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_22, [4, 32, 512, 64]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_210: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_94, [2048, 2048]);  mul_94 = None
        permute_115: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_72: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_210, permute_115);  view_210 = permute_115 = None
        view_211: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_72, [4, 512, 512]);  mm_72 = None
        view_212: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_211, [4, 512, -1, 64]);  view_211 = None
        permute_116: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_57: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_116, 2);  permute_116 = None
        expand_27: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_57, [4, 8, 4, 512, 64]);  unsqueeze_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_23: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_214: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_23, [4, 32, 512, 64]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_22: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_74, view_213, view_214, where_10, False, scale = 0.125);  add_74 = view_213 = view_214 = where_10 = None
        getitem_90: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_215: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_117, [4, 512, -1]);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_216: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_215, [2048, 2048]);  view_215 = None
        permute_118: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        mm_73: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_216, permute_118);  view_216 = permute_118 = None
        view_217: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_73, [4, 512, 2048]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_76: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_72, view_217);  add_72 = view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_213: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_76, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_22: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_213, 2)
        mean_21: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_77: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-05);  mean_21 = None
        rsqrt_21: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_99: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_213, rsqrt_21);  convert_element_type_213 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_214: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None
        mul_100: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg98_1, convert_element_type_214);  arg98_1 = convert_element_type_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_218: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_100, [2048, 2048])
        permute_119: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_74: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_218, permute_119);  view_218 = permute_119 = None
        view_219: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_74, [4, 512, 8192]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_217: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_219, torch.float32);  view_219 = None
        neg_32: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_217)
        exp_10: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_78: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_217, add_78);  convert_element_type_217 = add_78 = None
        convert_element_type_218: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_220: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_100, [2048, 2048]);  mul_100 = None
        permute_120: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_75: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_220, permute_120);  view_220 = permute_120 = None
        view_221: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_75, [4, 512, 8192]);  mm_75 = None
        mul_101: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_218, view_221);  convert_element_type_218 = view_221 = None
        view_222: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_101, [2048, 8192]);  mul_101 = None
        permute_121: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_76: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_222, permute_121);  view_222 = permute_121 = None
        view_223: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_76, [4, 512, 2048]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_79: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_76, view_223);  add_76 = view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_223: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_23: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_223, 2)
        mean_22: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_80: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-05);  mean_22 = None
        rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_102: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_223, rsqrt_22);  convert_element_type_223 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_224: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        mul_103: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg102_1, convert_element_type_224);  arg102_1 = convert_element_type_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_224: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_103, [2048, 2048])
        permute_122: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_77: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_224, permute_122);  view_224 = permute_122 = None
        view_225: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_77, [4, 512, 2048]);  mm_77 = None
        view_226: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_225, [4, 512, -1, 64]);  view_225 = None
        permute_123: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_58: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_104: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_123, unsqueeze_58)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_49: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_123, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_33: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_49);  slice_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_48: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_123, 3, 0, 32);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_23: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_33, slice_48], -1);  neg_33 = slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_59: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_105: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_23, unsqueeze_59);  cat_23 = None
        add_81: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_227: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_103, [2048, 2048])
        permute_124: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_78: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_227, permute_124);  view_227 = permute_124 = None
        view_228: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_78, [4, 512, 512]);  mm_78 = None
        view_229: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_228, [4, 512, -1, 64]);  view_228 = None
        permute_125: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_106: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_125, unsqueeze_58);  unsqueeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_51: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_125, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_34: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_51);  slice_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_50: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_125, 3, 0, 32);  permute_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_24: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_34, slice_50], -1);  neg_34 = slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_107: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_24, unsqueeze_59);  cat_24 = unsqueeze_59 = None
        add_82: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_60: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_82, 2);  add_82 = None
        expand_28: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_60, [4, 8, 4, 512, 64]);  unsqueeze_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_24: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_233: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_24, [4, 32, 512, 64]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_230: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_103, [2048, 2048]);  mul_103 = None
        permute_126: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_79: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_230, permute_126);  view_230 = permute_126 = None
        view_231: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_79, [4, 512, 512]);  mm_79 = None
        view_232: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_231, [4, 512, -1, 64]);  view_231 = None
        permute_127: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_61: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_127, 2);  permute_127 = None
        expand_29: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_61, [4, 8, 4, 512, 64]);  unsqueeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_25: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_234: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_25, [4, 32, 512, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_24: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_24, full_default_23);  full_default_24 = full_default_23 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_81, view_233, view_234, where_11, False, scale = 0.125);  add_81 = view_233 = view_234 = where_11 = None
        getitem_99: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_128, [4, 512, -1]);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_236: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_235, [2048, 2048]);  view_235 = None
        permute_129: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        mm_80: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_236, permute_129);  view_236 = permute_129 = None
        view_237: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_80, [4, 512, 2048]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_83: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_79, view_237);  add_79 = view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_233: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_24: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 2)
        mean_23: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_84: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-05);  mean_23 = None
        rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_108: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_233, rsqrt_23);  convert_element_type_233 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_234: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_108, torch.bfloat16);  mul_108 = None
        mul_109: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg107_1, convert_element_type_234);  arg107_1 = convert_element_type_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_238: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_109, [2048, 2048])
        permute_130: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_81: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_238, permute_130);  view_238 = permute_130 = None
        view_239: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_81, [4, 512, 8192]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_237: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        neg_35: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_237)
        exp_11: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_35);  neg_35 = None
        add_85: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_237, add_85);  convert_element_type_237 = add_85 = None
        convert_element_type_238: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_240: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_109, [2048, 2048]);  mul_109 = None
        permute_131: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_82: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_240, permute_131);  view_240 = permute_131 = None
        view_241: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_82, [4, 512, 8192]);  mm_82 = None
        mul_110: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_238, view_241);  convert_element_type_238 = view_241 = None
        view_242: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_110, [2048, 8192]);  mul_110 = None
        permute_132: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        mm_83: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_242, permute_132);  view_242 = permute_132 = None
        view_243: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_83, [4, 512, 2048]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_86: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_83, view_243);  add_83 = view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_243: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_86, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_25: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_243, 2)
        mean_24: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_87: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-05);  mean_24 = None
        rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_111: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_243, rsqrt_24);  convert_element_type_243 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_244: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_111, torch.bfloat16);  mul_111 = None
        mul_112: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg111_1, convert_element_type_244);  arg111_1 = convert_element_type_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_244: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_112, [2048, 2048])
        permute_133: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_84: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_244, permute_133);  view_244 = permute_133 = None
        view_245: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_84, [4, 512, 2048]);  mm_84 = None
        view_246: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_245, [4, 512, -1, 64]);  view_245 = None
        permute_134: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_62: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_113: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_134, unsqueeze_62)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_53: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_134, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_36: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_53);  slice_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_52: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_134, 3, 0, 32);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_25: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_36, slice_52], -1);  neg_36 = slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_63: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_114: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_25, unsqueeze_63);  cat_25 = None
        add_88: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_247: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_112, [2048, 2048])
        permute_135: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_85: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_247, permute_135);  view_247 = permute_135 = None
        view_248: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_85, [4, 512, 512]);  mm_85 = None
        view_249: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_248, [4, 512, -1, 64]);  view_248 = None
        permute_136: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_115: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_136, unsqueeze_62);  unsqueeze_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_55: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_136, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_37: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_55);  slice_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_54: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_136, 3, 0, 32);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_26: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_37, slice_54], -1);  neg_37 = slice_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_116: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_26, unsqueeze_63);  cat_26 = unsqueeze_63 = None
        add_89: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_64: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_89, 2);  add_89 = None
        expand_30: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_64, [4, 8, 4, 512, 64]);  unsqueeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_26: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_253: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_26, [4, 32, 512, 64]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_250: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_112, [2048, 2048]);  mul_112 = None
        permute_137: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_86: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_250, permute_137);  view_250 = permute_137 = None
        view_251: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_86, [4, 512, 512]);  mm_86 = None
        view_252: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_251, [4, 512, -1, 64]);  view_251 = None
        permute_138: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_65: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_138, 2);  permute_138 = None
        expand_31: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_65, [4, 8, 4, 512, 64]);  unsqueeze_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_27: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_254: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_27, [4, 32, 512, 64]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_26: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_26, full_default_25);  full_default_26 = full_default_25 = None
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_88, view_253, view_254, where_12, False, scale = 0.125);  add_88 = view_253 = view_254 = where_12 = None
        getitem_108: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_12[0];  _scaled_dot_product_cudnn_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_139: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_255: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_139, [4, 512, -1]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_256: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_255, [2048, 2048]);  view_255 = None
        permute_140: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_87: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_256, permute_140);  view_256 = permute_140 = None
        view_257: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_87, [4, 512, 2048]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_90: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_86, view_257);  add_86 = view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_253: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_90, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_26: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_253, 2)
        mean_25: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_91: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-05);  mean_25 = None
        rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        mul_117: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_253, rsqrt_25);  convert_element_type_253 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_254: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        mul_118: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg116_1, convert_element_type_254);  arg116_1 = convert_element_type_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_258: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_118, [2048, 2048])
        permute_141: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_88: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_258, permute_141);  view_258 = permute_141 = None
        view_259: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_88, [4, 512, 8192]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_257: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_259, torch.float32);  view_259 = None
        neg_38: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_257)
        exp_12: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_38);  neg_38 = None
        add_92: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_257, add_92);  convert_element_type_257 = add_92 = None
        convert_element_type_258: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_260: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_118, [2048, 2048]);  mul_118 = None
        permute_142: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_89: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_260, permute_142);  view_260 = permute_142 = None
        view_261: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_89, [4, 512, 8192]);  mm_89 = None
        mul_119: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_258, view_261);  convert_element_type_258 = view_261 = None
        view_262: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_119, [2048, 8192]);  mul_119 = None
        permute_143: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_90: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_262, permute_143);  view_262 = permute_143 = None
        view_263: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_90, [4, 512, 2048]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_93: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_90, view_263);  add_90 = view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_263: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_93, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_27: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_263, 2)
        mean_26: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_94: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-05);  mean_26 = None
        rsqrt_26: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_120: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_263, rsqrt_26);  convert_element_type_263 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_264: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        mul_121: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg120_1, convert_element_type_264);  arg120_1 = convert_element_type_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_264: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_121, [2048, 2048])
        permute_144: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_91: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_264, permute_144);  view_264 = permute_144 = None
        view_265: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_91, [4, 512, 2048]);  mm_91 = None
        view_266: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_265, [4, 512, -1, 64]);  view_265 = None
        permute_145: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_66: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_122: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_145, unsqueeze_66)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_57: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_145, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_39: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_57);  slice_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_56: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_145, 3, 0, 32);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_27: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_39, slice_56], -1);  neg_39 = slice_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_67: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_123: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_27, unsqueeze_67);  cat_27 = None
        add_95: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_267: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_121, [2048, 2048])
        permute_146: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_92: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_267, permute_146);  view_267 = permute_146 = None
        view_268: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_92, [4, 512, 512]);  mm_92 = None
        view_269: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_268, [4, 512, -1, 64]);  view_268 = None
        permute_147: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_124: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_147, unsqueeze_66);  unsqueeze_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_59: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_147, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_40: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_59);  slice_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_58: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_147, 3, 0, 32);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_28: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_40, slice_58], -1);  neg_40 = slice_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_125: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_28, unsqueeze_67);  cat_28 = unsqueeze_67 = None
        add_96: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_68: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_96, 2);  add_96 = None
        expand_32: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_68, [4, 8, 4, 512, 64]);  unsqueeze_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_28: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_273: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_28, [4, 32, 512, 64]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_270: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_121, [2048, 2048]);  mul_121 = None
        permute_148: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_93: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_270, permute_148);  view_270 = permute_148 = None
        view_271: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_93, [4, 512, 512]);  mm_93 = None
        view_272: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_271, [4, 512, -1, 64]);  view_271 = None
        permute_149: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_69: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_149, 2);  permute_149 = None
        expand_33: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_69, [4, 8, 4, 512, 64]);  unsqueeze_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_29: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_274: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_29, [4, 32, 512, 64]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_28: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_28, full_default_27);  full_default_28 = full_default_27 = None
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_95, view_273, view_274, where_13, False, scale = 0.125);  add_95 = view_273 = view_274 = where_13 = None
        getitem_117: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_13[0];  _scaled_dot_product_cudnn_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_150: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_275: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_150, [4, 512, -1]);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_276: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_275, [2048, 2048]);  view_275 = None
        permute_151: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        mm_94: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_276, permute_151);  view_276 = permute_151 = None
        view_277: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_94, [4, 512, 2048]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_97: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_93, view_277);  add_93 = view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_273: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_97, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_28: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_273, 2)
        mean_27: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_98: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-05);  mean_27 = None
        rsqrt_27: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_126: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_273, rsqrt_27);  convert_element_type_273 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_274: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        mul_127: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg125_1, convert_element_type_274);  arg125_1 = convert_element_type_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_278: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_127, [2048, 2048])
        permute_152: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_95: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_278, permute_152);  view_278 = permute_152 = None
        view_279: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_95, [4, 512, 8192]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_277: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        neg_41: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_277)
        exp_13: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_41);  neg_41 = None
        add_99: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_277, add_99);  convert_element_type_277 = add_99 = None
        convert_element_type_278: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_280: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_127, [2048, 2048]);  mul_127 = None
        permute_153: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_96: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_280, permute_153);  view_280 = permute_153 = None
        view_281: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_96, [4, 512, 8192]);  mm_96 = None
        mul_128: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_278, view_281);  convert_element_type_278 = view_281 = None
        view_282: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_128, [2048, 8192]);  mul_128 = None
        permute_154: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        mm_97: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_282, permute_154);  view_282 = permute_154 = None
        view_283: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_97, [4, 512, 2048]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_100: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_97, view_283);  add_97 = view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_283: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_100, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_29: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_283, 2)
        mean_28: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_101: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-05);  mean_28 = None
        rsqrt_28: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        mul_129: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_283, rsqrt_28);  convert_element_type_283 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_284: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        mul_130: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg129_1, convert_element_type_284);  arg129_1 = convert_element_type_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_284: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_130, [2048, 2048])
        permute_155: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        mm_98: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_284, permute_155);  view_284 = permute_155 = None
        view_285: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_98, [4, 512, 2048]);  mm_98 = None
        view_286: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_285, [4, 512, -1, 64]);  view_285 = None
        permute_156: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_70: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_131: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_156, unsqueeze_70)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_61: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_156, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_42: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_61);  slice_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_60: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_156, 3, 0, 32);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_29: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_42, slice_60], -1);  neg_42 = slice_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_71: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_132: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_29, unsqueeze_71);  cat_29 = None
        add_102: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_287: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_130, [2048, 2048])
        permute_157: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_99: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_287, permute_157);  view_287 = permute_157 = None
        view_288: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_99, [4, 512, 512]);  mm_99 = None
        view_289: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_288, [4, 512, -1, 64]);  view_288 = None
        permute_158: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_133: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_158, unsqueeze_70);  unsqueeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_63: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_158, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_43: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_63);  slice_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_62: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_158, 3, 0, 32);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_30: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_43, slice_62], -1);  neg_43 = slice_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_134: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_30, unsqueeze_71);  cat_30 = unsqueeze_71 = None
        add_103: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_72: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_103, 2);  add_103 = None
        expand_34: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_72, [4, 8, 4, 512, 64]);  unsqueeze_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_30: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_293: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_30, [4, 32, 512, 64]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_290: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_130, [2048, 2048]);  mul_130 = None
        permute_159: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        mm_100: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_290, permute_159);  view_290 = permute_159 = None
        view_291: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_100, [4, 512, 512]);  mm_100 = None
        view_292: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_291, [4, 512, -1, 64]);  view_291 = None
        permute_160: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_73: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_160, 2);  permute_160 = None
        expand_35: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_73, [4, 8, 4, 512, 64]);  unsqueeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_31: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_294: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_31, [4, 32, 512, 64]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_30: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_30, full_default_29);  full_default_30 = full_default_29 = None
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_102, view_293, view_294, where_14, False, scale = 0.125);  add_102 = view_293 = view_294 = where_14 = None
        getitem_126: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_14[0];  _scaled_dot_product_cudnn_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_161: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_295: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_161, [4, 512, -1]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_296: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_295, [2048, 2048]);  view_295 = None
        permute_162: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_101: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_296, permute_162);  view_296 = permute_162 = None
        view_297: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_101, [4, 512, 2048]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_104: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_100, view_297);  add_100 = view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_293: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_104, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_30: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_293, 2)
        mean_29: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_105: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-05);  mean_29 = None
        rsqrt_29: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        mul_135: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_293, rsqrt_29);  convert_element_type_293 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_294: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_135, torch.bfloat16);  mul_135 = None
        mul_136: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg134_1, convert_element_type_294);  arg134_1 = convert_element_type_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_298: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_136, [2048, 2048])
        permute_163: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_102: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_298, permute_163);  view_298 = permute_163 = None
        view_299: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_102, [4, 512, 8192]);  mm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_297: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        neg_44: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_297)
        exp_14: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_44);  neg_44 = None
        add_106: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_297, add_106);  convert_element_type_297 = add_106 = None
        convert_element_type_298: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_300: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_136, [2048, 2048]);  mul_136 = None
        permute_164: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        mm_103: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_300, permute_164);  view_300 = permute_164 = None
        view_301: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_103, [4, 512, 8192]);  mm_103 = None
        mul_137: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_298, view_301);  convert_element_type_298 = view_301 = None
        view_302: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_137, [2048, 8192]);  mul_137 = None
        permute_165: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_104: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_302, permute_165);  view_302 = permute_165 = None
        view_303: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_104, [4, 512, 2048]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_107: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_104, view_303);  add_104 = view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_303: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_31: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_303, 2)
        mean_30: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_108: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-05);  mean_30 = None
        rsqrt_30: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_138: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_303, rsqrt_30);  convert_element_type_303 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_304: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_138, torch.bfloat16);  mul_138 = None
        mul_139: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg138_1, convert_element_type_304);  arg138_1 = convert_element_type_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_304: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_139, [2048, 2048])
        permute_166: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        mm_105: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_304, permute_166);  view_304 = permute_166 = None
        view_305: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_105, [4, 512, 2048]);  mm_105 = None
        view_306: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(view_305, [4, 512, -1, 64]);  view_305 = None
        permute_167: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_74: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_140: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_167, unsqueeze_74)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_65: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_167, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_45: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_65);  slice_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_64: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_167, 3, 0, 32);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_31: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_45, slice_64], -1);  neg_45 = slice_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_75: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_141: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_31, unsqueeze_75);  cat_31 = None
        add_109: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_307: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_139, [2048, 2048])
        permute_168: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        mm_106: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_307, permute_168);  view_307 = permute_168 = None
        view_308: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_106, [4, 512, 512]);  mm_106 = None
        view_309: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_308, [4, 512, -1, 64]);  view_308 = None
        permute_169: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_142: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_169, unsqueeze_74);  unsqueeze_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_67: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_169, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_46: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_67);  slice_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_66: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_169, 3, 0, 32);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_32: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_46, slice_66], -1);  neg_46 = slice_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_143: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_32, unsqueeze_75);  cat_32 = unsqueeze_75 = None
        add_110: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_76: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_110, 2);  add_110 = None
        expand_36: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_76, [4, 8, 4, 512, 64]);  unsqueeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_32: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_313: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_32, [4, 32, 512, 64]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_310: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_139, [2048, 2048]);  mul_139 = None
        permute_170: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        mm_107: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_310, permute_170);  view_310 = permute_170 = None
        view_311: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_107, [4, 512, 512]);  mm_107 = None
        view_312: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(view_311, [4, 512, -1, 64]);  view_311 = None
        permute_171: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_77: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_171, 2);  permute_171 = None
        expand_37: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_77, [4, 8, 4, 512, 64]);  unsqueeze_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_33: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_314: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_33, [4, 32, 512, 64]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_32: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_32, full_default_31);  expand_1 = full_default_32 = full_default_31 = None
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_109, view_313, view_314, where_15, False, scale = 0.125);  add_109 = view_313 = view_314 = where_15 = None
        getitem_135: "bf16[4, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_15[0];  _scaled_dot_product_cudnn_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[4, 513]" = torch.ops.aten.constant_pad_nd.default(arg148_1, [0, 1], -100.0);  arg148_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_68: "i64[4, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_34: "i64[4, 512]" = torch.ops.aten.clone.default(slice_68, memory_format = torch.contiguous_format);  slice_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_327: "i64[2048]" = torch.ops.aten.reshape.default(clone_34, [-1]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        ne_2: "b8[2048]" = torch.ops.aten.ne.Scalar(view_327, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_172: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_315: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_172, [4, 512, -1]);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_316: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_315, [2048, 2048]);  view_315 = None
        permute_173: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        mm_108: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_316, permute_173);  view_316 = permute_173 = None
        view_317: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_108, [4, 512, 2048]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_111: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_107, view_317);  add_107 = view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_313: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_111, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_32: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_313, 2)
        mean_31: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_112: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-05);  mean_31 = None
        rsqrt_31: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_144: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_313, rsqrt_31);  convert_element_type_313 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_314: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None
        mul_145: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg143_1, convert_element_type_314);  arg143_1 = convert_element_type_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_318: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_145, [2048, 2048])
        permute_174: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        mm_109: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_318, permute_174);  view_318 = permute_174 = None
        view_319: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_109, [4, 512, 8192]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_317: "f32[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(view_319, torch.float32);  view_319 = None
        neg_47: "f32[4, 512, 8192]" = torch.ops.aten.neg.default(convert_element_type_317)
        exp_15: "f32[4, 512, 8192]" = torch.ops.aten.exp.default(neg_47);  neg_47 = None
        add_113: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[4, 512, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_317, add_113);  convert_element_type_317 = add_113 = None
        convert_element_type_318: "bf16[4, 512, 8192]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_320: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_145, [2048, 2048]);  mul_145 = None
        permute_175: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        mm_110: "bf16[2048, 8192]" = torch.ops.aten.mm.default(view_320, permute_175);  view_320 = permute_175 = None
        view_321: "bf16[4, 512, 8192]" = torch.ops.aten.reshape.default(mm_110, [4, 512, 8192]);  mm_110 = None
        mul_146: "bf16[4, 512, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_318, view_321);  convert_element_type_318 = view_321 = None
        view_322: "bf16[2048, 8192]" = torch.ops.aten.reshape.default(mul_146, [2048, 8192]);  mul_146 = None
        permute_176: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        mm_111: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_322, permute_176);  view_322 = permute_176 = None
        view_323: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_111, [4, 512, 2048]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_114: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_111, view_323);  add_111 = view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_323: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_33: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_323, 2)
        mean_32: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_115: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_32, 1e-05);  mean_32 = None
        rsqrt_32: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_147: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_323, rsqrt_32);  convert_element_type_323 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_324: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_147, torch.bfloat16);  mul_147 = None
        mul_148: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg147_1, convert_element_type_324);  arg147_1 = convert_element_type_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:487 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_324: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_148, [2048, 2048]);  mul_148 = None
        permute_177: "bf16[2048, 128256]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_112: "bf16[2048, 128256]" = torch.ops.aten.mm.default(view_324, permute_177);  view_324 = permute_177 = None
        view_325: "bf16[4, 512, 128256]" = torch.ops.aten.reshape.default(mm_112, [4, 512, 128256]);  mm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_327: "f32[4, 512, 128256]" = torch.ops.prims.convert_element_type.default(view_325, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_326: "f32[2048, 128256]" = torch.ops.aten.reshape.default(convert_element_type_327, [-1, 128256]);  convert_element_type_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax: "f32[2048, 1]" = torch.ops.aten.amax.default(view_326, [1], True)
        sub_2: "f32[2048, 128256]" = torch.ops.aten.sub.Tensor(view_326, amax);  view_326 = amax = None
        exp_16: "f32[2048, 128256]" = torch.ops.aten.exp.default(sub_2)
        sum_1: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [1], True);  exp_16 = None
        log: "f32[2048, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_3: "f32[2048, 128256]" = torch.ops.aten.sub.Tensor(sub_2, log);  sub_2 = log = None
        ne_1: "b8[2048]" = torch.ops.aten.ne.Scalar(view_327, -100)
        full_default_33: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "i64[2048]" = torch.ops.aten.where.self(ne_1, view_327, full_default_33);  ne_1 = full_default_33 = None
        unsqueeze_78: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(where_16, 1);  where_16 = None
        gather: "f32[2048, 1]" = torch.ops.aten.gather.default(sub_3, 1, unsqueeze_78);  sub_3 = unsqueeze_78 = None
        squeeze: "f32[2048]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_48: "f32[2048]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_34: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "f32[2048]" = torch.ops.aten.where.self(ne_2, neg_48, full_default_34);  ne_2 = neg_48 = full_default_34 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_17);  where_17 = None
        ne_3: "b8[2048]" = torch.ops.aten.ne.Scalar(view_327, -100);  view_327 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
        convert_element_type_328: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        div_16: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_328);  sum_3 = convert_element_type_328 = None
        return (div_16, view_325)
