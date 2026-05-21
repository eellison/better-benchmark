class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 512]", arg1_1: "bf16[151936, 2048]", arg2_1: "f32[64]", arg3_1: "bf16[2048]", arg4_1: "bf16[4096, 2048]", arg5_1: "bf16[128]", arg6_1: "bf16[512, 2048]", arg7_1: "bf16[128]", arg8_1: "bf16[512, 2048]", arg9_1: "bf16[2048, 4096]", arg10_1: "bf16[2048]", arg11_1: "bf16[128, 2048]", arg12_1: "bf16[128, 1536, 2048]", arg13_1: "bf16[128, 2048, 768]", arg14_1: "bf16[2048]", arg15_1: "bf16[4096, 2048]", arg16_1: "bf16[128]", arg17_1: "bf16[512, 2048]", arg18_1: "bf16[128]", arg19_1: "bf16[512, 2048]", arg20_1: "bf16[2048, 4096]", arg21_1: "bf16[2048]", arg22_1: "bf16[128, 2048]", arg23_1: "bf16[128, 1536, 2048]", arg24_1: "bf16[128, 2048, 768]", arg25_1: "bf16[2048]", arg26_1: "bf16[4096, 2048]", arg27_1: "bf16[128]", arg28_1: "bf16[512, 2048]", arg29_1: "bf16[128]", arg30_1: "bf16[512, 2048]", arg31_1: "bf16[2048, 4096]", arg32_1: "bf16[2048]", arg33_1: "bf16[128, 2048]", arg34_1: "bf16[128, 1536, 2048]", arg35_1: "bf16[128, 2048, 768]", arg36_1: "bf16[2048]", arg37_1: "bf16[4096, 2048]", arg38_1: "bf16[128]", arg39_1: "bf16[512, 2048]", arg40_1: "bf16[128]", arg41_1: "bf16[512, 2048]", arg42_1: "bf16[2048, 4096]", arg43_1: "bf16[2048]", arg44_1: "bf16[128, 2048]", arg45_1: "bf16[128, 1536, 2048]", arg46_1: "bf16[128, 2048, 768]", arg47_1: "bf16[2048]", arg48_1: "bf16[151936, 2048]", arg49_1: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:489 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "bf16[4, 512, 2048]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_3: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 2)
        mean: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_3: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_3: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt);  convert_element_type_3 = rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_4: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_4: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg3_1, convert_element_type_4);  arg3_1 = convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_4: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_4, [2048, 2048])
        permute_1: "bf16[2048, 4096]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
        view_5: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm, [4, 512, 4096]);  mm = None
        view_6: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_5, [4, 512, -1, 128]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_7: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_2: "f32[4, 512, 32, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_7, 2)
        mean_1: "f32[4, 512, 32, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_4: "f32[4, 512, 32, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[4, 512, 32, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_5: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_7, rsqrt_1);  convert_element_type_7 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_8: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        mul_6: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(arg5_1, convert_element_type_8);  arg5_1 = convert_element_type_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_2: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(mul_6, [0, 2, 1, 3]);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:438 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        expand_2: "f32[1, 64, 1]" = torch.ops.aten.expand.default(unsqueeze_11, [1, -1, 1]);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_3: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_2, [1, 64, 1]);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:493 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:494 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:439 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_12: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        convert_element_type: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_12, torch.float32);  unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:443 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_4: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type, [1, 1, 512]);  convert_element_type = None
        mul: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_3, expand_4);  expand_3 = expand_4 = None
        permute: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:444 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_13: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_5: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_13, [1, 512, 2, 64]);  unsqueeze_13 = None
        clone: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_3: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(clone, [1, 512, 128]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:445 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 512, 128]" = torch.ops.aten.cos.default(view_3)
        mul_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:448 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_1: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:82 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_14: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_9: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_2, unsqueeze_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_5: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_2, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_5);  slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_4: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_2, 3, 0, 64);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_1: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg, slice_4], -1);  neg = slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:446 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 512, 128]" = torch.ops.aten.sin.default(view_3);  view_3 = None
        mul_2: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:448 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:83 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_15: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_10: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_15);  cat_1 = None
        add_6: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_7: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_4, [2048, 2048])
        permute_3: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_1: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_7, permute_3);  view_7 = permute_3 = None
        view_8: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_1, [4, 512, 512]);  mm_1 = None
        view_9: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_8, [4, 512, -1, 128]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_11: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_3: "f32[4, 512, 4, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_11, 2)
        mean_2: "f32[4, 512, 4, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_5: "f32[4, 512, 4, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[4, 512, 4, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_7: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_11, rsqrt_2);  convert_element_type_11 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_12: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        mul_8: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(arg7_1, convert_element_type_12);  arg7_1 = convert_element_type_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_4: "bf16[4, 4, 512, 128]" = torch.ops.aten.permute.default(mul_8, [0, 2, 1, 3]);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_11: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_14);  unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_7: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(permute_4, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "bf16[4, 4, 512, 64]" = torch.ops.aten.neg.default(slice_7);  slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_6: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(permute_4, 3, 0, 64);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_2: "bf16[4, 4, 512, 128]" = torch.ops.aten.cat.default([neg_1, slice_6], -1);  neg_1 = slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_12: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_15);  cat_2 = unsqueeze_15 = None
        add_7: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_16: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_7, 2);  add_7 = None
        expand_6: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_16, [4, 4, 8, 512, 128]);  unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_2: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_13: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_2, [4, 32, 512, 128]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_10: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_4, [2048, 2048]);  mul_4 = None
        permute_5: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        mm_2: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
        view_11: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 512]);  mm_2 = None
        view_12: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_11, [4, 512, -1, 128]);  view_11 = None
        permute_6: "bf16[4, 4, 512, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_17: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_6, 2);  permute_6 = None
        expand_7: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_17, [4, 4, 8, 512, 128]);  unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_3: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
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
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_6, view_13, view_14, where, False, scale = 0.08838834764831845);  add_6 = view_13 = view_14 = where = None
        getitem: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_7, [4, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_16: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_15, [2048, 4096]);  view_15 = None
        permute_8: "bf16[4096, 2048]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_3: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
        view_17: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 2048]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:346 in forward, code: hidden_states = residual + hidden_states
        add_8: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(embedding, view_17);  embedding = view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_17: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_4: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_17, 2)
        mean_3: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_9: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_13: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_17, rsqrt_3);  convert_element_type_17 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_18: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        mul_14: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg10_1, convert_element_type_18);  arg10_1 = convert_element_type_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:283 in forward, code: hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        view_18: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_14, [-1, 2048]);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_9: "bf16[2048, 128]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_4: "bf16[2048, 128]" = torch.ops.aten.mm.default(view_18, permute_9);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:266 in forward, code: router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        convert_element_type_21: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        amax: "f32[2048, 1]" = torch.ops.aten.amax.default(convert_element_type_21, [-1], True)
        sub_2: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_21, amax);  convert_element_type_21 = amax = None
        exp: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:267 in forward, code: router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        topk = torch.ops.aten.topk.default(div, 8);  div = None
        getitem_9: "f32[2048, 8]" = topk[0]
        getitem_10: "i64[2048, 8]" = topk[1];  topk = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:386 in grouped_mm_experts_forward, code: expert_ids = top_k_index.reshape(-1)  # (S,)
        view_21: "i64[16384]" = torch.ops.aten.reshape.default(getitem_10, [-1]);  getitem_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:389 in grouped_mm_experts_forward, code: expert_ids_g, perm = torch.sort(expert_ids)
        sort = torch.ops.aten.sort.default(view_21);  view_21 = None
        getitem_11: "i64[16384]" = sort[0]
        getitem_12: "i64[16384]" = sort[1];  sort = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:416 in grouped_mm_experts_forward, code: sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
        ge: "b8[16384]" = torch.ops.aten.ge.Scalar(getitem_11, 128)
        unsqueeze_18: "b8[16384, 1]" = torch.ops.aten.unsqueeze.default(ge, -1);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        full_default_3: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:390 in grouped_mm_experts_forward, code: selected_hidden_states_g = hidden_states[perm // num_top_k]
        div_2: "i64[16384]" = torch.ops.aten.div.Tensor_mode(getitem_12, 8, rounding_mode = 'floor')
        index_2: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(view_18, [div_2]);  view_18 = div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        where_1: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, index_2);  full_default_3 = index_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_10: "bf16[128, 2048, 1536]" = torch.ops.aten.permute.default(arg12_1, [0, 2, 1]);  arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:397 in grouped_mm_experts_forward, code: histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
        convert_element_type_23: "i32[16384]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.int32);  getitem_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:398 in grouped_mm_experts_forward, code: tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
        histc: "i32[128]" = torch.ops.aten.histc.default(convert_element_type_23, 128, 0, 127);  convert_element_type_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:399 in grouped_mm_experts_forward, code: offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
        cumsum_1: "i32[128]" = torch.ops.aten.cumsum.default(histc, 0, dtype = torch.int32);  histc = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        _grouped_mm: "bf16[16384, 1536]" = torch.ops.aten._grouped_mm.default(where_1, permute_10, cumsum_1);  where_1 = permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        split = torch.ops.aten.split.Tensor(_grouped_mm, 768, -1);  _grouped_mm = None
        getitem_13: "bf16[16384, 768]" = split[0]
        getitem_14: "bf16[16384, 768]" = split[1];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        full_default_4: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_24: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
        neg_2: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_24)
        exp_1: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_10: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_3: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_24, add_10);  convert_element_type_24 = add_10 = None
        convert_element_type_25: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:515 in _default_apply_gate, code: return self.act_fn(gate) * up  # (S, intermediate_dim)
        mul_15: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, getitem_14);  convert_element_type_25 = getitem_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_11: "bf16[128, 768, 2048]" = torch.ops.aten.permute.default(arg13_1, [0, 2, 1]);  arg13_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        _grouped_mm_1: "bf16[16384, 2048]" = torch.ops.aten._grouped_mm.default(mul_15, permute_11, cumsum_1);  mul_15 = permute_11 = cumsum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:269 in forward, code: router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        sum_2: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(getitem_9, [-1], True)
        div_1: "f32[2048, 8]" = torch.ops.aten.div.Tensor(getitem_9, sum_2);  getitem_9 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_22: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_20: "bf16[16384]" = torch.ops.aten.reshape.default(convert_element_type_22, [-1]);  convert_element_type_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_3: "bf16[16384]" = torch.ops.aten.index.Tensor(view_20, [getitem_12]);  view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        unsqueeze_19: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_3, -1);  index_3 = None
        mul_16: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(_grouped_mm_1, unsqueeze_19);  _grouped_mm_1 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        where_2: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_18, full_default_4, mul_16);  unsqueeze_18 = full_default_4 = mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:464 in grouped_mm_experts_forward, code: inv_perm = torch.empty_like(perm)
        empty: "i64[16384]" = torch.ops.aten.empty.memory_format([16384], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:465 in grouped_mm_experts_forward, code: inv_perm[perm] = torch.arange(perm.size(0), device=device)
        iota_5: "i64[16384]" = torch.ops.prims.iota.default(16384, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put: "i64[16384]" = torch.ops.aten.index_put.default(empty, [getitem_12], iota_5);  empty = getitem_12 = iota_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:466 in grouped_mm_experts_forward, code: weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)
        index_4: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(where_2, [index_put]);  where_2 = index_put = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:472 in grouped_mm_experts_forward, code: final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        view_22: "bf16[2048, 8, 2048]" = torch.ops.aten.reshape.default(index_4, [2048, 8, 2048]);  index_4 = None
        sum_3: "bf16[2048, 2048]" = torch.ops.aten.sum.dim_IntList(view_22, [1]);  view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:286 in forward, code: return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        view_23: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(sum_3, [4, 512, 2048]);  sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 in forward, code: hidden_states = residual + hidden_states
        add_11: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_8, view_23);  add_8 = view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_26: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_5: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_26, 2)
        mean_4: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_12: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_17: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_26, rsqrt_4);  convert_element_type_26 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_27: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        mul_18: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg14_1, convert_element_type_27);  arg14_1 = convert_element_type_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_24: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_18, [2048, 2048])
        permute_12: "bf16[2048, 4096]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_5: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_24, permute_12);  view_24 = permute_12 = None
        view_25: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_5, [4, 512, 4096]);  mm_5 = None
        view_26: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_25, [4, 512, -1, 128]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_30: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(view_26, torch.float32);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_6: "f32[4, 512, 32, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_30, 2)
        mean_5: "f32[4, 512, 32, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_13: "f32[4, 512, 32, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[4, 512, 32, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_19: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_30, rsqrt_5);  convert_element_type_30 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_31: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        mul_20: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(arg16_1, convert_element_type_31);  arg16_1 = convert_element_type_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_13: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(mul_20, [0, 2, 1, 3]);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:82 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_20: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_23: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_13, unsqueeze_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_9: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_13, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_3: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_9);  slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_8: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_13, 3, 0, 64);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_3: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg_3, slice_8], -1);  neg_3 = slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:83 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_21: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_24: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_21);  cat_3 = None
        add_15: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_27: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_18, [2048, 2048])
        permute_14: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        mm_6: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_27, permute_14);  view_27 = permute_14 = None
        view_28: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_6, [4, 512, 512]);  mm_6 = None
        view_29: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_28, [4, 512, -1, 128]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_34: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_7: "f32[4, 512, 4, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_34, 2)
        mean_6: "f32[4, 512, 4, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_14: "f32[4, 512, 4, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[4, 512, 4, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_21: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_34, rsqrt_6);  convert_element_type_34 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_35: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        mul_22: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(arg18_1, convert_element_type_35);  arg18_1 = convert_element_type_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_15: "bf16[4, 4, 512, 128]" = torch.ops.aten.permute.default(mul_22, [0, 2, 1, 3]);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_25: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(permute_15, unsqueeze_20);  unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_11: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(permute_15, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_4: "bf16[4, 4, 512, 64]" = torch.ops.aten.neg.default(slice_11);  slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_10: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(permute_15, 3, 0, 64);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_4: "bf16[4, 4, 512, 128]" = torch.ops.aten.cat.default([neg_4, slice_10], -1);  neg_4 = slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_26: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_21);  cat_4 = unsqueeze_21 = None
        add_16: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_22: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_16, 2);  add_16 = None
        expand_8: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_22, [4, 4, 8, 512, 128]);  unsqueeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_4: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_33: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_4, [4, 32, 512, 128]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_30: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_18, [2048, 2048]);  mul_18 = None
        permute_16: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_7: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_30, permute_16);  view_30 = permute_16 = None
        view_31: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_7, [4, 512, 512]);  mm_7 = None
        view_32: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_31, [4, 512, -1, 128]);  view_31 = None
        permute_17: "bf16[4, 4, 512, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_23: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_17, 2);  permute_17 = None
        expand_9: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_23, [4, 4, 8, 512, 128]);  unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_5: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_34: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_5, [4, 32, 512, 128]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_6: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_15, view_33, view_34, where_3, False, scale = 0.08838834764831845);  add_15 = view_33 = view_34 = where_3 = None
        getitem_15: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_35: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_18, [4, 512, -1]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_36: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_35, [2048, 4096]);  view_35 = None
        permute_19: "bf16[4096, 2048]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_8: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_36, permute_19);  view_36 = permute_19 = None
        view_37: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_8, [4, 512, 2048]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:346 in forward, code: hidden_states = residual + hidden_states
        add_17: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_11, view_37);  add_11 = view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_40: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_8: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_40, 2)
        mean_7: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_18: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_27: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_40, rsqrt_7);  convert_element_type_40 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_41: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        mul_28: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg21_1, convert_element_type_41);  arg21_1 = convert_element_type_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:283 in forward, code: hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        view_38: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_28, [-1, 2048]);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_20: "bf16[2048, 128]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_9: "bf16[2048, 128]" = torch.ops.aten.mm.default(view_38, permute_20);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:266 in forward, code: router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        convert_element_type_44: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        amax_1: "f32[2048, 1]" = torch.ops.aten.amax.default(convert_element_type_44, [-1], True)
        sub_3: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_44, amax_1);  convert_element_type_44 = amax_1 = None
        exp_2: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_4: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_4);  exp_2 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:267 in forward, code: router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        topk_1 = torch.ops.aten.topk.default(div_4, 8);  div_4 = None
        getitem_24: "f32[2048, 8]" = topk_1[0]
        getitem_25: "i64[2048, 8]" = topk_1[1];  topk_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:386 in grouped_mm_experts_forward, code: expert_ids = top_k_index.reshape(-1)  # (S,)
        view_41: "i64[16384]" = torch.ops.aten.reshape.default(getitem_25, [-1]);  getitem_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:389 in grouped_mm_experts_forward, code: expert_ids_g, perm = torch.sort(expert_ids)
        sort_1 = torch.ops.aten.sort.default(view_41);  view_41 = None
        getitem_26: "i64[16384]" = sort_1[0]
        getitem_27: "i64[16384]" = sort_1[1];  sort_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:416 in grouped_mm_experts_forward, code: sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
        ge_1: "b8[16384]" = torch.ops.aten.ge.Scalar(getitem_26, 128)
        unsqueeze_24: "b8[16384, 1]" = torch.ops.aten.unsqueeze.default(ge_1, -1);  ge_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        full_default_7: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:390 in grouped_mm_experts_forward, code: selected_hidden_states_g = hidden_states[perm // num_top_k]
        div_6: "i64[16384]" = torch.ops.aten.div.Tensor_mode(getitem_27, 8, rounding_mode = 'floor')
        index_5: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(view_38, [div_6]);  view_38 = div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        where_4: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_24, full_default_7, index_5);  full_default_7 = index_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_21: "bf16[128, 2048, 1536]" = torch.ops.aten.permute.default(arg23_1, [0, 2, 1]);  arg23_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:397 in grouped_mm_experts_forward, code: histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
        convert_element_type_46: "i32[16384]" = torch.ops.prims.convert_element_type.default(getitem_26, torch.int32);  getitem_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:398 in grouped_mm_experts_forward, code: tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
        histc_1: "i32[128]" = torch.ops.aten.histc.default(convert_element_type_46, 128, 0, 127);  convert_element_type_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:399 in grouped_mm_experts_forward, code: offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
        cumsum_2: "i32[128]" = torch.ops.aten.cumsum.default(histc_1, 0, dtype = torch.int32);  histc_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        _grouped_mm_2: "bf16[16384, 1536]" = torch.ops.aten._grouped_mm.default(where_4, permute_21, cumsum_2);  where_4 = permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        split_1 = torch.ops.aten.split.Tensor(_grouped_mm_2, 768, -1);  _grouped_mm_2 = None
        getitem_28: "bf16[16384, 768]" = split_1[0]
        getitem_29: "bf16[16384, 768]" = split_1[1];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        full_default_8: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_47: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem_28, torch.float32);  getitem_28 = None
        neg_5: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_47)
        exp_3: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_19: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_7: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_47, add_19);  convert_element_type_47 = add_19 = None
        convert_element_type_48: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:515 in _default_apply_gate, code: return self.act_fn(gate) * up  # (S, intermediate_dim)
        mul_29: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_48, getitem_29);  convert_element_type_48 = getitem_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_22: "bf16[128, 768, 2048]" = torch.ops.aten.permute.default(arg24_1, [0, 2, 1]);  arg24_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        _grouped_mm_3: "bf16[16384, 2048]" = torch.ops.aten._grouped_mm.default(mul_29, permute_22, cumsum_2);  mul_29 = permute_22 = cumsum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:269 in forward, code: router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        sum_5: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(getitem_24, [-1], True)
        div_5: "f32[2048, 8]" = torch.ops.aten.div.Tensor(getitem_24, sum_5);  getitem_24 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_45: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_40: "bf16[16384]" = torch.ops.aten.reshape.default(convert_element_type_45, [-1]);  convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_6: "bf16[16384]" = torch.ops.aten.index.Tensor(view_40, [getitem_27]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        unsqueeze_25: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_6, -1);  index_6 = None
        mul_30: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(_grouped_mm_3, unsqueeze_25);  _grouped_mm_3 = unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        where_5: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_24, full_default_8, mul_30);  unsqueeze_24 = full_default_8 = mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:464 in grouped_mm_experts_forward, code: inv_perm = torch.empty_like(perm)
        empty_1: "i64[16384]" = torch.ops.aten.empty.memory_format([16384], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:465 in grouped_mm_experts_forward, code: inv_perm[perm] = torch.arange(perm.size(0), device=device)
        iota_6: "i64[16384]" = torch.ops.prims.iota.default(16384, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put_1: "i64[16384]" = torch.ops.aten.index_put.default(empty_1, [getitem_27], iota_6);  empty_1 = getitem_27 = iota_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:466 in grouped_mm_experts_forward, code: weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)
        index_7: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(where_5, [index_put_1]);  where_5 = index_put_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:472 in grouped_mm_experts_forward, code: final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        view_42: "bf16[2048, 8, 2048]" = torch.ops.aten.reshape.default(index_7, [2048, 8, 2048]);  index_7 = None
        sum_6: "bf16[2048, 2048]" = torch.ops.aten.sum.dim_IntList(view_42, [1]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:286 in forward, code: return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        view_43: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(sum_6, [4, 512, 2048]);  sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 in forward, code: hidden_states = residual + hidden_states
        add_20: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_17, view_43);  add_17 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_49: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_9: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_49, 2)
        mean_8: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_21: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_31: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_49, rsqrt_8);  convert_element_type_49 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_50: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        mul_32: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg25_1, convert_element_type_50);  arg25_1 = convert_element_type_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_44: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_32, [2048, 2048])
        permute_23: "bf16[2048, 4096]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        mm_10: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_44, permute_23);  view_44 = permute_23 = None
        view_45: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_10, [4, 512, 4096]);  mm_10 = None
        view_46: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_45, [4, 512, -1, 128]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_53: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_10: "f32[4, 512, 32, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_53, 2)
        mean_9: "f32[4, 512, 32, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_22: "f32[4, 512, 32, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[4, 512, 32, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_33: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_53, rsqrt_9);  convert_element_type_53 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_54: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None
        mul_34: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(arg27_1, convert_element_type_54);  arg27_1 = convert_element_type_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_24: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(mul_34, [0, 2, 1, 3]);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:82 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_26: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_37: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_24, unsqueeze_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_13: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_24, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_6: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_13);  slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_12: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_24, 3, 0, 64);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_5: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg_6, slice_12], -1);  neg_6 = slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:83 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_27: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_38: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_27);  cat_5 = None
        add_24: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_47: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_32, [2048, 2048])
        permute_25: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_11: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_47, permute_25);  view_47 = permute_25 = None
        view_48: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_11, [4, 512, 512]);  mm_11 = None
        view_49: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_48, [4, 512, -1, 128]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_57: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_11: "f32[4, 512, 4, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_57, 2)
        mean_10: "f32[4, 512, 4, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_23: "f32[4, 512, 4, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[4, 512, 4, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_35: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_57, rsqrt_10);  convert_element_type_57 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_58: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        mul_36: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(arg29_1, convert_element_type_58);  arg29_1 = convert_element_type_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_26: "bf16[4, 4, 512, 128]" = torch.ops.aten.permute.default(mul_36, [0, 2, 1, 3]);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_39: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(permute_26, unsqueeze_26);  unsqueeze_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_15: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(permute_26, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_7: "bf16[4, 4, 512, 64]" = torch.ops.aten.neg.default(slice_15);  slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_14: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(permute_26, 3, 0, 64);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_6: "bf16[4, 4, 512, 128]" = torch.ops.aten.cat.default([neg_7, slice_14], -1);  neg_7 = slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_40: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_27);  cat_6 = unsqueeze_27 = None
        add_25: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_28: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_25, 2);  add_25 = None
        expand_10: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_28, [4, 4, 8, 512, 128]);  unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_6: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_53: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_6, [4, 32, 512, 128]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_50: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_32, [2048, 2048]);  mul_32 = None
        permute_27: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_12: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_50, permute_27);  view_50 = permute_27 = None
        view_51: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_12, [4, 512, 512]);  mm_12 = None
        view_52: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_51, [4, 512, -1, 128]);  view_51 = None
        permute_28: "bf16[4, 4, 512, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_29: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_28, 2);  permute_28 = None
        expand_11: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_29, [4, 4, 8, 512, 128]);  unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_7: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_54: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_7, [4, 32, 512, 128]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_10: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_24, view_53, view_54, where_6, False, scale = 0.08838834764831845);  add_24 = view_53 = view_54 = where_6 = None
        getitem_30: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_30, [0, 2, 1, 3]);  getitem_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_55: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_29, [4, 512, -1]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_56: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_55, [2048, 4096]);  view_55 = None
        permute_30: "bf16[4096, 2048]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_13: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_56, permute_30);  view_56 = permute_30 = None
        view_57: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_13, [4, 512, 2048]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:346 in forward, code: hidden_states = residual + hidden_states
        add_26: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_20, view_57);  add_20 = view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_63: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_12: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_63, 2)
        mean_11: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_27: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_41: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_63, rsqrt_11);  convert_element_type_63 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_64: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        mul_42: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg32_1, convert_element_type_64);  arg32_1 = convert_element_type_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:283 in forward, code: hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        view_58: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_42, [-1, 2048]);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_31: "bf16[2048, 128]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_14: "bf16[2048, 128]" = torch.ops.aten.mm.default(view_58, permute_31);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:266 in forward, code: router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        convert_element_type_67: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        amax_2: "f32[2048, 1]" = torch.ops.aten.amax.default(convert_element_type_67, [-1], True)
        sub_4: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_67, amax_2);  convert_element_type_67 = amax_2 = None
        exp_4: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_7: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_8: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_7);  exp_4 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:267 in forward, code: router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        topk_2 = torch.ops.aten.topk.default(div_8, 8);  div_8 = None
        getitem_39: "f32[2048, 8]" = topk_2[0]
        getitem_40: "i64[2048, 8]" = topk_2[1];  topk_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:386 in grouped_mm_experts_forward, code: expert_ids = top_k_index.reshape(-1)  # (S,)
        view_61: "i64[16384]" = torch.ops.aten.reshape.default(getitem_40, [-1]);  getitem_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:389 in grouped_mm_experts_forward, code: expert_ids_g, perm = torch.sort(expert_ids)
        sort_2 = torch.ops.aten.sort.default(view_61);  view_61 = None
        getitem_41: "i64[16384]" = sort_2[0]
        getitem_42: "i64[16384]" = sort_2[1];  sort_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:416 in grouped_mm_experts_forward, code: sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
        ge_2: "b8[16384]" = torch.ops.aten.ge.Scalar(getitem_41, 128)
        unsqueeze_30: "b8[16384, 1]" = torch.ops.aten.unsqueeze.default(ge_2, -1);  ge_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        full_default_11: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:390 in grouped_mm_experts_forward, code: selected_hidden_states_g = hidden_states[perm // num_top_k]
        div_10: "i64[16384]" = torch.ops.aten.div.Tensor_mode(getitem_42, 8, rounding_mode = 'floor')
        index_8: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(view_58, [div_10]);  view_58 = div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        where_7: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_30, full_default_11, index_8);  full_default_11 = index_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_32: "bf16[128, 2048, 1536]" = torch.ops.aten.permute.default(arg34_1, [0, 2, 1]);  arg34_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:397 in grouped_mm_experts_forward, code: histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
        convert_element_type_69: "i32[16384]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.int32);  getitem_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:398 in grouped_mm_experts_forward, code: tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
        histc_2: "i32[128]" = torch.ops.aten.histc.default(convert_element_type_69, 128, 0, 127);  convert_element_type_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:399 in grouped_mm_experts_forward, code: offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
        cumsum_3: "i32[128]" = torch.ops.aten.cumsum.default(histc_2, 0, dtype = torch.int32);  histc_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        _grouped_mm_4: "bf16[16384, 1536]" = torch.ops.aten._grouped_mm.default(where_7, permute_32, cumsum_3);  where_7 = permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        split_2 = torch.ops.aten.split.Tensor(_grouped_mm_4, 768, -1);  _grouped_mm_4 = None
        getitem_43: "bf16[16384, 768]" = split_2[0]
        getitem_44: "bf16[16384, 768]" = split_2[1];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        full_default_12: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_70: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
        neg_8: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_70)
        exp_5: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_28: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_11: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_70, add_28);  convert_element_type_70 = add_28 = None
        convert_element_type_71: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:515 in _default_apply_gate, code: return self.act_fn(gate) * up  # (S, intermediate_dim)
        mul_43: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_71, getitem_44);  convert_element_type_71 = getitem_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_33: "bf16[128, 768, 2048]" = torch.ops.aten.permute.default(arg35_1, [0, 2, 1]);  arg35_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        _grouped_mm_5: "bf16[16384, 2048]" = torch.ops.aten._grouped_mm.default(mul_43, permute_33, cumsum_3);  mul_43 = permute_33 = cumsum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:269 in forward, code: router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        sum_8: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(getitem_39, [-1], True)
        div_9: "f32[2048, 8]" = torch.ops.aten.div.Tensor(getitem_39, sum_8);  getitem_39 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_68: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_60: "bf16[16384]" = torch.ops.aten.reshape.default(convert_element_type_68, [-1]);  convert_element_type_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_9: "bf16[16384]" = torch.ops.aten.index.Tensor(view_60, [getitem_42]);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        unsqueeze_31: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_9, -1);  index_9 = None
        mul_44: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(_grouped_mm_5, unsqueeze_31);  _grouped_mm_5 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        where_8: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_30, full_default_12, mul_44);  unsqueeze_30 = full_default_12 = mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:464 in grouped_mm_experts_forward, code: inv_perm = torch.empty_like(perm)
        empty_2: "i64[16384]" = torch.ops.aten.empty.memory_format([16384], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:465 in grouped_mm_experts_forward, code: inv_perm[perm] = torch.arange(perm.size(0), device=device)
        iota_7: "i64[16384]" = torch.ops.prims.iota.default(16384, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put_2: "i64[16384]" = torch.ops.aten.index_put.default(empty_2, [getitem_42], iota_7);  empty_2 = getitem_42 = iota_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:466 in grouped_mm_experts_forward, code: weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)
        index_10: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(where_8, [index_put_2]);  where_8 = index_put_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:472 in grouped_mm_experts_forward, code: final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        view_62: "bf16[2048, 8, 2048]" = torch.ops.aten.reshape.default(index_10, [2048, 8, 2048]);  index_10 = None
        sum_9: "bf16[2048, 2048]" = torch.ops.aten.sum.dim_IntList(view_62, [1]);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:286 in forward, code: return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        view_63: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(sum_9, [4, 512, 2048]);  sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 in forward, code: hidden_states = residual + hidden_states
        add_29: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_26, view_63);  add_26 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_72: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_13: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_72, 2)
        mean_12: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_30: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_45: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_72, rsqrt_12);  convert_element_type_72 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_73: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        mul_46: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg36_1, convert_element_type_73);  arg36_1 = convert_element_type_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_64: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_46, [2048, 2048])
        permute_34: "bf16[2048, 4096]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_15: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_64, permute_34);  view_64 = permute_34 = None
        view_65: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_15, [4, 512, 4096]);  mm_15 = None
        view_66: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_65, [4, 512, -1, 128]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_76: "f32[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(view_66, torch.float32);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_14: "f32[4, 512, 32, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_76, 2)
        mean_13: "f32[4, 512, 32, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_31: "f32[4, 512, 32, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[4, 512, 32, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_47: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_76, rsqrt_13);  convert_element_type_76 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_77: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_47, torch.bfloat16);  mul_47 = None
        mul_48: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(arg38_1, convert_element_type_77);  arg38_1 = convert_element_type_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_35: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(mul_48, [0, 2, 1, 3]);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:82 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_32: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_51: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_35, unsqueeze_32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_17: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_35, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_9: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_17);  slice_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_16: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_35, 3, 0, 64);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_7: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg_9, slice_16], -1);  neg_9 = slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:83 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_33: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:84 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_52: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_33);  cat_7 = None
        add_33: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_67: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_46, [2048, 2048])
        permute_36: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_16: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_67, permute_36);  view_67 = permute_36 = None
        view_68: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_16, [4, 512, 512]);  mm_16 = None
        view_69: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_68, [4, 512, -1, 128]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_80: "f32[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_15: "f32[4, 512, 4, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_80, 2)
        mean_14: "f32[4, 512, 4, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_32: "f32[4, 512, 4, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[4, 512, 4, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_49: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_80, rsqrt_14);  convert_element_type_80 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_81: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        mul_50: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(arg40_1, convert_element_type_81);  arg40_1 = convert_element_type_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_37: "bf16[4, 4, 512, 128]" = torch.ops.aten.permute.default(mul_50, [0, 2, 1, 3]);  mul_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_53: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(permute_37, unsqueeze_32);  unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:59 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_19: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(permute_37, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_10: "bf16[4, 4, 512, 64]" = torch.ops.aten.neg.default(slice_19);  slice_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:58 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_18: "bf16[4, 4, 512, 64]" = torch.ops.aten.slice.Tensor(permute_37, 3, 0, 64);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:60 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_8: "bf16[4, 4, 512, 128]" = torch.ops.aten.cat.default([neg_10, slice_18], -1);  neg_10 = slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:85 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_54: "bf16[4, 4, 512, 128]" = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_33);  cat_8 = unsqueeze_33 = None
        add_34: "bf16[4, 4, 512, 128]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_34: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_34, 2);  add_34 = None
        expand_12: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_34, [4, 4, 8, 512, 128]);  unsqueeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_8: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_73: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_8, [4, 32, 512, 128]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_70: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_46, [2048, 2048]);  mul_46 = None
        permute_38: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_17: "bf16[2048, 512]" = torch.ops.aten.mm.default(view_70, permute_38);  view_70 = permute_38 = None
        view_71: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_17, [4, 512, 512]);  mm_17 = None
        view_72: "bf16[4, 512, 4, 128]" = torch.ops.aten.reshape.default(view_71, [4, 512, -1, 128]);  view_71 = None
        permute_39: "bf16[4, 4, 512, 128]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_35: "bf16[4, 4, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_39, 2);  permute_39 = None
        expand_13: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_35, [4, 4, 8, 512, 128]);  unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_9: "bf16[4, 4, 8, 512, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_74: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_9, [4, 32, 512, 128]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_14: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_14, full_default_13);  expand_1 = full_default_14 = full_default_13 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_33, view_73, view_74, where_9, False, scale = 0.08838834764831845);  add_33 = view_73 = view_74 = where_9 = None
        getitem_45: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_75: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_40, [4, 512, -1]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        view_76: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_75, [2048, 4096]);  view_75 = None
        permute_41: "bf16[4096, 2048]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_18: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_76, permute_41);  view_76 = permute_41 = None
        view_77: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_18, [4, 512, 2048]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:346 in forward, code: hidden_states = residual + hidden_states
        add_35: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_29, view_77);  add_29 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_86: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_16: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_86, 2)
        mean_15: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_36: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_55: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_86, rsqrt_15);  convert_element_type_86 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_87: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        mul_56: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg43_1, convert_element_type_87);  arg43_1 = convert_element_type_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:283 in forward, code: hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        view_78: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_56, [-1, 2048]);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_42: "bf16[2048, 128]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        mm_19: "bf16[2048, 128]" = torch.ops.aten.mm.default(view_78, permute_42);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:266 in forward, code: router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        convert_element_type_90: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        amax_3: "f32[2048, 1]" = torch.ops.aten.amax.default(convert_element_type_90, [-1], True)
        sub_5: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_90, amax_3);  convert_element_type_90 = amax_3 = None
        exp_6: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_10: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_12: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_10);  exp_6 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:267 in forward, code: router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        topk_3 = torch.ops.aten.topk.default(div_12, 8);  div_12 = None
        getitem_54: "f32[2048, 8]" = topk_3[0]
        getitem_55: "i64[2048, 8]" = topk_3[1];  topk_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:386 in grouped_mm_experts_forward, code: expert_ids = top_k_index.reshape(-1)  # (S,)
        view_81: "i64[16384]" = torch.ops.aten.reshape.default(getitem_55, [-1]);  getitem_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:389 in grouped_mm_experts_forward, code: expert_ids_g, perm = torch.sort(expert_ids)
        sort_3 = torch.ops.aten.sort.default(view_81);  view_81 = None
        getitem_56: "i64[16384]" = sort_3[0]
        getitem_57: "i64[16384]" = sort_3[1];  sort_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:416 in grouped_mm_experts_forward, code: sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
        ge_3: "b8[16384]" = torch.ops.aten.ge.Scalar(getitem_56, 128)
        unsqueeze_36: "b8[16384, 1]" = torch.ops.aten.unsqueeze.default(ge_3, -1);  ge_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        full_default_15: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:390 in grouped_mm_experts_forward, code: selected_hidden_states_g = hidden_states[perm // num_top_k]
        div_14: "i64[16384]" = torch.ops.aten.div.Tensor_mode(getitem_57, 8, rounding_mode = 'floor')
        index_11: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(view_78, [div_14]);  view_78 = div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        where_10: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_36, full_default_15, index_11);  full_default_15 = index_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_43: "bf16[128, 2048, 1536]" = torch.ops.aten.permute.default(arg45_1, [0, 2, 1]);  arg45_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:397 in grouped_mm_experts_forward, code: histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
        convert_element_type_92: "i32[16384]" = torch.ops.prims.convert_element_type.default(getitem_56, torch.int32);  getitem_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:398 in grouped_mm_experts_forward, code: tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
        histc_3: "i32[128]" = torch.ops.aten.histc.default(convert_element_type_92, 128, 0, 127);  convert_element_type_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:399 in grouped_mm_experts_forward, code: offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
        cumsum_4: "i32[128]" = torch.ops.aten.cumsum.default(histc_3, 0, dtype = torch.int32);  histc_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        _grouped_mm_6: "bf16[16384, 1536]" = torch.ops.aten._grouped_mm.default(where_10, permute_43, cumsum_4);  where_10 = permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        split_3 = torch.ops.aten.split.Tensor(_grouped_mm_6, 768, -1);  _grouped_mm_6 = None
        getitem_58: "bf16[16384, 768]" = split_3[0]
        getitem_59: "bf16[16384, 768]" = split_3[1];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[4, 513]" = torch.ops.aten.constant_pad_nd.default(arg49_1, [0, 1], -100.0);  arg49_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_20: "i64[4, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_10: "i64[4, 512]" = torch.ops.aten.clone.default(slice_20, memory_format = torch.contiguous_format);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_87: "i64[2048]" = torch.ops.aten.reshape.default(clone_10, [-1]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        ne_2: "b8[2048]" = torch.ops.aten.ne.Scalar(view_87, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        full_default_16: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_93: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem_58, torch.float32);  getitem_58 = None
        neg_11: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_93)
        exp_7: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_37: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_15: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_93, add_37);  convert_element_type_93 = add_37 = None
        convert_element_type_94: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:515 in _default_apply_gate, code: return self.act_fn(gate) * up  # (S, intermediate_dim)
        mul_57: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_94, getitem_59);  convert_element_type_94 = getitem_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_44: "bf16[128, 768, 2048]" = torch.ops.aten.permute.default(arg46_1, [0, 2, 1]);  arg46_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        _grouped_mm_7: "bf16[16384, 2048]" = torch.ops.aten._grouped_mm.default(mul_57, permute_44, cumsum_4);  mul_57 = permute_44 = cumsum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:269 in forward, code: router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        sum_11: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(getitem_54, [-1], True)
        div_13: "f32[2048, 8]" = torch.ops.aten.div.Tensor(getitem_54, sum_11);  getitem_54 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_91: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        view_80: "bf16[16384]" = torch.ops.aten.reshape.default(convert_element_type_91, [-1]);  convert_element_type_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_12: "bf16[16384]" = torch.ops.aten.index.Tensor(view_80, [getitem_57]);  view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        unsqueeze_37: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_12, -1);  index_12 = None
        mul_58: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(_grouped_mm_7, unsqueeze_37);  _grouped_mm_7 = unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        where_11: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_36, full_default_16, mul_58);  unsqueeze_36 = full_default_16 = mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:464 in grouped_mm_experts_forward, code: inv_perm = torch.empty_like(perm)
        empty_3: "i64[16384]" = torch.ops.aten.empty.memory_format([16384], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:465 in grouped_mm_experts_forward, code: inv_perm[perm] = torch.arange(perm.size(0), device=device)
        iota_8: "i64[16384]" = torch.ops.prims.iota.default(16384, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put_3: "i64[16384]" = torch.ops.aten.index_put.default(empty_3, [getitem_57], iota_8);  empty_3 = getitem_57 = iota_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:466 in grouped_mm_experts_forward, code: weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)
        index_13: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(where_11, [index_put_3]);  where_11 = index_put_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:472 in grouped_mm_experts_forward, code: final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        view_82: "bf16[2048, 8, 2048]" = torch.ops.aten.reshape.default(index_13, [2048, 8, 2048]);  index_13 = None
        sum_12: "bf16[2048, 2048]" = torch.ops.aten.sum.dim_IntList(view_82, [1]);  view_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:286 in forward, code: return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        view_83: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(sum_12, [4, 512, 2048]);  sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 in forward, code: hidden_states = residual + hidden_states
        add_38: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_35, view_83);  add_35 = view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_95: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_17: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_95, 2)
        mean_16: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_39: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_59: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_95, rsqrt_16);  convert_element_type_95 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_96: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        mul_60: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg47_1, convert_element_type_96);  arg47_1 = convert_element_type_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:684 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_84: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_60, [2048, 2048]);  mul_60 = None
        permute_45: "bf16[2048, 151936]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        mm_20: "bf16[2048, 151936]" = torch.ops.aten.mm.default(view_84, permute_45);  view_84 = permute_45 = None
        view_85: "bf16[4, 512, 151936]" = torch.ops.aten.reshape.default(mm_20, [4, 512, 151936]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_99: "f32[4, 512, 151936]" = torch.ops.prims.convert_element_type.default(view_85, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_86: "f32[2048, 151936]" = torch.ops.aten.reshape.default(convert_element_type_99, [-1, 151936]);  convert_element_type_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_4: "f32[2048, 1]" = torch.ops.aten.amax.default(view_86, [1], True)
        sub_6: "f32[2048, 151936]" = torch.ops.aten.sub.Tensor(view_86, amax_4);  view_86 = amax_4 = None
        exp_8: "f32[2048, 151936]" = torch.ops.aten.exp.default(sub_6)
        sum_13: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [1], True);  exp_8 = None
        log: "f32[2048, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_7: "f32[2048, 151936]" = torch.ops.aten.sub.Tensor(sub_6, log);  sub_6 = log = None
        ne_1: "b8[2048]" = torch.ops.aten.ne.Scalar(view_87, -100)
        full_default_17: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[2048]" = torch.ops.aten.where.self(ne_1, view_87, full_default_17);  ne_1 = full_default_17 = None
        unsqueeze_38: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather: "f32[2048, 1]" = torch.ops.aten.gather.default(sub_7, 1, unsqueeze_38);  sub_7 = unsqueeze_38 = None
        squeeze: "f32[2048]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_12: "f32[2048]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_18: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f32[2048]" = torch.ops.aten.where.self(ne_2, neg_12, full_default_18);  ne_2 = neg_12 = full_default_18 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        ne_3: "b8[2048]" = torch.ops.aten.ne.Scalar(view_87, -100);  view_87 = None
        sum_14: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
        convert_element_type_100: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        div_16: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_100);  sum_15 = convert_element_type_100 = None
        return (div_16, view_85)
