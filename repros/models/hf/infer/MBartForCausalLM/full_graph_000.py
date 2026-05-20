class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[8, 1024]", arg1_1: "f32[50265, 1024]", arg2_1: "f32[1026, 1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024]", arg6_1: "f32[1024]", arg7_1: "f32[1024, 1024]", arg8_1: "f32[1024]", arg9_1: "f32[1024, 1024]", arg10_1: "f32[1024]", arg11_1: "f32[1024, 1024]", arg12_1: "f32[1024]", arg13_1: "f32[1024, 1024]", arg14_1: "f32[1024]", arg15_1: "f32[1024]", arg16_1: "f32[1024]", arg17_1: "f32[4096, 1024]", arg18_1: "f32[4096]", arg19_1: "f32[1024, 4096]", arg20_1: "f32[1024]", arg21_1: "f32[1024]", arg22_1: "f32[1024]", arg23_1: "f32[1024, 1024]", arg24_1: "f32[1024]", arg25_1: "f32[1024, 1024]", arg26_1: "f32[1024]", arg27_1: "f32[1024, 1024]", arg28_1: "f32[1024]", arg29_1: "f32[1024, 1024]", arg30_1: "f32[1024]", arg31_1: "f32[1024]", arg32_1: "f32[1024]", arg33_1: "f32[4096, 1024]", arg34_1: "f32[4096]", arg35_1: "f32[1024, 4096]", arg36_1: "f32[1024]", arg37_1: "f32[1024]", arg38_1: "f32[1024]", arg39_1: "f32[1024, 1024]", arg40_1: "f32[1024]", arg41_1: "f32[1024, 1024]", arg42_1: "f32[1024]", arg43_1: "f32[1024, 1024]", arg44_1: "f32[1024]", arg45_1: "f32[1024, 1024]", arg46_1: "f32[1024]", arg47_1: "f32[1024]", arg48_1: "f32[1024]", arg49_1: "f32[4096, 1024]", arg50_1: "f32[4096]", arg51_1: "f32[1024, 4096]", arg52_1: "f32[1024]", arg53_1: "f32[1024]", arg54_1: "f32[1024]", arg55_1: "f32[1024, 1024]", arg56_1: "f32[1024]", arg57_1: "f32[1024, 1024]", arg58_1: "f32[1024]", arg59_1: "f32[1024, 1024]", arg60_1: "f32[1024]", arg61_1: "f32[1024, 1024]", arg62_1: "f32[1024]", arg63_1: "f32[1024]", arg64_1: "f32[1024]", arg65_1: "f32[4096, 1024]", arg66_1: "f32[4096]", arg67_1: "f32[1024, 4096]", arg68_1: "f32[1024]", arg69_1: "f32[1024]", arg70_1: "f32[1024]", arg71_1: "f32[1024, 1024]", arg72_1: "f32[1024]", arg73_1: "f32[1024, 1024]", arg74_1: "f32[1024]", arg75_1: "f32[1024, 1024]", arg76_1: "f32[1024]", arg77_1: "f32[1024, 1024]", arg78_1: "f32[1024]", arg79_1: "f32[1024]", arg80_1: "f32[1024]", arg81_1: "f32[4096, 1024]", arg82_1: "f32[4096]", arg83_1: "f32[1024, 4096]", arg84_1: "f32[1024]", arg85_1: "f32[1024]", arg86_1: "f32[1024]", arg87_1: "f32[1024, 1024]", arg88_1: "f32[1024]", arg89_1: "f32[1024, 1024]", arg90_1: "f32[1024]", arg91_1: "f32[1024, 1024]", arg92_1: "f32[1024]", arg93_1: "f32[1024, 1024]", arg94_1: "f32[1024]", arg95_1: "f32[1024]", arg96_1: "f32[1024]", arg97_1: "f32[4096, 1024]", arg98_1: "f32[4096]", arg99_1: "f32[1024, 4096]", arg100_1: "f32[1024]", arg101_1: "f32[1024]", arg102_1: "f32[1024]", arg103_1: "f32[1024, 1024]", arg104_1: "f32[1024]", arg105_1: "f32[1024, 1024]", arg106_1: "f32[1024]", arg107_1: "f32[1024, 1024]", arg108_1: "f32[1024]", arg109_1: "f32[1024, 1024]", arg110_1: "f32[1024]", arg111_1: "f32[1024]", arg112_1: "f32[1024]", arg113_1: "f32[4096, 1024]", arg114_1: "f32[4096]", arg115_1: "f32[1024, 4096]", arg116_1: "f32[1024]", arg117_1: "f32[1024]", arg118_1: "f32[1024]", arg119_1: "f32[1024, 1024]", arg120_1: "f32[1024]", arg121_1: "f32[1024, 1024]", arg122_1: "f32[1024]", arg123_1: "f32[1024, 1024]", arg124_1: "f32[1024]", arg125_1: "f32[1024, 1024]", arg126_1: "f32[1024]", arg127_1: "f32[1024]", arg128_1: "f32[1024]", arg129_1: "f32[4096, 1024]", arg130_1: "f32[4096]", arg131_1: "f32[1024, 4096]", arg132_1: "f32[1024]", arg133_1: "f32[1024]", arg134_1: "f32[1024]", arg135_1: "f32[1024, 1024]", arg136_1: "f32[1024]", arg137_1: "f32[1024, 1024]", arg138_1: "f32[1024]", arg139_1: "f32[1024, 1024]", arg140_1: "f32[1024]", arg141_1: "f32[1024, 1024]", arg142_1: "f32[1024]", arg143_1: "f32[1024]", arg144_1: "f32[1024]", arg145_1: "f32[4096, 1024]", arg146_1: "f32[4096]", arg147_1: "f32[1024, 4096]", arg148_1: "f32[1024]", arg149_1: "f32[1024]", arg150_1: "f32[1024]", arg151_1: "f32[1024, 1024]", arg152_1: "f32[1024]", arg153_1: "f32[1024, 1024]", arg154_1: "f32[1024]", arg155_1: "f32[1024, 1024]", arg156_1: "f32[1024]", arg157_1: "f32[1024, 1024]", arg158_1: "f32[1024]", arg159_1: "f32[1024]", arg160_1: "f32[1024]", arg161_1: "f32[4096, 1024]", arg162_1: "f32[4096]", arg163_1: "f32[1024, 4096]", arg164_1: "f32[1024]", arg165_1: "f32[1024]", arg166_1: "f32[1024]", arg167_1: "f32[1024, 1024]", arg168_1: "f32[1024]", arg169_1: "f32[1024, 1024]", arg170_1: "f32[1024]", arg171_1: "f32[1024, 1024]", arg172_1: "f32[1024]", arg173_1: "f32[1024, 1024]", arg174_1: "f32[1024]", arg175_1: "f32[1024]", arg176_1: "f32[1024]", arg177_1: "f32[4096, 1024]", arg178_1: "f32[4096]", arg179_1: "f32[1024, 4096]", arg180_1: "f32[1024]", arg181_1: "f32[1024]", arg182_1: "f32[1024]", arg183_1: "f32[1024, 1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024, 1024]", arg186_1: "f32[1024]", arg187_1: "f32[1024, 1024]", arg188_1: "f32[1024]", arg189_1: "f32[1024, 1024]", arg190_1: "f32[1024]", arg191_1: "f32[1024]", arg192_1: "f32[1024]", arg193_1: "f32[4096, 1024]", arg194_1: "f32[4096]", arg195_1: "f32[1024, 4096]", arg196_1: "f32[1024]", arg197_1: "f32[1024]", arg198_1: "f32[1024]", arg199_1: "i64[8, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:123 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "f32[8, 1024, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg0_1 = None
        mul: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:703 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:107 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_9: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:109 in forward, code: return super().forward(position_ids + self.offset)
        add_5: "i64[1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_9, 2);  unsqueeze_9 = None
        embedding_1: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg2_1, add_5);  arg2_1 = add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:732 in forward, code: hidden_states = inputs_embeds + position_ids.to(inputs_embeds.device)
        add_6: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:733 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        sub: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_6, getitem_1);  add_6 = getitem_1 = None
        add_7: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_8: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 1024, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_8, getitem_3);  getitem_3 = None
        add_9: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_3: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_4: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
        add_10: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_4, arg6_1);  mul_4 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_10, [8192, 1024])
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg8_1, view, permute);  arg8_1 = view = permute = None
        view_1: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm, [8, 1024, 1024]);  addmm = None
        view_2: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_1, [8, 1024, -1, 64]);  view_1 = None
        permute_1: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_3: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_10, [8192, 1024])
        permute_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg10_1, view_3, permute_2);  arg10_1 = view_3 = permute_2 = None
        view_4: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_1, [8, 1024, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_4, [8, 1024, -1, 64]);  view_4 = None
        permute_4: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_5: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_10, [8192, 1024]);  add_10 = None
        permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg12_1, view_5, permute_3);  arg12_1 = view_5 = permute_3 = None
        view_6: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_2, [8, 1024, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_6, [8, 1024, -1, 64]);  view_6 = None
        permute_5: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[1024]" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_4: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[1024]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_1: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_2);  unsqueeze_5 = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le, [8, -1, 1024, 1024]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_1, full_default);  full_default_1 = full_default = None
        expand_2: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where, [8, 16, 1024, 1024]);  where = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_4, permute_5, expand_2, False, scale = 0.125);  permute_1 = permute_4 = permute_5 = expand_2 = None
        getitem_4: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_6, [8, 1024, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_9, [8192, 1024]);  view_9 = None
        permute_7: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_3: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg14_1, view_10, permute_7);  arg14_1 = view_10 = permute_7 = None
        view_11: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_3, [8, 1024, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_11: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_8, view_11);  add_8 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 1024, 1]" = var_mean_2[0]
        getitem_9: "f32[8, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_11, getitem_9);  getitem_9 = None
        add_12: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_5: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_6: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_5, arg15_1);  mul_5 = arg15_1 = None
        add_13: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_6, arg16_1);  mul_6 = arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_12: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_13, [8192, 1024]);  add_13 = None
        permute_8: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_4: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg18_1, view_12, permute_8);  arg18_1 = view_12 = permute_8 = None
        view_13: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_4, [8, 1024, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_7: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_8: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_14: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_7, add_14);  mul_7 = add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_14: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_9, [8192, 4096]);  mul_9 = None
        permute_9: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_5: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg20_1, view_14, permute_9);  arg20_1 = view_14 = permute_9 = None
        view_15: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_5, [8, 1024, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_15: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_11, view_15);  add_11 = view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_10: "f32[8, 1024, 1]" = var_mean_3[0]
        getitem_11: "f32[8, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_15, getitem_11);  getitem_11 = None
        add_16: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_3: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_10: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_17: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_16: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_17, [8192, 1024])
        permute_10: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_6: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg24_1, view_16, permute_10);  arg24_1 = view_16 = permute_10 = None
        view_17: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_6, [8, 1024, 1024]);  addmm_6 = None
        view_18: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_17, [8, 1024, -1, 64]);  view_17 = None
        permute_11: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_19: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_17, [8192, 1024])
        permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_7: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg26_1, view_19, permute_12);  arg26_1 = view_19 = permute_12 = None
        view_20: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_7, [8, 1024, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_23: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_20, [8, 1024, -1, 64]);  view_20 = None
        permute_14: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_21: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_17, [8192, 1024]);  add_17 = None
        permute_13: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_8: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg28_1, view_21, permute_13);  arg28_1 = view_21 = permute_13 = None
        view_22: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_8, [8, 1024, 1024]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_24: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_22, [8, 1024, -1, 64]);  view_22 = None
        permute_15: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_3, full_default_2);  full_default_3 = full_default_2 = None
        expand_3: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_1, [8, 16, 1024, 1024]);  where_1 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_14, permute_15, expand_3, False, scale = 0.125);  permute_11 = permute_14 = permute_15 = expand_3 = None
        getitem_12: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_16: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_25: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_16, [8, 1024, -1]);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_26: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_25, [8192, 1024]);  view_25 = None
        permute_17: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_9: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg30_1, view_26, permute_17);  arg30_1 = view_26 = permute_17 = None
        view_27: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_9, [8, 1024, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_18: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_15, view_27);  add_15 = view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 1024, 1]" = var_mean_4[0]
        getitem_17: "f32[8, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_18, getitem_17);  getitem_17 = None
        add_19: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_12: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_13: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_12, arg31_1);  mul_12 = arg31_1 = None
        add_20: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_13, arg32_1);  mul_13 = arg32_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_28: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_20, [8192, 1024]);  add_20 = None
        permute_18: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_10: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg34_1, view_28, permute_18);  arg34_1 = view_28 = permute_18 = None
        view_29: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_10, [8, 1024, 4096]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_14: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_29, 0.5)
        mul_15: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_29, 0.7071067811865476);  view_29 = None
        erf_1: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_21: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_14, add_21);  mul_14 = add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_30: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_16, [8192, 4096]);  mul_16 = None
        permute_19: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_11: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg36_1, view_30, permute_19);  arg36_1 = view_30 = permute_19 = None
        view_31: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_11, [8, 1024, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_22: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_18, view_31);  add_18 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_18: "f32[8, 1024, 1]" = var_mean_5[0]
        getitem_19: "f32[8, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_22, getitem_19);  getitem_19 = None
        add_23: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_5: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_17: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_18: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg37_1);  mul_17 = arg37_1 = None
        add_24: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_18, arg38_1);  mul_18 = arg38_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_32: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_24, [8192, 1024])
        permute_20: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_12: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg40_1, view_32, permute_20);  arg40_1 = view_32 = permute_20 = None
        view_33: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_12, [8, 1024, 1024]);  addmm_12 = None
        view_34: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_33, [8, 1024, -1, 64]);  view_33 = None
        permute_21: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_35: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_24, [8192, 1024])
        permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_13: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg42_1, view_35, permute_22);  arg42_1 = view_35 = permute_22 = None
        view_36: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_13, [8, 1024, 1024]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_39: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_36, [8, 1024, -1, 64]);  view_36 = None
        permute_24: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_37: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_24, [8192, 1024]);  add_24 = None
        permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_14: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg44_1, view_37, permute_23);  arg44_1 = view_37 = permute_23 = None
        view_38: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_14, [8, 1024, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_40: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_38, [8, 1024, -1, 64]);  view_38 = None
        permute_25: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_5, full_default_4);  full_default_5 = full_default_4 = None
        expand_4: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_2, [8, 16, 1024, 1024]);  where_2 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_24, permute_25, expand_4, False, scale = 0.125);  permute_21 = permute_24 = permute_25 = expand_4 = None
        getitem_20: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_26: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3]);  getitem_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_41: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_26, [8, 1024, -1]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_42: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_41, [8192, 1024]);  view_41 = None
        permute_27: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_15: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg46_1, view_42, permute_27);  arg46_1 = view_42 = permute_27 = None
        view_43: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_15, [8, 1024, 1024]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_25: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_22, view_43);  add_22 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 1024, 1]" = var_mean_6[0]
        getitem_25: "f32[8, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_25, getitem_25);  getitem_25 = None
        add_26: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_19: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_20: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_19, arg47_1);  mul_19 = arg47_1 = None
        add_27: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_20, arg48_1);  mul_20 = arg48_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_44: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_27, [8192, 1024]);  add_27 = None
        permute_28: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_16: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg50_1, view_44, permute_28);  arg50_1 = view_44 = permute_28 = None
        view_45: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_16, [8, 1024, 4096]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_21: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_22: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
        add_28: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_23: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_21, add_28);  mul_21 = add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_46: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_23, [8192, 4096]);  mul_23 = None
        permute_29: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_17: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg52_1, view_46, permute_29);  arg52_1 = view_46 = permute_29 = None
        view_47: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_17, [8, 1024, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_29: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_25, view_47);  add_25 = view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 1024, 1]" = var_mean_7[0]
        getitem_27: "f32[8, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_29, getitem_27);  getitem_27 = None
        add_30: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_7: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_24: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_25: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_24, arg53_1);  mul_24 = arg53_1 = None
        add_31: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_25, arg54_1);  mul_25 = arg54_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_48: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_31, [8192, 1024])
        permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_18: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg56_1, view_48, permute_30);  arg56_1 = view_48 = permute_30 = None
        view_49: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_18, [8, 1024, 1024]);  addmm_18 = None
        view_50: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_49, [8, 1024, -1, 64]);  view_49 = None
        permute_31: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_51: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_31, [8192, 1024])
        permute_32: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_19: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg58_1, view_51, permute_32);  arg58_1 = view_51 = permute_32 = None
        view_52: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_19, [8, 1024, 1024]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_55: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_52, [8, 1024, -1, 64]);  view_52 = None
        permute_34: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_53: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_31, [8192, 1024]);  add_31 = None
        permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_20: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg60_1, view_53, permute_33);  arg60_1 = view_53 = permute_33 = None
        view_54: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_20, [8, 1024, 1024]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_56: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_54, [8, 1024, -1, 64]);  view_54 = None
        permute_35: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        expand_5: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_3, [8, 16, 1024, 1024]);  where_3 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_34, permute_35, expand_5, False, scale = 0.125);  permute_31 = permute_34 = permute_35 = expand_5 = None
        getitem_28: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_36: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_57: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_36, [8, 1024, -1]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_58: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_57, [8192, 1024]);  view_57 = None
        permute_37: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_21: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg62_1, view_58, permute_37);  arg62_1 = view_58 = permute_37 = None
        view_59: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_21, [8, 1024, 1024]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_32: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_29, view_59);  add_29 = view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 1024, 1]" = var_mean_8[0]
        getitem_33: "f32[8, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_8: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_32, getitem_33);  getitem_33 = None
        add_33: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_8: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_26: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_27: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_26, arg63_1);  mul_26 = arg63_1 = None
        add_34: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_27, arg64_1);  mul_27 = arg64_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_60: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_34, [8192, 1024]);  add_34 = None
        permute_38: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_22: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg66_1, view_60, permute_38);  arg66_1 = view_60 = permute_38 = None
        view_61: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_22, [8, 1024, 4096]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_28: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_29: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
        erf_3: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
        add_35: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_30: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_28, add_35);  mul_28 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_62: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_30, [8192, 4096]);  mul_30 = None
        permute_39: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_23: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg68_1, view_62, permute_39);  arg68_1 = view_62 = permute_39 = None
        view_63: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_23, [8, 1024, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_36: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_32, view_63);  add_32 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_34: "f32[8, 1024, 1]" = var_mean_9[0]
        getitem_35: "f32[8, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_36, getitem_35);  getitem_35 = None
        add_37: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_9: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_31: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_32: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_31, arg69_1);  mul_31 = arg69_1 = None
        add_38: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_32, arg70_1);  mul_32 = arg70_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_64: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_38, [8192, 1024])
        permute_40: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_24: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg72_1, view_64, permute_40);  arg72_1 = view_64 = permute_40 = None
        view_65: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_24, [8, 1024, 1024]);  addmm_24 = None
        view_66: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_65, [8, 1024, -1, 64]);  view_65 = None
        permute_41: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_67: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_38, [8192, 1024])
        permute_42: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_25: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg74_1, view_67, permute_42);  arg74_1 = view_67 = permute_42 = None
        view_68: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_25, [8, 1024, 1024]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_71: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_68, [8, 1024, -1, 64]);  view_68 = None
        permute_44: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_69: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_38, [8192, 1024]);  add_38 = None
        permute_43: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_26: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg76_1, view_69, permute_43);  arg76_1 = view_69 = permute_43 = None
        view_70: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_26, [8, 1024, 1024]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_72: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_70, [8, 1024, -1, 64]);  view_70 = None
        permute_45: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_9: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_9, full_default_8);  full_default_9 = full_default_8 = None
        expand_6: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_4, [8, 16, 1024, 1024]);  where_4 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_44, permute_45, expand_6, False, scale = 0.125);  permute_41 = permute_44 = permute_45 = expand_6 = None
        getitem_36: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_46: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_73: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_46, [8, 1024, -1]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_74: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_73, [8192, 1024]);  view_73 = None
        permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_27: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg78_1, view_74, permute_47);  arg78_1 = view_74 = permute_47 = None
        view_75: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_27, [8, 1024, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_39: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_36, view_75);  add_36 = view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 1024, 1]" = var_mean_10[0]
        getitem_41: "f32[8, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_10: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_39, getitem_41);  getitem_41 = None
        add_40: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_10: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_33: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_34: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_33, arg79_1);  mul_33 = arg79_1 = None
        add_41: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_34, arg80_1);  mul_34 = arg80_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_76: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_41, [8192, 1024]);  add_41 = None
        permute_48: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_28: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg82_1, view_76, permute_48);  arg82_1 = view_76 = permute_48 = None
        view_77: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_28, [8, 1024, 4096]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_35: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        mul_36: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_77, 0.7071067811865476);  view_77 = None
        erf_4: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_42: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_37: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_35, add_42);  mul_35 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_78: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_37, [8192, 4096]);  mul_37 = None
        permute_49: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_29: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg84_1, view_78, permute_49);  arg84_1 = view_78 = permute_49 = None
        view_79: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_29, [8, 1024, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_43: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_39, view_79);  add_39 = view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 1024, 1]" = var_mean_11[0]
        getitem_43: "f32[8, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_11: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_43, getitem_43);  getitem_43 = None
        add_44: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_11: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_38: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_39: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_38, arg85_1);  mul_38 = arg85_1 = None
        add_45: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_39, arg86_1);  mul_39 = arg86_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_80: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_45, [8192, 1024])
        permute_50: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_30: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg88_1, view_80, permute_50);  arg88_1 = view_80 = permute_50 = None
        view_81: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_30, [8, 1024, 1024]);  addmm_30 = None
        view_82: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_81, [8, 1024, -1, 64]);  view_81 = None
        permute_51: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_83: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_45, [8192, 1024])
        permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_31: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg90_1, view_83, permute_52);  arg90_1 = view_83 = permute_52 = None
        view_84: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_31, [8, 1024, 1024]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_87: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_84, [8, 1024, -1, 64]);  view_84 = None
        permute_54: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_85: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_45, [8192, 1024]);  add_45 = None
        permute_53: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_32: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg92_1, view_85, permute_53);  arg92_1 = view_85 = permute_53 = None
        view_86: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_32, [8, 1024, 1024]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_88: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_86, [8, 1024, -1, 64]);  view_86 = None
        permute_55: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_11: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_11, full_default_10);  full_default_11 = full_default_10 = None
        expand_7: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_5, [8, 16, 1024, 1024]);  where_5 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_54, permute_55, expand_7, False, scale = 0.125);  permute_51 = permute_54 = permute_55 = expand_7 = None
        getitem_44: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_44, [0, 2, 1, 3]);  getitem_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_89: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_56, [8, 1024, -1]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_90: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_89, [8192, 1024]);  view_89 = None
        permute_57: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_33: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg94_1, view_90, permute_57);  arg94_1 = view_90 = permute_57 = None
        view_91: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_33, [8, 1024, 1024]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_46: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_43, view_91);  add_43 = view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_48: "f32[8, 1024, 1]" = var_mean_12[0]
        getitem_49: "f32[8, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_12: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_46, getitem_49);  getitem_49 = None
        add_47: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_12: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_40: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_41: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_40, arg95_1);  mul_40 = arg95_1 = None
        add_48: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_41, arg96_1);  mul_41 = arg96_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_92: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_48, [8192, 1024]);  add_48 = None
        permute_58: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_34: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg98_1, view_92, permute_58);  arg98_1 = view_92 = permute_58 = None
        view_93: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_34, [8, 1024, 4096]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_42: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_43: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_93, 0.7071067811865476);  view_93 = None
        erf_5: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_49: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_44: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_42, add_49);  mul_42 = add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_94: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_44, [8192, 4096]);  mul_44 = None
        permute_59: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_35: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg100_1, view_94, permute_59);  arg100_1 = view_94 = permute_59 = None
        view_95: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_35, [8, 1024, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_50: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_46, view_95);  add_46 = view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_50: "f32[8, 1024, 1]" = var_mean_13[0]
        getitem_51: "f32[8, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_13: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_50, getitem_51);  getitem_51 = None
        add_51: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_13: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_45: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_46: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_45, arg101_1);  mul_45 = arg101_1 = None
        add_52: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_46, arg102_1);  mul_46 = arg102_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_96: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_52, [8192, 1024])
        permute_60: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_36: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg104_1, view_96, permute_60);  arg104_1 = view_96 = permute_60 = None
        view_97: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_36, [8, 1024, 1024]);  addmm_36 = None
        view_98: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_97, [8, 1024, -1, 64]);  view_97 = None
        permute_61: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_99: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_52, [8192, 1024])
        permute_62: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_37: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg106_1, view_99, permute_62);  arg106_1 = view_99 = permute_62 = None
        view_100: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_37, [8, 1024, 1024]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_103: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_100, [8, 1024, -1, 64]);  view_100 = None
        permute_64: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_101: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_52, [8192, 1024]);  add_52 = None
        permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_38: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg108_1, view_101, permute_63);  arg108_1 = view_101 = permute_63 = None
        view_102: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_38, [8, 1024, 1024]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_104: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_102, [8, 1024, -1, 64]);  view_102 = None
        permute_65: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_13: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        expand_8: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_6, [8, 16, 1024, 1024]);  where_6 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_61, permute_64, permute_65, expand_8, False, scale = 0.125);  permute_61 = permute_64 = permute_65 = expand_8 = None
        getitem_52: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_66: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_66, [8, 1024, -1]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_106: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_105, [8192, 1024]);  view_105 = None
        permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_39: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg110_1, view_106, permute_67);  arg110_1 = view_106 = permute_67 = None
        view_107: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_39, [8, 1024, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_53: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_50, view_107);  add_50 = view_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_56: "f32[8, 1024, 1]" = var_mean_14[0]
        getitem_57: "f32[8, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_14: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_53, getitem_57);  getitem_57 = None
        add_54: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_14: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_47: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_48: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_47, arg111_1);  mul_47 = arg111_1 = None
        add_55: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_48, arg112_1);  mul_48 = arg112_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_108: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_55, [8192, 1024]);  add_55 = None
        permute_68: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_40: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg114_1, view_108, permute_68);  arg114_1 = view_108 = permute_68 = None
        view_109: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_40, [8, 1024, 4096]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_49: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_50: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_6: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_56: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_51: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_49, add_56);  mul_49 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_110: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_51, [8192, 4096]);  mul_51 = None
        permute_69: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_41: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg116_1, view_110, permute_69);  arg116_1 = view_110 = permute_69 = None
        view_111: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_41, [8, 1024, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_57: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_53, view_111);  add_53 = view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_58: "f32[8, 1024, 1]" = var_mean_15[0]
        getitem_59: "f32[8, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_57, getitem_59);  getitem_59 = None
        add_58: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_15: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_52: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_53: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_52, arg117_1);  mul_52 = arg117_1 = None
        add_59: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_53, arg118_1);  mul_53 = arg118_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_112: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_59, [8192, 1024])
        permute_70: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_42: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg120_1, view_112, permute_70);  arg120_1 = view_112 = permute_70 = None
        view_113: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_42, [8, 1024, 1024]);  addmm_42 = None
        view_114: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_113, [8, 1024, -1, 64]);  view_113 = None
        permute_71: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_115: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_59, [8192, 1024])
        permute_72: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_43: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg122_1, view_115, permute_72);  arg122_1 = view_115 = permute_72 = None
        view_116: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_43, [8, 1024, 1024]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_119: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_116, [8, 1024, -1, 64]);  view_116 = None
        permute_74: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_117: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_59, [8192, 1024]);  add_59 = None
        permute_73: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_44: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg124_1, view_117, permute_73);  arg124_1 = view_117 = permute_73 = None
        view_118: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_44, [8, 1024, 1024]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_120: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_118, [8, 1024, -1, 64]);  view_118 = None
        permute_75: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_15: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_15, full_default_14);  full_default_15 = full_default_14 = None
        expand_9: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_7, [8, 16, 1024, 1024]);  where_7 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_71, permute_74, permute_75, expand_9, False, scale = 0.125);  permute_71 = permute_74 = permute_75 = expand_9 = None
        getitem_60: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_76: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_121: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_76, [8, 1024, -1]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_122: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_121, [8192, 1024]);  view_121 = None
        permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_45: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg126_1, view_122, permute_77);  arg126_1 = view_122 = permute_77 = None
        view_123: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_45, [8, 1024, 1024]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_60: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_57, view_123);  add_57 = view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_64: "f32[8, 1024, 1]" = var_mean_16[0]
        getitem_65: "f32[8, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_16: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_60, getitem_65);  getitem_65 = None
        add_61: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_16: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_54: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_55: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_54, arg127_1);  mul_54 = arg127_1 = None
        add_62: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_55, arg128_1);  mul_55 = arg128_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_124: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_62, [8192, 1024]);  add_62 = None
        permute_78: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_46: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg130_1, view_124, permute_78);  arg130_1 = view_124 = permute_78 = None
        view_125: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_46, [8, 1024, 4096]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_56: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
        mul_57: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_125, 0.7071067811865476);  view_125 = None
        erf_7: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
        add_63: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_58: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_56, add_63);  mul_56 = add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_126: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_58, [8192, 4096]);  mul_58 = None
        permute_79: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_47: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg132_1, view_126, permute_79);  arg132_1 = view_126 = permute_79 = None
        view_127: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_47, [8, 1024, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_64: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_60, view_127);  add_60 = view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_66: "f32[8, 1024, 1]" = var_mean_17[0]
        getitem_67: "f32[8, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_17: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_64, getitem_67);  getitem_67 = None
        add_65: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_17: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_59: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_60: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_59, arg133_1);  mul_59 = arg133_1 = None
        add_66: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_60, arg134_1);  mul_60 = arg134_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_128: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_66, [8192, 1024])
        permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_48: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg136_1, view_128, permute_80);  arg136_1 = view_128 = permute_80 = None
        view_129: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_48, [8, 1024, 1024]);  addmm_48 = None
        view_130: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_129, [8, 1024, -1, 64]);  view_129 = None
        permute_81: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_131: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_66, [8192, 1024])
        permute_82: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_49: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg138_1, view_131, permute_82);  arg138_1 = view_131 = permute_82 = None
        view_132: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_49, [8, 1024, 1024]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_135: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_132, [8, 1024, -1, 64]);  view_132 = None
        permute_84: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_133: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_66, [8192, 1024]);  add_66 = None
        permute_83: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_50: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg140_1, view_133, permute_83);  arg140_1 = view_133 = permute_83 = None
        view_134: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_50, [8, 1024, 1024]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_136: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_134, [8, 1024, -1, 64]);  view_134 = None
        permute_85: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_17: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_17, full_default_16);  full_default_17 = full_default_16 = None
        expand_10: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_8, [8, 16, 1024, 1024]);  where_8 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_81, permute_84, permute_85, expand_10, False, scale = 0.125);  permute_81 = permute_84 = permute_85 = expand_10 = None
        getitem_68: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_86: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_68, [0, 2, 1, 3]);  getitem_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_137: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_86, [8, 1024, -1]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_138: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_137, [8192, 1024]);  view_137 = None
        permute_87: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_51: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg142_1, view_138, permute_87);  arg142_1 = view_138 = permute_87 = None
        view_139: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_51, [8, 1024, 1024]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_67: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_64, view_139);  add_64 = view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_72: "f32[8, 1024, 1]" = var_mean_18[0]
        getitem_73: "f32[8, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_67, getitem_73);  getitem_73 = None
        add_68: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_18: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_61: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_62: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_61, arg143_1);  mul_61 = arg143_1 = None
        add_69: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_62, arg144_1);  mul_62 = arg144_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_140: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_69, [8192, 1024]);  add_69 = None
        permute_88: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_52: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg146_1, view_140, permute_88);  arg146_1 = view_140 = permute_88 = None
        view_141: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_52, [8, 1024, 4096]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_63: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_64: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_8: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_70: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_65: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_63, add_70);  mul_63 = add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_142: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_65, [8192, 4096]);  mul_65 = None
        permute_89: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_53: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg148_1, view_142, permute_89);  arg148_1 = view_142 = permute_89 = None
        view_143: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_53, [8, 1024, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_71: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_67, view_143);  add_67 = view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_74: "f32[8, 1024, 1]" = var_mean_19[0]
        getitem_75: "f32[8, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_19: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_71, getitem_75);  getitem_75 = None
        add_72: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_19: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_66: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_67: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_66, arg149_1);  mul_66 = arg149_1 = None
        add_73: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_67, arg150_1);  mul_67 = arg150_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_144: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_73, [8192, 1024])
        permute_90: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_54: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg152_1, view_144, permute_90);  arg152_1 = view_144 = permute_90 = None
        view_145: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_54, [8, 1024, 1024]);  addmm_54 = None
        view_146: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_145, [8, 1024, -1, 64]);  view_145 = None
        permute_91: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_147: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_73, [8192, 1024])
        permute_92: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_55: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg154_1, view_147, permute_92);  arg154_1 = view_147 = permute_92 = None
        view_148: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_55, [8, 1024, 1024]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_151: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_148, [8, 1024, -1, 64]);  view_148 = None
        permute_94: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_149: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_73, [8192, 1024]);  add_73 = None
        permute_93: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_56: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg156_1, view_149, permute_93);  arg156_1 = view_149 = permute_93 = None
        view_150: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_56, [8, 1024, 1024]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_152: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_150, [8, 1024, -1, 64]);  view_150 = None
        permute_95: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_19: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        expand_11: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_9, [8, 16, 1024, 1024]);  where_9 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_91, permute_94, permute_95, expand_11, False, scale = 0.125);  permute_91 = permute_94 = permute_95 = expand_11 = None
        getitem_76: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_96: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_76, [0, 2, 1, 3]);  getitem_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_153: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_96, [8, 1024, -1]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_154: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_153, [8192, 1024]);  view_153 = None
        permute_97: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_57: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg158_1, view_154, permute_97);  arg158_1 = view_154 = permute_97 = None
        view_155: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_57, [8, 1024, 1024]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_74: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_71, view_155);  add_71 = view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_80: "f32[8, 1024, 1]" = var_mean_20[0]
        getitem_81: "f32[8, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_20: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_74, getitem_81);  getitem_81 = None
        add_75: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_20: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_68: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_69: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_68, arg159_1);  mul_68 = arg159_1 = None
        add_76: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_69, arg160_1);  mul_69 = arg160_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_156: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_76, [8192, 1024]);  add_76 = None
        permute_98: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_58: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg162_1, view_156, permute_98);  arg162_1 = view_156 = permute_98 = None
        view_157: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_58, [8, 1024, 4096]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_70: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_157, 0.5)
        mul_71: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_157, 0.7071067811865476);  view_157 = None
        erf_9: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_77: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_72: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_70, add_77);  mul_70 = add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_158: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_72, [8192, 4096]);  mul_72 = None
        permute_99: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_59: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg164_1, view_158, permute_99);  arg164_1 = view_158 = permute_99 = None
        view_159: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_59, [8, 1024, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_78: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_74, view_159);  add_74 = view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_82: "f32[8, 1024, 1]" = var_mean_21[0]
        getitem_83: "f32[8, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_21: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_78, getitem_83);  getitem_83 = None
        add_79: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_21: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_73: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_74: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_73, arg165_1);  mul_73 = arg165_1 = None
        add_80: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_74, arg166_1);  mul_74 = arg166_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_160: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_80, [8192, 1024])
        permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_60: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg168_1, view_160, permute_100);  arg168_1 = view_160 = permute_100 = None
        view_161: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_60, [8, 1024, 1024]);  addmm_60 = None
        view_162: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_161, [8, 1024, -1, 64]);  view_161 = None
        permute_101: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_163: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_80, [8192, 1024])
        permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_61: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg170_1, view_163, permute_102);  arg170_1 = view_163 = permute_102 = None
        view_164: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_61, [8, 1024, 1024]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_167: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_164, [8, 1024, -1, 64]);  view_164 = None
        permute_104: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_165: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_80, [8192, 1024]);  add_80 = None
        permute_103: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_62: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg172_1, view_165, permute_103);  arg172_1 = view_165 = permute_103 = None
        view_166: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_62, [8, 1024, 1024]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_168: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_166, [8, 1024, -1, 64]);  view_166 = None
        permute_105: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_21: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_21, full_default_20);  full_default_21 = full_default_20 = None
        expand_12: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_10, [8, 16, 1024, 1024]);  where_10 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_101, permute_104, permute_105, expand_12, False, scale = 0.125);  permute_101 = permute_104 = permute_105 = expand_12 = None
        getitem_84: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_84, [0, 2, 1, 3]);  getitem_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_106, [8, 1024, -1]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_170: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_169, [8192, 1024]);  view_169 = None
        permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_63: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg174_1, view_170, permute_107);  arg174_1 = view_170 = permute_107 = None
        view_171: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_63, [8, 1024, 1024]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_81: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_78, view_171);  add_78 = view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_88: "f32[8, 1024, 1]" = var_mean_22[0]
        getitem_89: "f32[8, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_22: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_81, getitem_89);  getitem_89 = None
        add_82: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_22: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_75: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_76: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_75, arg175_1);  mul_75 = arg175_1 = None
        add_83: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_76, arg176_1);  mul_76 = arg176_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_172: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_83, [8192, 1024]);  add_83 = None
        permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_64: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg178_1, view_172, permute_108);  arg178_1 = view_172 = permute_108 = None
        view_173: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_64, [8, 1024, 4096]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_77: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_78: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_10: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_84: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_79: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_77, add_84);  mul_77 = add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_174: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_79, [8192, 4096]);  mul_79 = None
        permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_65: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg180_1, view_174, permute_109);  arg180_1 = view_174 = permute_109 = None
        view_175: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_65, [8, 1024, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_85: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_81, view_175);  add_81 = view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_90: "f32[8, 1024, 1]" = var_mean_23[0]
        getitem_91: "f32[8, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_23: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_85, getitem_91);  getitem_91 = None
        add_86: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_23: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_80: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_81: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_80, arg181_1);  mul_80 = arg181_1 = None
        add_87: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_81, arg182_1);  mul_81 = arg182_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_176: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_87, [8192, 1024])
        permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_66: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg184_1, view_176, permute_110);  arg184_1 = view_176 = permute_110 = None
        view_177: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_66, [8, 1024, 1024]);  addmm_66 = None
        view_178: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_177, [8, 1024, -1, 64]);  view_177 = None
        permute_111: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        view_179: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_87, [8192, 1024])
        permute_112: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_67: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg186_1, view_179, permute_112);  arg186_1 = view_179 = permute_112 = None
        view_180: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_67, [8, 1024, 1024]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:243 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_183: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_180, [8, 1024, -1, 64]);  view_180 = None
        permute_114: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        view_181: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_87, [8192, 1024]);  add_87 = None
        permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_68: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg188_1, view_181, permute_113);  arg188_1 = view_181 = permute_113 = None
        view_182: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_68, [8, 1024, 1024]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:244 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_184: "f32[8, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_182, [8, 1024, -1, 64]);  view_182 = None
        permute_115: "f32[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_23: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_23, full_default_22);  expand = full_default_23 = full_default_22 = None
        expand_13: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_11, [8, 16, 1024, 1024]);  where_11 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_111, permute_114, permute_115, expand_13, False, scale = 0.125);  permute_111 = permute_114 = permute_115 = expand_13 = None
        getitem_92: "f32[8, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_116: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_92, [0, 2, 1, 3]);  getitem_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:267 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_185: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_116, [8, 1024, -1]);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:268 in forward, code: attn_output = self.out_proj(attn_output)
        view_186: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_185, [8192, 1024]);  view_185 = None
        permute_117: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_69: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg190_1, view_186, permute_117);  arg190_1 = view_186 = permute_117 = None
        view_187: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_69, [8, 1024, 1024]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:393 in forward, code: hidden_states = residual + hidden_states
        add_88: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_85, view_187);  add_85 = view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:412 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_96: "f32[8, 1024, 1]" = var_mean_24[0]
        getitem_97: "f32[8, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_24: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_88, getitem_97);  getitem_97 = None
        add_89: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_24: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_82: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_83: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_82, arg191_1);  mul_82 = arg191_1 = None
        add_90: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_83, arg192_1);  mul_83 = arg192_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:413 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_188: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_90, [8192, 1024]);  add_90 = None
        permute_118: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_70: "f32[8192, 4096]" = torch.ops.aten.addmm.default(arg194_1, view_188, permute_118);  arg194_1 = view_188 = permute_118 = None
        view_189: "f32[8, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_70, [8, 1024, 4096]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_84: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_85: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_11: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_85);  mul_85 = None
        add_91: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_86: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_84, add_91);  mul_84 = add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        view_190: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_86, [8192, 4096]);  mul_86 = None
        permute_119: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_71: "f32[8192, 1024]" = torch.ops.aten.addmm.default(arg196_1, view_190, permute_119);  arg196_1 = view_190 = permute_119 = None
        view_191: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_71, [8, 1024, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_92: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_88, view_191);  add_88 = view_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:754 in forward, code: hidden_states = self.layer_norm(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_98: "f32[8, 1024, 1]" = var_mean_25[0]
        getitem_99: "f32[8, 1024, 1]" = var_mean_25[1];  var_mean_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:1377 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_195: "i64[8192]" = torch.ops.aten.reshape.default(arg199_1, [-1]);  arg199_1 = None
        ne_1: "b8[8192]" = torch.ops.aten.ne.Scalar(view_195, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:754 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_25: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_92, getitem_99);  add_92 = getitem_99 = None
        add_93: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_25: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_87: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        mul_88: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_87, arg197_1);  mul_87 = arg197_1 = None
        add_94: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_88, arg198_1);  mul_88 = arg198_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:1371 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_192: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_94, [8192, 1024]);  add_94 = None
        permute_120: "f32[1024, 50265]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "f32[1024, 50268]" = torch.ops.aten.constant_pad_nd.default(permute_120, [0, 3, 0, 0]);  permute_120 = None
        mm_default: "f32[8192, 50268]" = torch.ops.aten.mm.default(view_192, constant_pad_nd_default);  view_192 = constant_pad_nd_default = None
        slice_tensor: "f32[8192, 50265]" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        view_193: "f32[8, 1024, 50265]" = torch.ops.aten.reshape.default(slice_tensor, [8, 1024, 50265]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:1377 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_194: "f32[8192, 50265]" = torch.ops.aten.reshape.default(view_193, [-1, 50265])
        amax: "f32[8192, 1]" = torch.ops.aten.amax.default(view_194, [1], True)
        sub_26: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(view_194, amax);  view_194 = amax = None
        exp: "f32[8192, 50265]" = torch.ops.aten.exp.default(sub_26)
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(view_195, -100)
        full_default_24: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[8192]" = torch.ops.aten.where.self(ne, view_195, full_default_24);  ne = full_default_24 = None
        unsqueeze_10: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze_10);  sub_27 = unsqueeze_10 = None
        squeeze: "f32[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_25: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f32[8192]" = torch.ops.aten.where.self(ne_1, neg, full_default_25);  ne_1 = neg = full_default_25 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        ne_2: "b8[8192]" = torch.ops.aten.ne.Scalar(view_195, -100);  view_195 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = convert_element_type = None
        return (div, view_193)
