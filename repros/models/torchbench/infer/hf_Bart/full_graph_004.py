class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "f16[50265, 768]", arg2_1: "f16[1, 512, 768]", arg3_1: "f16[1026, 768]", arg4_1: "f16[768]", arg5_1: "f16[768]", arg6_1: "f16[768, 768]", arg7_1: "f16[768]", arg8_1: "f16[768, 768]", arg9_1: "f16[768]", arg10_1: "f16[768, 768]", arg11_1: "f16[768]", arg12_1: "f16[768, 768]", arg13_1: "f16[768]", arg14_1: "f16[768]", arg15_1: "f16[768]", arg16_1: "f16[768, 768]", arg17_1: "f16[768]", arg18_1: "f16[768, 768]", arg19_1: "f16[768]", arg20_1: "f16[768, 768]", arg21_1: "f16[768]", arg22_1: "f16[768, 768]", arg23_1: "f16[768]", arg24_1: "f16[768]", arg25_1: "f16[768]", arg26_1: "f16[3072, 768]", arg27_1: "f16[3072]", arg28_1: "f16[768, 3072]", arg29_1: "f16[768]", arg30_1: "f16[768]", arg31_1: "f16[768]", arg32_1: "f16[768, 768]", arg33_1: "f16[768]", arg34_1: "f16[768, 768]", arg35_1: "f16[768]", arg36_1: "f16[768, 768]", arg37_1: "f16[768]", arg38_1: "f16[768, 768]", arg39_1: "f16[768]", arg40_1: "f16[768]", arg41_1: "f16[768]", arg42_1: "f16[768, 768]", arg43_1: "f16[768]", arg44_1: "f16[768, 768]", arg45_1: "f16[768]", arg46_1: "f16[768, 768]", arg47_1: "f16[768]", arg48_1: "f16[768, 768]", arg49_1: "f16[768]", arg50_1: "f16[768]", arg51_1: "f16[768]", arg52_1: "f16[3072, 768]", arg53_1: "f16[3072]", arg54_1: "f16[768, 3072]", arg55_1: "f16[768]", arg56_1: "f16[768]", arg57_1: "f16[768]", arg58_1: "f16[768, 768]", arg59_1: "f16[768]", arg60_1: "f16[768, 768]", arg61_1: "f16[768]", arg62_1: "f16[768, 768]", arg63_1: "f16[768]", arg64_1: "f16[768, 768]", arg65_1: "f16[768]", arg66_1: "f16[768]", arg67_1: "f16[768]", arg68_1: "f16[768, 768]", arg69_1: "f16[768]", arg70_1: "f16[768, 768]", arg71_1: "f16[768]", arg72_1: "f16[768, 768]", arg73_1: "f16[768]", arg74_1: "f16[768, 768]", arg75_1: "f16[768]", arg76_1: "f16[768]", arg77_1: "f16[768]", arg78_1: "f16[3072, 768]", arg79_1: "f16[3072]", arg80_1: "f16[768, 3072]", arg81_1: "f16[768]", arg82_1: "f16[768]", arg83_1: "f16[768]", arg84_1: "f16[768, 768]", arg85_1: "f16[768]", arg86_1: "f16[768, 768]", arg87_1: "f16[768]", arg88_1: "f16[768, 768]", arg89_1: "f16[768]", arg90_1: "f16[768, 768]", arg91_1: "f16[768]", arg92_1: "f16[768]", arg93_1: "f16[768]", arg94_1: "f16[768, 768]", arg95_1: "f16[768]", arg96_1: "f16[768, 768]", arg97_1: "f16[768]", arg98_1: "f16[768, 768]", arg99_1: "f16[768]", arg100_1: "f16[768, 768]", arg101_1: "f16[768]", arg102_1: "f16[768]", arg103_1: "f16[768]", arg104_1: "f16[3072, 768]", arg105_1: "f16[3072]", arg106_1: "f16[768, 3072]", arg107_1: "f16[768]", arg108_1: "f16[768]", arg109_1: "f16[768]", arg110_1: "f16[768, 768]", arg111_1: "f16[768]", arg112_1: "f16[768, 768]", arg113_1: "f16[768]", arg114_1: "f16[768, 768]", arg115_1: "f16[768]", arg116_1: "f16[768, 768]", arg117_1: "f16[768]", arg118_1: "f16[768]", arg119_1: "f16[768]", arg120_1: "f16[768, 768]", arg121_1: "f16[768]", arg122_1: "f16[768, 768]", arg123_1: "f16[768]", arg124_1: "f16[768, 768]", arg125_1: "f16[768]", arg126_1: "f16[768, 768]", arg127_1: "f16[768]", arg128_1: "f16[768]", arg129_1: "f16[768]", arg130_1: "f16[3072, 768]", arg131_1: "f16[3072]", arg132_1: "f16[768, 3072]", arg133_1: "f16[768]", arg134_1: "f16[768]", arg135_1: "f16[768]", arg136_1: "f16[768, 768]", arg137_1: "f16[768]", arg138_1: "f16[768, 768]", arg139_1: "f16[768]", arg140_1: "f16[768, 768]", arg141_1: "f16[768]", arg142_1: "f16[768, 768]", arg143_1: "f16[768]", arg144_1: "f16[768]", arg145_1: "f16[768]", arg146_1: "f16[768, 768]", arg147_1: "f16[768]", arg148_1: "f16[768, 768]", arg149_1: "f16[768]", arg150_1: "f16[768, 768]", arg151_1: "f16[768]", arg152_1: "f16[768, 768]", arg153_1: "f16[768]", arg154_1: "f16[768]", arg155_1: "f16[768]", arg156_1: "f16[3072, 768]", arg157_1: "f16[3072]", arg158_1: "f16[768, 3072]", arg159_1: "f16[768]", arg160_1: "f16[768]", arg161_1: "f16[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:111 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul: "f16[1, 512, 768]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:621 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:96 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_9: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:98 in forward, code: return super().forward(position_ids + self.offset)
        add_5: "i64[1, 512]" = torch.ops.aten.add.Tensor(unsqueeze_9, 2);  unsqueeze_9 = None
        embedding_1: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, add_5);  arg3_1 = add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:651 in forward, code: hidden_states = inputs_embeds + positions
        add_6: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:652 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        convert_element_type: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32);  add_6 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean[0]
        getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_7: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        add_8: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None
        convert_element_type_1: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_8, torch.float16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_1, [512, 768])
        permute: "f16[768, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "f16[512, 768]" = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        view_1: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm, [1, 512, 768]);  addmm = None
        view_2: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [1, 512, -1, 64]);  view_1 = None
        permute_1: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_3: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_1, [512, 768])
        permute_2: "f16[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "f16[512, 768]" = torch.ops.aten.addmm.default(arg9_1, view_3, permute_2);  arg9_1 = view_3 = permute_2 = None
        view_4: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [1, 512, -1, 64]);  view_4 = None
        permute_4: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_5: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_1, [512, 768])
        permute_3: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_5, permute_3);  arg11_1 = view_5 = permute_3 = None
        view_6: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [1, 512, 768]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_6, [1, 512, -1, 64]);  view_6 = None
        permute_5: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[512]" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_3: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_4: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[512]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_2);  unsqueeze_5 = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 512, 512]" = torch.ops.aten.expand.default(le, [1, -1, 512, 512]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_1, full_default);  full_default_1 = full_default = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_1, permute_4, permute_5, where, False, scale = 0.125);  permute_1 = permute_4 = permute_5 = where = None
        getitem_2: "f16[1, 12, 512, 64]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(permute_6, [1, 512, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f16[512, 768]" = torch.ops.aten.reshape.default(view_9, [512, 768]);  view_9 = None
        permute_7: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_10, permute_7);  arg13_1 = view_10 = permute_7 = None
        view_11: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:363 in forward, code: hidden_states = residual + hidden_states
        add_9: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_1, view_11);  convert_element_type_1 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:364 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_14, [2], correction = 0, keepdim = True)
        getitem_11: "f32[1, 512, 1]" = var_mean_1[0]
        getitem_12: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_12);  convert_element_type_14 = getitem_12 = None
        add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg14_1);  mul_3 = arg14_1 = None
        add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, arg15_1);  mul_4 = arg15_1 = None
        convert_element_type_15: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_11, torch.float16);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_12: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_15, [512, 768])
        permute_8: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_12, permute_8);  arg17_1 = view_12 = permute_8 = None
        view_13: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 768]);  addmm_4 = None
        view_14: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_13, [1, 512, -1, 64]);  view_13 = None
        permute_9: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_25: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_9, torch.float32);  permute_9 = None
        mul_5: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_25, 0.3535533905932738);  convert_element_type_25 = None
        expand_2: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_5, [1, 12, 512, 64]);  mul_5 = None
        view_21: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_2, [12, 512, 64]);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_15: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768])
        permute_10: "f16[768, 768]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5: "f16[512, 768]" = torch.ops.aten.addmm.default(arg19_1, view_15, permute_10);  arg19_1 = view_15 = permute_10 = None
        view_16: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_19: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_16, [1, 512, -1, 64]);  view_16 = None
        permute_12: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_26: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_12, torch.float32);  permute_12 = None
        permute_14: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_26, [0, 1, 3, 2]);  convert_element_type_26 = None
        mul_6: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_14, 0.3535533905932738);  permute_14 = None
        expand_3: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_6, [1, 12, 64, 512]);  mul_6 = None
        view_22: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_3, [12, 64, 512]);  expand_3 = None
        bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_21, view_22);  view_21 = view_22 = None
        view_23: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [1, 12, 512, 512]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_7: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[512]" = torch.ops.aten.add.Tensor(iota_7, 0);  iota_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_6: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_7: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 512, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_8, 0);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[1, 1, 512, 512]" = torch.ops.aten.expand.default(ge, [1, -1, 512, 512]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_3: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_3, full_default_2);  full_default_3 = full_default_2 = None
        add_12: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_23, where_1);  view_23 = where_1 = None
        eq: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_12, -inf)
        logical_not: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_4: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_12, [-1], True)
        sub_2: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_12, amax);  add_12 = amax = None
        exp: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        where_2: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_4, div);  logical_not_1 = full_default_4 = div = None
        expand_4: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_2, [1, 12, 512, 512]);  where_2 = None
        view_24: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_4, [12, 512, 512]);  expand_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_17: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768])
        permute_11: "f16[768, 768]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_6: "f16[512, 768]" = torch.ops.aten.addmm.default(arg21_1, view_17, permute_11);  arg21_1 = view_17 = permute_11 = None
        view_18: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [1, 512, 768]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_20: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_18, [1, 512, -1, 64]);  view_18 = None
        permute_13: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_27: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_13, torch.float32);  permute_13 = None
        expand_5: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_27, [1, 12, 512, 64]);  convert_element_type_27 = None
        view_25: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_5, [12, 512, 64]);  expand_5 = None
        bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_24, view_25);  view_24 = view_25 = None
        view_26: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
        convert_element_type_29: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_26, torch.float16);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_29, [0, 2, 1, 3]);  convert_element_type_29 = None
        clone_2: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_27: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_2, [1, 512, -1]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_28: "f16[512, 768]" = torch.ops.aten.reshape.default(view_27, [512, 768]);  view_27 = None
        permute_16: "f16[768, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_7: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_28, permute_16);  arg23_1 = view_28 = permute_16 = None
        view_29: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:378 in forward, code: hidden_states = residual + hidden_states
        add_13: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_15, view_29);  convert_element_type_15 = view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:379 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_33: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_33, [2], correction = 0, keepdim = True)
        getitem_13: "f32[1, 512, 1]" = var_mean_2[0]
        getitem_14: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_14);  convert_element_type_33 = getitem_14 = None
        add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_7: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_7, arg24_1);  mul_7 = arg24_1 = None
        add_15: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_8, arg25_1);  mul_8 = arg25_1 = None
        convert_element_type_34: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_15, torch.float16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:383 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_30: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_34, [512, 768])
        permute_17: "f16[768, 3072]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg27_1, view_30, permute_17);  arg27_1 = view_30 = permute_17 = None
        view_31: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_8, [1, 512, 3072]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_38: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_31, torch.float32);  view_31 = None
        mul_9: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.5)
        mul_10: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.7071067811865476);  convert_element_type_38 = None
        erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
        add_16: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_11: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_9, add_16);  mul_9 = add_16 = None
        convert_element_type_39: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_11, torch.float16);  mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:385 in forward, code: hidden_states = self.fc2(hidden_states)
        view_32: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_39, [512, 3072]);  convert_element_type_39 = None
        permute_18: "f16[3072, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9: "f16[512, 768]" = torch.ops.aten.addmm.default(arg29_1, view_32, permute_18);  arg29_1 = view_32 = permute_18 = None
        view_33: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [1, 512, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:387 in forward, code: hidden_states = residual + hidden_states
        add_17: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_34, view_33);  convert_element_type_34 = view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_43: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32);  add_17 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_43, [2], correction = 0, keepdim = True)
        getitem_15: "f32[1, 512, 1]" = var_mean_3[0]
        getitem_16: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_4: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_43, getitem_16);  convert_element_type_43 = getitem_16 = None
        add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_15, 1e-05);  getitem_15 = None
        rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_13: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, arg30_1);  mul_12 = arg30_1 = None
        add_19: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_13, arg31_1);  mul_13 = arg31_1 = None
        convert_element_type_44: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.float16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_34: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_44, [512, 768])
        permute_19: "f16[768, 768]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10: "f16[512, 768]" = torch.ops.aten.addmm.default(arg33_1, view_34, permute_19);  arg33_1 = view_34 = permute_19 = None
        view_35: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 768]);  addmm_10 = None
        view_36: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_35, [1, 512, -1, 64]);  view_35 = None
        permute_20: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_37: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_44, [512, 768])
        permute_21: "f16[768, 768]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11: "f16[512, 768]" = torch.ops.aten.addmm.default(arg35_1, view_37, permute_21);  arg35_1 = view_37 = permute_21 = None
        view_38: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_41: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_38, [1, 512, -1, 64]);  view_38 = None
        permute_23: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_39: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_44, [512, 768])
        permute_22: "f16[768, 768]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_12: "f16[512, 768]" = torch.ops.aten.addmm.default(arg37_1, view_39, permute_22);  arg37_1 = view_39 = permute_22 = None
        view_40: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [1, 512, 768]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_42: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_40, [1, 512, -1, 64]);  view_40 = None
        permute_24: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_6: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_20, permute_23, permute_24, where_3, False, scale = 0.125);  permute_20 = permute_23 = permute_24 = where_3 = None
        getitem_17: "f16[1, 12, 512, 64]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_25: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_17, [0, 2, 1, 3]);  getitem_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_43: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(permute_25, [1, 512, -1]);  permute_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_44: "f16[512, 768]" = torch.ops.aten.reshape.default(view_43, [512, 768]);  view_43 = None
        permute_26: "f16[768, 768]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_13: "f16[512, 768]" = torch.ops.aten.addmm.default(arg39_1, view_44, permute_26);  arg39_1 = view_44 = permute_26 = None
        view_45: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:363 in forward, code: hidden_states = residual + hidden_states
        add_20: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_44, view_45);  convert_element_type_44 = view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:364 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_57: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32);  add_20 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_57, [2], correction = 0, keepdim = True)
        getitem_26: "f32[1, 512, 1]" = var_mean_4[0]
        getitem_27: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_57, getitem_27);  convert_element_type_57 = getitem_27 = None
        add_21: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_14: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_15: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg40_1);  mul_14 = arg40_1 = None
        add_22: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_15, arg41_1);  mul_15 = arg41_1 = None
        convert_element_type_58: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_22, torch.float16);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_46: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_58, [512, 768])
        permute_27: "f16[768, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14: "f16[512, 768]" = torch.ops.aten.addmm.default(arg43_1, view_46, permute_27);  arg43_1 = view_46 = permute_27 = None
        view_47: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [1, 512, 768]);  addmm_14 = None
        view_48: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_47, [1, 512, -1, 64]);  view_47 = None
        permute_28: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_68: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_28, torch.float32);  permute_28 = None
        mul_16: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_68, 0.3535533905932738);  convert_element_type_68 = None
        expand_6: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_16, [1, 12, 512, 64]);  mul_16 = None
        view_55: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_6, [12, 512, 64]);  expand_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_49: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768])
        permute_29: "f16[768, 768]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15: "f16[512, 768]" = torch.ops.aten.addmm.default(arg45_1, view_49, permute_29);  arg45_1 = view_49 = permute_29 = None
        view_50: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [1, 512, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_53: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_50, [1, 512, -1, 64]);  view_50 = None
        permute_31: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_69: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_31, torch.float32);  permute_31 = None
        permute_33: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_69, [0, 1, 3, 2]);  convert_element_type_69 = None
        mul_17: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_33, 0.3535533905932738);  permute_33 = None
        expand_7: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_17, [1, 12, 64, 512]);  mul_17 = None
        view_56: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_7, [12, 64, 512]);  expand_7 = None
        bmm_2: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
        view_57: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 512, 512]);  bmm_2 = None
        full_default_8: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        add_23: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_57, where_4);  view_57 = where_4 = None
        eq_1: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_23, -inf)
        logical_not_2: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_default_9: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_23, [-1], True)
        sub_6: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_23, amax_1);  add_23 = amax_1 = None
        exp_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        where_5: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full_default_9, div_1);  logical_not_3 = full_default_9 = div_1 = None
        expand_8: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_5, [1, 12, 512, 512]);  where_5 = None
        view_58: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_8, [12, 512, 512]);  expand_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_51: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768])
        permute_30: "f16[768, 768]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_16: "f16[512, 768]" = torch.ops.aten.addmm.default(arg47_1, view_51, permute_30);  arg47_1 = view_51 = permute_30 = None
        view_52: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 768]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_54: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_52, [1, 512, -1, 64]);  view_52 = None
        permute_32: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_70: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_32, torch.float32);  permute_32 = None
        expand_9: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_70, [1, 12, 512, 64]);  convert_element_type_70 = None
        view_59: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_9, [12, 512, 64]);  expand_9 = None
        bmm_3: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
        view_60: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 512, 64]);  bmm_3 = None
        convert_element_type_72: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_60, torch.float16);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_34: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_72, [0, 2, 1, 3]);  convert_element_type_72 = None
        clone_7: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_61: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_7, [1, 512, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_62: "f16[512, 768]" = torch.ops.aten.reshape.default(view_61, [512, 768]);  view_61 = None
        permute_35: "f16[768, 768]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_17: "f16[512, 768]" = torch.ops.aten.addmm.default(arg49_1, view_62, permute_35);  arg49_1 = view_62 = permute_35 = None
        view_63: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:378 in forward, code: hidden_states = residual + hidden_states
        add_24: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_58, view_63);  convert_element_type_58 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:379 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_76: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32);  add_24 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_76, [2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 512, 1]" = var_mean_5[0]
        getitem_29: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_7: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_76, getitem_29);  convert_element_type_76 = getitem_29 = None
        add_25: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg50_1);  mul_18 = arg50_1 = None
        add_26: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, arg51_1);  mul_19 = arg51_1 = None
        convert_element_type_77: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_26, torch.float16);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:383 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_64: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_77, [512, 768])
        permute_36: "f16[768, 3072]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_18: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg53_1, view_64, permute_36);  arg53_1 = view_64 = permute_36 = None
        view_65: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_18, [1, 512, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_81: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_65, torch.float32);  view_65 = None
        mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 0.5)
        mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 0.7071067811865476);  convert_element_type_81 = None
        erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_21);  mul_21 = None
        add_27: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_22: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_27);  mul_20 = add_27 = None
        convert_element_type_82: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_22, torch.float16);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:385 in forward, code: hidden_states = self.fc2(hidden_states)
        view_66: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_82, [512, 3072]);  convert_element_type_82 = None
        permute_37: "f16[3072, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_19: "f16[512, 768]" = torch.ops.aten.addmm.default(arg55_1, view_66, permute_37);  arg55_1 = view_66 = permute_37 = None
        view_67: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:387 in forward, code: hidden_states = residual + hidden_states
        add_28: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_77, view_67);  convert_element_type_77 = view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_86: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_28, torch.float32);  add_28 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_86, [2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 512, 1]" = var_mean_6[0]
        getitem_31: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_86, getitem_31);  convert_element_type_86 = getitem_31 = None
        add_29: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_23: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_23, arg56_1);  mul_23 = arg56_1 = None
        add_30: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_24, arg57_1);  mul_24 = arg57_1 = None
        convert_element_type_87: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_30, torch.float16);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_68: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_87, [512, 768])
        permute_38: "f16[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20: "f16[512, 768]" = torch.ops.aten.addmm.default(arg59_1, view_68, permute_38);  arg59_1 = view_68 = permute_38 = None
        view_69: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [1, 512, 768]);  addmm_20 = None
        view_70: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_69, [1, 512, -1, 64]);  view_69 = None
        permute_39: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_71: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_87, [512, 768])
        permute_40: "f16[768, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21: "f16[512, 768]" = torch.ops.aten.addmm.default(arg61_1, view_71, permute_40);  arg61_1 = view_71 = permute_40 = None
        view_72: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_75: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_72, [1, 512, -1, 64]);  view_72 = None
        permute_42: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_73: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_87, [512, 768])
        permute_41: "f16[768, 768]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_22: "f16[512, 768]" = torch.ops.aten.addmm.default(arg63_1, view_73, permute_41);  arg63_1 = view_73 = permute_41 = None
        view_74: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 768]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_76: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_74, [1, 512, -1, 64]);  view_74 = None
        permute_43: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_11: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_11, full_default_10);  full_default_11 = full_default_10 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_39, permute_42, permute_43, where_6, False, scale = 0.125);  permute_39 = permute_42 = permute_43 = where_6 = None
        getitem_32: "f16[1, 12, 512, 64]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_44: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_32, [0, 2, 1, 3]);  getitem_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_77: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(permute_44, [1, 512, -1]);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_78: "f16[512, 768]" = torch.ops.aten.reshape.default(view_77, [512, 768]);  view_77 = None
        permute_45: "f16[768, 768]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_23: "f16[512, 768]" = torch.ops.aten.addmm.default(arg65_1, view_78, permute_45);  arg65_1 = view_78 = permute_45 = None
        view_79: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:363 in forward, code: hidden_states = residual + hidden_states
        add_31: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_87, view_79);  convert_element_type_87 = view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:364 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_100: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32);  add_31 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_100, [2], correction = 0, keepdim = True)
        getitem_41: "f32[1, 512, 1]" = var_mean_7[0]
        getitem_42: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_100, getitem_42);  convert_element_type_100 = getitem_42 = None
        add_32: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
        rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = rsqrt_7 = None
        mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg66_1);  mul_25 = arg66_1 = None
        add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, arg67_1);  mul_26 = arg67_1 = None
        convert_element_type_101: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_33, torch.float16);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_80: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_101, [512, 768])
        permute_46: "f16[768, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_24: "f16[512, 768]" = torch.ops.aten.addmm.default(arg69_1, view_80, permute_46);  arg69_1 = view_80 = permute_46 = None
        view_81: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [1, 512, 768]);  addmm_24 = None
        view_82: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_81, [1, 512, -1, 64]);  view_81 = None
        permute_47: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_111: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_47, torch.float32);  permute_47 = None
        mul_27: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_111, 0.3535533905932738);  convert_element_type_111 = None
        expand_10: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_27, [1, 12, 512, 64]);  mul_27 = None
        view_89: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_10, [12, 512, 64]);  expand_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_83: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768])
        permute_48: "f16[768, 768]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_25: "f16[512, 768]" = torch.ops.aten.addmm.default(arg71_1, view_83, permute_48);  arg71_1 = view_83 = permute_48 = None
        view_84: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [1, 512, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_87: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_84, [1, 512, -1, 64]);  view_84 = None
        permute_50: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_112: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_50, torch.float32);  permute_50 = None
        permute_52: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_112, [0, 1, 3, 2]);  convert_element_type_112 = None
        mul_28: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_52, 0.3535533905932738);  permute_52 = None
        expand_11: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_28, [1, 12, 64, 512]);  mul_28 = None
        view_90: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_11, [12, 64, 512]);  expand_11 = None
        bmm_4: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_89, view_90);  view_89 = view_90 = None
        view_91: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 512, 512]);  bmm_4 = None
        full_default_13: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        add_34: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_91, where_7);  view_91 = where_7 = None
        eq_2: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_34, -inf)
        logical_not_4: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_default_14: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_34, [-1], True)
        sub_10: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_34, amax_2);  add_34 = amax_2 = None
        exp_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        where_8: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full_default_14, div_2);  logical_not_5 = full_default_14 = div_2 = None
        expand_12: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_8, [1, 12, 512, 512]);  where_8 = None
        view_92: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_12, [12, 512, 512]);  expand_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_85: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768])
        permute_49: "f16[768, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_26: "f16[512, 768]" = torch.ops.aten.addmm.default(arg73_1, view_85, permute_49);  arg73_1 = view_85 = permute_49 = None
        view_86: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [1, 512, 768]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_88: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_86, [1, 512, -1, 64]);  view_86 = None
        permute_51: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_113: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_51, torch.float32);  permute_51 = None
        expand_13: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_113, [1, 12, 512, 64]);  convert_element_type_113 = None
        view_93: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_13, [12, 512, 64]);  expand_13 = None
        bmm_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_92, view_93);  view_92 = view_93 = None
        view_94: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 512, 64]);  bmm_5 = None
        convert_element_type_115: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_94, torch.float16);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_53: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_115, [0, 2, 1, 3]);  convert_element_type_115 = None
        clone_12: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_95: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_12, [1, 512, -1]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_96: "f16[512, 768]" = torch.ops.aten.reshape.default(view_95, [512, 768]);  view_95 = None
        permute_54: "f16[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_27: "f16[512, 768]" = torch.ops.aten.addmm.default(arg75_1, view_96, permute_54);  arg75_1 = view_96 = permute_54 = None
        view_97: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [1, 512, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:378 in forward, code: hidden_states = residual + hidden_states
        add_35: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_101, view_97);  convert_element_type_101 = view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:379 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_119: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_119, [2], correction = 0, keepdim = True)
        getitem_43: "f32[1, 512, 1]" = var_mean_8[0]
        getitem_44: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_119, getitem_44);  convert_element_type_119 = getitem_44 = None
        add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_43, 1e-05);  getitem_43 = None
        rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_29: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = rsqrt_8 = None
        mul_30: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, arg76_1);  mul_29 = arg76_1 = None
        add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_30, arg77_1);  mul_30 = arg77_1 = None
        convert_element_type_120: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:383 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_98: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_120, [512, 768])
        permute_55: "f16[768, 3072]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_28: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg79_1, view_98, permute_55);  arg79_1 = view_98 = permute_55 = None
        view_99: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_124: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_124, 0.5)
        mul_32: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_124, 0.7071067811865476);  convert_element_type_124 = None
        erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_32);  mul_32 = None
        add_38: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_33: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_31, add_38);  mul_31 = add_38 = None
        convert_element_type_125: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_33, torch.float16);  mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:385 in forward, code: hidden_states = self.fc2(hidden_states)
        view_100: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_125, [512, 3072]);  convert_element_type_125 = None
        permute_56: "f16[3072, 768]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_29: "f16[512, 768]" = torch.ops.aten.addmm.default(arg81_1, view_100, permute_56);  arg81_1 = view_100 = permute_56 = None
        view_101: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_29, [1, 512, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:387 in forward, code: hidden_states = residual + hidden_states
        add_39: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_120, view_101);  convert_element_type_120 = view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_129: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_39, torch.float32);  add_39 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_129, [2], correction = 0, keepdim = True)
        getitem_45: "f32[1, 512, 1]" = var_mean_9[0]
        getitem_46: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_129, getitem_46);  convert_element_type_129 = getitem_46 = None
        add_40: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_45, 1e-05);  getitem_45 = None
        rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_9);  sub_12 = rsqrt_9 = None
        mul_35: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg82_1);  mul_34 = arg82_1 = None
        add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, arg83_1);  mul_35 = arg83_1 = None
        convert_element_type_130: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_41, torch.float16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_102: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_130, [512, 768])
        permute_57: "f16[768, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_30: "f16[512, 768]" = torch.ops.aten.addmm.default(arg85_1, view_102, permute_57);  arg85_1 = view_102 = permute_57 = None
        view_103: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [1, 512, 768]);  addmm_30 = None
        view_104: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_103, [1, 512, -1, 64]);  view_103 = None
        permute_58: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_105: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_130, [512, 768])
        permute_59: "f16[768, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_31: "f16[512, 768]" = torch.ops.aten.addmm.default(arg87_1, view_105, permute_59);  arg87_1 = view_105 = permute_59 = None
        view_106: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_31, [1, 512, 768]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_109: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_106, [1, 512, -1, 64]);  view_106 = None
        permute_61: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_107: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_130, [512, 768])
        permute_60: "f16[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_32: "f16[512, 768]" = torch.ops.aten.addmm.default(arg89_1, view_107, permute_60);  arg89_1 = view_107 = permute_60 = None
        view_108: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [1, 512, 768]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_110: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_108, [1, 512, -1, 64]);  view_108 = None
        permute_62: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_16: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_58, permute_61, permute_62, where_9, False, scale = 0.125);  permute_58 = permute_61 = permute_62 = where_9 = None
        getitem_47: "f16[1, 12, 512, 64]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_47, [0, 2, 1, 3]);  getitem_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_111: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(permute_63, [1, 512, -1]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_112: "f16[512, 768]" = torch.ops.aten.reshape.default(view_111, [512, 768]);  view_111 = None
        permute_64: "f16[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_33: "f16[512, 768]" = torch.ops.aten.addmm.default(arg91_1, view_112, permute_64);  arg91_1 = view_112 = permute_64 = None
        view_113: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_33, [1, 512, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:363 in forward, code: hidden_states = residual + hidden_states
        add_42: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_130, view_113);  convert_element_type_130 = view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:364 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_143: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_42, torch.float32);  add_42 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_143, [2], correction = 0, keepdim = True)
        getitem_56: "f32[1, 512, 1]" = var_mean_10[0]
        getitem_57: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_143, getitem_57);  convert_element_type_143 = getitem_57 = None
        add_43: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_36: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_10);  sub_13 = rsqrt_10 = None
        mul_37: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg92_1);  mul_36 = arg92_1 = None
        add_44: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, arg93_1);  mul_37 = arg93_1 = None
        convert_element_type_144: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_44, torch.float16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_114: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_144, [512, 768])
        permute_65: "f16[768, 768]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_34: "f16[512, 768]" = torch.ops.aten.addmm.default(arg95_1, view_114, permute_65);  arg95_1 = view_114 = permute_65 = None
        view_115: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 768]);  addmm_34 = None
        view_116: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_115, [1, 512, -1, 64]);  view_115 = None
        permute_66: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_154: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_66, torch.float32);  permute_66 = None
        mul_38: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_154, 0.3535533905932738);  convert_element_type_154 = None
        expand_14: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_38, [1, 12, 512, 64]);  mul_38 = None
        view_123: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_14, [12, 512, 64]);  expand_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_117: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768])
        permute_67: "f16[768, 768]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_35: "f16[512, 768]" = torch.ops.aten.addmm.default(arg97_1, view_117, permute_67);  arg97_1 = view_117 = permute_67 = None
        view_118: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_35, [1, 512, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_121: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_118, [1, 512, -1, 64]);  view_118 = None
        permute_69: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_155: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_69, torch.float32);  permute_69 = None
        permute_71: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_155, [0, 1, 3, 2]);  convert_element_type_155 = None
        mul_39: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_71, 0.3535533905932738);  permute_71 = None
        expand_15: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_39, [1, 12, 64, 512]);  mul_39 = None
        view_124: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_15, [12, 64, 512]);  expand_15 = None
        bmm_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_123, view_124);  view_123 = view_124 = None
        view_125: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 512, 512]);  bmm_6 = None
        full_default_18: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_18, full_default_17);  full_default_18 = full_default_17 = None
        add_45: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_125, where_10);  view_125 = where_10 = None
        eq_3: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_45, -inf)
        logical_not_6: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_default_19: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_45, [-1], True)
        sub_14: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_45, amax_3);  add_45 = amax_3 = None
        exp_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        where_11: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full_default_19, div_3);  logical_not_7 = full_default_19 = div_3 = None
        expand_16: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_11, [1, 12, 512, 512]);  where_11 = None
        view_126: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_16, [12, 512, 512]);  expand_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_119: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768])
        permute_68: "f16[768, 768]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_36: "f16[512, 768]" = torch.ops.aten.addmm.default(arg99_1, view_119, permute_68);  arg99_1 = view_119 = permute_68 = None
        view_120: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [1, 512, 768]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_122: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_120, [1, 512, -1, 64]);  view_120 = None
        permute_70: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_156: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_70, torch.float32);  permute_70 = None
        expand_17: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_156, [1, 12, 512, 64]);  convert_element_type_156 = None
        view_127: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_17, [12, 512, 64]);  expand_17 = None
        bmm_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_126, view_127);  view_126 = view_127 = None
        view_128: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 512, 64]);  bmm_7 = None
        convert_element_type_158: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_128, torch.float16);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_72: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_158, [0, 2, 1, 3]);  convert_element_type_158 = None
        clone_17: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_129: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_17, [1, 512, -1]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_130: "f16[512, 768]" = torch.ops.aten.reshape.default(view_129, [512, 768]);  view_129 = None
        permute_73: "f16[768, 768]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_37: "f16[512, 768]" = torch.ops.aten.addmm.default(arg101_1, view_130, permute_73);  arg101_1 = view_130 = permute_73 = None
        view_131: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_37, [1, 512, 768]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:378 in forward, code: hidden_states = residual + hidden_states
        add_46: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_144, view_131);  convert_element_type_144 = view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:379 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_162: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_46, torch.float32);  add_46 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_162, [2], correction = 0, keepdim = True)
        getitem_58: "f32[1, 512, 1]" = var_mean_11[0]
        getitem_59: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_162, getitem_59);  convert_element_type_162 = getitem_59 = None
        add_47: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = rsqrt_11 = None
        mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg102_1);  mul_40 = arg102_1 = None
        add_48: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, arg103_1);  mul_41 = arg103_1 = None
        convert_element_type_163: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_48, torch.float16);  add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:383 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_132: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_163, [512, 768])
        permute_74: "f16[768, 3072]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_38: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg105_1, view_132, permute_74);  arg105_1 = view_132 = permute_74 = None
        view_133: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_38, [1, 512, 3072]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_167: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        mul_42: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_167, 0.5)
        mul_43: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_167, 0.7071067811865476);  convert_element_type_167 = None
        erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_49: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_42, add_49);  mul_42 = add_49 = None
        convert_element_type_168: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_44, torch.float16);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:385 in forward, code: hidden_states = self.fc2(hidden_states)
        view_134: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_168, [512, 3072]);  convert_element_type_168 = None
        permute_75: "f16[3072, 768]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_39: "f16[512, 768]" = torch.ops.aten.addmm.default(arg107_1, view_134, permute_75);  arg107_1 = view_134 = permute_75 = None
        view_135: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_39, [1, 512, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:387 in forward, code: hidden_states = residual + hidden_states
        add_50: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_163, view_135);  convert_element_type_163 = view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_172: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_50, torch.float32);  add_50 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_172, [2], correction = 0, keepdim = True)
        getitem_60: "f32[1, 512, 1]" = var_mean_12[0]
        getitem_61: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_172, getitem_61);  convert_element_type_172 = getitem_61 = None
        add_51: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_45: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_12);  sub_16 = rsqrt_12 = None
        mul_46: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg108_1);  mul_45 = arg108_1 = None
        add_52: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, arg109_1);  mul_46 = arg109_1 = None
        convert_element_type_173: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_52, torch.float16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_136: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_173, [512, 768])
        permute_76: "f16[768, 768]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_40: "f16[512, 768]" = torch.ops.aten.addmm.default(arg111_1, view_136, permute_76);  arg111_1 = view_136 = permute_76 = None
        view_137: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_40, [1, 512, 768]);  addmm_40 = None
        view_138: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_137, [1, 512, -1, 64]);  view_137 = None
        permute_77: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_139: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_173, [512, 768])
        permute_78: "f16[768, 768]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_41: "f16[512, 768]" = torch.ops.aten.addmm.default(arg113_1, view_139, permute_78);  arg113_1 = view_139 = permute_78 = None
        view_140: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_41, [1, 512, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_143: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_140, [1, 512, -1, 64]);  view_140 = None
        permute_80: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_141: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_173, [512, 768])
        permute_79: "f16[768, 768]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_42: "f16[512, 768]" = torch.ops.aten.addmm.default(arg115_1, view_141, permute_79);  arg115_1 = view_141 = permute_79 = None
        view_142: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_42, [1, 512, 768]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_144: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_142, [1, 512, -1, 64]);  view_142 = None
        permute_81: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_21: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_21, full_default_20);  full_default_21 = full_default_20 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_77, permute_80, permute_81, where_12, False, scale = 0.125);  permute_77 = permute_80 = permute_81 = where_12 = None
        getitem_62: "f16[1, 12, 512, 64]" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_82: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3]);  getitem_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_145: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(permute_82, [1, 512, -1]);  permute_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_146: "f16[512, 768]" = torch.ops.aten.reshape.default(view_145, [512, 768]);  view_145 = None
        permute_83: "f16[768, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_43: "f16[512, 768]" = torch.ops.aten.addmm.default(arg117_1, view_146, permute_83);  arg117_1 = view_146 = permute_83 = None
        view_147: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_43, [1, 512, 768]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:363 in forward, code: hidden_states = residual + hidden_states
        add_53: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_173, view_147);  convert_element_type_173 = view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:364 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_186: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_53, torch.float32);  add_53 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_186, [2], correction = 0, keepdim = True)
        getitem_71: "f32[1, 512, 1]" = var_mean_13[0]
        getitem_72: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_186, getitem_72);  convert_element_type_186 = getitem_72 = None
        add_54: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_71, 1e-05);  getitem_71 = None
        rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_47: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_13);  sub_17 = rsqrt_13 = None
        mul_48: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_47, arg118_1);  mul_47 = arg118_1 = None
        add_55: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_48, arg119_1);  mul_48 = arg119_1 = None
        convert_element_type_187: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_55, torch.float16);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_148: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_187, [512, 768])
        permute_84: "f16[768, 768]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_44: "f16[512, 768]" = torch.ops.aten.addmm.default(arg121_1, view_148, permute_84);  arg121_1 = view_148 = permute_84 = None
        view_149: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_44, [1, 512, 768]);  addmm_44 = None
        view_150: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_149, [1, 512, -1, 64]);  view_149 = None
        permute_85: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_197: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_85, torch.float32);  permute_85 = None
        mul_49: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_197, 0.3535533905932738);  convert_element_type_197 = None
        expand_18: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_49, [1, 12, 512, 64]);  mul_49 = None
        view_157: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_18, [12, 512, 64]);  expand_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_151: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768])
        permute_86: "f16[768, 768]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_45: "f16[512, 768]" = torch.ops.aten.addmm.default(arg123_1, view_151, permute_86);  arg123_1 = view_151 = permute_86 = None
        view_152: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_45, [1, 512, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_155: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_152, [1, 512, -1, 64]);  view_152 = None
        permute_88: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_198: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_88, torch.float32);  permute_88 = None
        permute_90: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_198, [0, 1, 3, 2]);  convert_element_type_198 = None
        mul_50: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_90, 0.3535533905932738);  permute_90 = None
        expand_19: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_50, [1, 12, 64, 512]);  mul_50 = None
        view_158: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_19, [12, 64, 512]);  expand_19 = None
        bmm_8: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_157, view_158);  view_157 = view_158 = None
        view_159: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 512, 512]);  bmm_8 = None
        full_default_23: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_23, full_default_22);  full_default_23 = full_default_22 = None
        add_56: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_159, where_13);  view_159 = where_13 = None
        eq_4: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_56, -inf)
        logical_not_8: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_default_24: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_56, [-1], True)
        sub_18: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_56, amax_4);  add_56 = amax_4 = None
        exp_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        where_14: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full_default_24, div_4);  logical_not_9 = full_default_24 = div_4 = None
        expand_20: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_14, [1, 12, 512, 512]);  where_14 = None
        view_160: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_20, [12, 512, 512]);  expand_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_153: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768])
        permute_87: "f16[768, 768]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_46: "f16[512, 768]" = torch.ops.aten.addmm.default(arg125_1, view_153, permute_87);  arg125_1 = view_153 = permute_87 = None
        view_154: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_46, [1, 512, 768]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_156: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_154, [1, 512, -1, 64]);  view_154 = None
        permute_89: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_199: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_89, torch.float32);  permute_89 = None
        expand_21: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_199, [1, 12, 512, 64]);  convert_element_type_199 = None
        view_161: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_21, [12, 512, 64]);  expand_21 = None
        bmm_9: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_160, view_161);  view_160 = view_161 = None
        view_162: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 512, 64]);  bmm_9 = None
        convert_element_type_201: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_162, torch.float16);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_91: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_201, [0, 2, 1, 3]);  convert_element_type_201 = None
        clone_22: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_163: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_22, [1, 512, -1]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_164: "f16[512, 768]" = torch.ops.aten.reshape.default(view_163, [512, 768]);  view_163 = None
        permute_92: "f16[768, 768]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_47: "f16[512, 768]" = torch.ops.aten.addmm.default(arg127_1, view_164, permute_92);  arg127_1 = view_164 = permute_92 = None
        view_165: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_47, [1, 512, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:378 in forward, code: hidden_states = residual + hidden_states
        add_57: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_187, view_165);  convert_element_type_187 = view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:379 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_205: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_205, [2], correction = 0, keepdim = True)
        getitem_73: "f32[1, 512, 1]" = var_mean_14[0]
        getitem_74: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_19: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_205, getitem_74);  convert_element_type_205 = getitem_74 = None
        add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_73, 1e-05);  getitem_73 = None
        rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_14);  sub_19 = rsqrt_14 = None
        mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_51, arg128_1);  mul_51 = arg128_1 = None
        add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_52, arg129_1);  mul_52 = arg129_1 = None
        convert_element_type_206: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.float16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:383 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_166: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_206, [512, 768])
        permute_93: "f16[768, 3072]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_48: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg131_1, view_166, permute_93);  arg131_1 = view_166 = permute_93 = None
        view_167: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_48, [1, 512, 3072]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_210: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_167, torch.float32);  view_167 = None
        mul_53: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_210, 0.5)
        mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_210, 0.7071067811865476);  convert_element_type_210 = None
        erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_60: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_60);  mul_53 = add_60 = None
        convert_element_type_211: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_55, torch.float16);  mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:385 in forward, code: hidden_states = self.fc2(hidden_states)
        view_168: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_211, [512, 3072]);  convert_element_type_211 = None
        permute_94: "f16[3072, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_49: "f16[512, 768]" = torch.ops.aten.addmm.default(arg133_1, view_168, permute_94);  arg133_1 = view_168 = permute_94 = None
        view_169: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_49, [1, 512, 768]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:387 in forward, code: hidden_states = residual + hidden_states
        add_61: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_206, view_169);  convert_element_type_206 = view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_215: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_61, torch.float32);  add_61 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_215, [2], correction = 0, keepdim = True)
        getitem_75: "f32[1, 512, 1]" = var_mean_15[0]
        getitem_76: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_215, getitem_76);  convert_element_type_215 = getitem_76 = None
        add_62: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_56: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_15);  sub_20 = rsqrt_15 = None
        mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg134_1);  mul_56 = arg134_1 = None
        add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_57, arg135_1);  mul_57 = arg135_1 = None
        convert_element_type_216: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_63, torch.float16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_170: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_216, [512, 768])
        permute_95: "f16[768, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_50: "f16[512, 768]" = torch.ops.aten.addmm.default(arg137_1, view_170, permute_95);  arg137_1 = view_170 = permute_95 = None
        view_171: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_50, [1, 512, 768]);  addmm_50 = None
        view_172: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_171, [1, 512, -1, 64]);  view_171 = None
        permute_96: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_173: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_216, [512, 768])
        permute_97: "f16[768, 768]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_51: "f16[512, 768]" = torch.ops.aten.addmm.default(arg139_1, view_173, permute_97);  arg139_1 = view_173 = permute_97 = None
        view_174: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_51, [1, 512, 768]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_177: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_174, [1, 512, -1, 64]);  view_174 = None
        permute_99: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_175: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_216, [512, 768])
        permute_98: "f16[768, 768]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_52: "f16[512, 768]" = torch.ops.aten.addmm.default(arg141_1, view_175, permute_98);  arg141_1 = view_175 = permute_98 = None
        view_176: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_52, [1, 512, 768]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_178: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_176, [1, 512, -1, 64]);  view_176 = None
        permute_100: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_26: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_26, full_default_25);  expand = full_default_26 = full_default_25 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_96, permute_99, permute_100, where_15, False, scale = 0.125);  permute_96 = permute_99 = permute_100 = where_15 = None
        getitem_77: "f16[1, 12, 512, 64]" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_101: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_77, [0, 2, 1, 3]);  getitem_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_179: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(permute_101, [1, 512, -1]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_180: "f16[512, 768]" = torch.ops.aten.reshape.default(view_179, [512, 768]);  view_179 = None
        permute_102: "f16[768, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_53: "f16[512, 768]" = torch.ops.aten.addmm.default(arg143_1, view_180, permute_102);  arg143_1 = view_180 = permute_102 = None
        view_181: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_53, [1, 512, 768]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:363 in forward, code: hidden_states = residual + hidden_states
        add_64: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_216, view_181);  convert_element_type_216 = view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:364 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_229: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_64, torch.float32);  add_64 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_229, [2], correction = 0, keepdim = True)
        getitem_86: "f32[1, 512, 1]" = var_mean_16[0]
        getitem_87: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_229, getitem_87);  convert_element_type_229 = getitem_87 = None
        add_65: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_16);  sub_21 = rsqrt_16 = None
        mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg144_1);  mul_58 = arg144_1 = None
        add_66: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, arg145_1);  mul_59 = arg145_1 = None
        convert_element_type_230: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_66, torch.float16);  add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_182: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_230, [512, 768])
        permute_103: "f16[768, 768]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_54: "f16[512, 768]" = torch.ops.aten.addmm.default(arg147_1, view_182, permute_103);  arg147_1 = view_182 = permute_103 = None
        view_183: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_54, [1, 512, 768]);  addmm_54 = None
        view_184: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_183, [1, 512, -1, 64]);  view_183 = None
        permute_104: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_240: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_104, torch.float32);  permute_104 = None
        mul_60: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_240, 0.3535533905932738);  convert_element_type_240 = None
        expand_22: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_60, [1, 12, 512, 64]);  mul_60 = None
        view_191: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_22, [12, 512, 64]);  expand_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        view_185: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768])
        permute_105: "f16[768, 768]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_55: "f16[512, 768]" = torch.ops.aten.addmm.default(arg149_1, view_185, permute_105);  arg149_1 = view_185 = permute_105 = None
        view_186: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_55, [1, 512, 768]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_189: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_186, [1, 512, -1, 64]);  view_186 = None
        permute_107: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_241: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_107, torch.float32);  permute_107 = None
        permute_109: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_241, [0, 1, 3, 2]);  convert_element_type_241 = None
        mul_61: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_109, 0.3535533905932738);  permute_109 = None
        expand_23: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_61, [1, 12, 64, 512]);  mul_61 = None
        view_192: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_23, [12, 64, 512]);  expand_23 = None
        bmm_10: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_191, view_192);  view_191 = view_192 = None
        view_193: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 512, 512]);  bmm_10 = None
        full_default_28: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_28, full_default_27);  expand_1 = full_default_28 = full_default_27 = None
        add_67: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_193, where_16);  view_193 = where_16 = None
        eq_5: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_67, -inf)
        logical_not_10: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_default_29: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_67, [-1], True)
        sub_22: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_67, amax_5);  add_67 = amax_5 = None
        exp_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        where_17: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full_default_29, div_5);  logical_not_11 = full_default_29 = div_5 = None
        expand_24: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_17, [1, 12, 512, 512]);  where_17 = None
        view_194: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_24, [12, 512, 512]);  expand_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_187: "f16[512, 768]" = torch.ops.aten.reshape.default(arg2_1, [512, 768]);  arg2_1 = None
        permute_106: "f16[768, 768]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_56: "f16[512, 768]" = torch.ops.aten.addmm.default(arg151_1, view_187, permute_106);  arg151_1 = view_187 = permute_106 = None
        view_188: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_56, [1, 512, 768]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_190: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_188, [1, 512, -1, 64]);  view_188 = None
        permute_108: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_242: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_108, torch.float32);  permute_108 = None
        expand_25: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_242, [1, 12, 512, 64]);  convert_element_type_242 = None
        view_195: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_25, [12, 512, 64]);  expand_25 = None
        bmm_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_194, view_195);  view_194 = view_195 = None
        view_196: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 512, 64]);  bmm_11 = None
        convert_element_type_244: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_196, torch.float16);  view_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_110: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_244, [0, 2, 1, 3]);  convert_element_type_244 = None
        clone_27: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_197: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_27, [1, 512, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_198: "f16[512, 768]" = torch.ops.aten.reshape.default(view_197, [512, 768]);  view_197 = None
        permute_111: "f16[768, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_57: "f16[512, 768]" = torch.ops.aten.addmm.default(arg153_1, view_198, permute_111);  arg153_1 = view_198 = permute_111 = None
        view_199: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_57, [1, 512, 768]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:378 in forward, code: hidden_states = residual + hidden_states
        add_68: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_230, view_199);  convert_element_type_230 = view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:379 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_248: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_68, torch.float32);  add_68 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_248, [2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 512, 1]" = var_mean_17[0]
        getitem_89: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_248, getitem_89);  convert_element_type_248 = getitem_89 = None
        add_69: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_62: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_17);  sub_23 = rsqrt_17 = None
        mul_63: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_62, arg154_1);  mul_62 = arg154_1 = None
        add_70: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_63, arg155_1);  mul_63 = arg155_1 = None
        convert_element_type_249: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_70, torch.float16);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:383 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_200: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_249, [512, 768])
        permute_112: "f16[768, 3072]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_58: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg157_1, view_200, permute_112);  arg157_1 = view_200 = permute_112 = None
        view_201: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [1, 512, 3072]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_253: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_201, torch.float32);  view_201 = None
        mul_64: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_253, 0.5)
        mul_65: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_253, 0.7071067811865476);  convert_element_type_253 = None
        erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_65);  mul_65 = None
        add_71: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_66: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_64, add_71);  mul_64 = add_71 = None
        convert_element_type_254: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_66, torch.float16);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:385 in forward, code: hidden_states = self.fc2(hidden_states)
        view_202: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_254, [512, 3072]);  convert_element_type_254 = None
        permute_113: "f16[3072, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_59: "f16[512, 768]" = torch.ops.aten.addmm.default(arg159_1, view_202, permute_113);  arg159_1 = view_202 = permute_113 = None
        view_203: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_59, [1, 512, 768]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:387 in forward, code: hidden_states = residual + hidden_states
        add_72: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_249, view_203);  convert_element_type_249 = view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_258: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_72, torch.float32);  add_72 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_258, [2], correction = 0, keepdim = True)
        getitem_90: "f32[1, 512, 1]" = var_mean_18[0]
        getitem_91: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_258, getitem_91);  convert_element_type_258 = getitem_91 = None
        add_73: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_67: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_18);  sub_24 = rsqrt_18 = None
        mul_68: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_67, arg160_1);  mul_67 = arg160_1 = None
        add_74: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_68, arg161_1);  mul_68 = arg161_1 = None
        convert_element_type_259: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_74, torch.float16);  add_74 = None
        return (convert_element_type_259,)
