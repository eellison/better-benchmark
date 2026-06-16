class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 1024][1024, 1]cuda:0", arg1_1: "bf16[50005, 768][768, 1]cuda:0", arg2_1: "bf16[1026, 768][768, 1]cuda:0", arg3_1: "bf16[768][1]cuda:0", arg4_1: "bf16[768][1]cuda:0", arg5_1: "bf16[768, 768][768, 1]cuda:0", arg6_1: "bf16[768][1]cuda:0", arg7_1: "bf16[768, 768][768, 1]cuda:0", arg8_1: "bf16[768][1]cuda:0", arg9_1: "bf16[768, 768][768, 1]cuda:0", arg10_1: "bf16[768][1]cuda:0", arg11_1: "bf16[768, 768][768, 1]cuda:0", arg12_1: "bf16[768][1]cuda:0", arg13_1: "bf16[768][1]cuda:0", arg14_1: "bf16[768][1]cuda:0", arg15_1: "bf16[3072, 768][768, 1]cuda:0", arg16_1: "bf16[3072][1]cuda:0", arg17_1: "bf16[768, 3072][3072, 1]cuda:0", arg18_1: "bf16[768][1]cuda:0", arg19_1: "bf16[768][1]cuda:0", arg20_1: "bf16[768][1]cuda:0", arg21_1: "bf16[768, 768][768, 1]cuda:0", arg22_1: "bf16[768][1]cuda:0", arg23_1: "bf16[768, 768][768, 1]cuda:0", arg24_1: "bf16[768][1]cuda:0", arg25_1: "bf16[768, 768][768, 1]cuda:0", arg26_1: "bf16[768][1]cuda:0", arg27_1: "bf16[768, 768][768, 1]cuda:0", arg28_1: "bf16[768][1]cuda:0", arg29_1: "bf16[768][1]cuda:0", arg30_1: "bf16[768][1]cuda:0", arg31_1: "bf16[3072, 768][768, 1]cuda:0", arg32_1: "bf16[3072][1]cuda:0", arg33_1: "bf16[768, 3072][3072, 1]cuda:0", arg34_1: "bf16[768][1]cuda:0", arg35_1: "bf16[768][1]cuda:0", arg36_1: "bf16[768][1]cuda:0", arg37_1: "bf16[768, 768][768, 1]cuda:0", arg38_1: "bf16[768][1]cuda:0", arg39_1: "bf16[768, 768][768, 1]cuda:0", arg40_1: "bf16[768][1]cuda:0", arg41_1: "bf16[768, 768][768, 1]cuda:0", arg42_1: "bf16[768][1]cuda:0", arg43_1: "bf16[768, 768][768, 1]cuda:0", arg44_1: "bf16[768][1]cuda:0", arg45_1: "bf16[768][1]cuda:0", arg46_1: "bf16[768][1]cuda:0", arg47_1: "bf16[3072, 768][768, 1]cuda:0", arg48_1: "bf16[3072][1]cuda:0", arg49_1: "bf16[768, 3072][3072, 1]cuda:0", arg50_1: "bf16[768][1]cuda:0", arg51_1: "bf16[768][1]cuda:0", arg52_1: "bf16[768][1]cuda:0", arg53_1: "bf16[768, 768][768, 1]cuda:0", arg54_1: "bf16[768][1]cuda:0", arg55_1: "bf16[768, 768][768, 1]cuda:0", arg56_1: "bf16[768][1]cuda:0", arg57_1: "bf16[768, 768][768, 1]cuda:0", arg58_1: "bf16[768][1]cuda:0", arg59_1: "bf16[768, 768][768, 1]cuda:0", arg60_1: "bf16[768][1]cuda:0", arg61_1: "bf16[768][1]cuda:0", arg62_1: "bf16[768][1]cuda:0", arg63_1: "bf16[3072, 768][768, 1]cuda:0", arg64_1: "bf16[3072][1]cuda:0", arg65_1: "bf16[768, 3072][3072, 1]cuda:0", arg66_1: "bf16[768][1]cuda:0", arg67_1: "bf16[768][1]cuda:0", arg68_1: "bf16[768][1]cuda:0", arg69_1: "bf16[768, 768][768, 1]cuda:0", arg70_1: "bf16[768][1]cuda:0", arg71_1: "bf16[768, 768][768, 1]cuda:0", arg72_1: "bf16[768][1]cuda:0", arg73_1: "bf16[768, 768][768, 1]cuda:0", arg74_1: "bf16[768][1]cuda:0", arg75_1: "bf16[768, 768][768, 1]cuda:0", arg76_1: "bf16[768][1]cuda:0", arg77_1: "bf16[768][1]cuda:0", arg78_1: "bf16[768][1]cuda:0", arg79_1: "bf16[3072, 768][768, 1]cuda:0", arg80_1: "bf16[3072][1]cuda:0", arg81_1: "bf16[768, 3072][3072, 1]cuda:0", arg82_1: "bf16[768][1]cuda:0", arg83_1: "bf16[768][1]cuda:0", arg84_1: "bf16[768][1]cuda:0", arg85_1: "bf16[768, 768][768, 1]cuda:0", arg86_1: "bf16[768][1]cuda:0", arg87_1: "bf16[768, 768][768, 1]cuda:0", arg88_1: "bf16[768][1]cuda:0", arg89_1: "bf16[768, 768][768, 1]cuda:0", arg90_1: "bf16[768][1]cuda:0", arg91_1: "bf16[768, 768][768, 1]cuda:0", arg92_1: "bf16[768][1]cuda:0", arg93_1: "bf16[768][1]cuda:0", arg94_1: "bf16[768][1]cuda:0", arg95_1: "bf16[3072, 768][768, 1]cuda:0", arg96_1: "bf16[3072][1]cuda:0", arg97_1: "bf16[768, 3072][3072, 1]cuda:0", arg98_1: "bf16[768][1]cuda:0", arg99_1: "bf16[768][1]cuda:0", arg100_1: "bf16[768][1]cuda:0", arg101_1: "i64[16, 1024][1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:71 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg0_1 = None
        mul: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, 27.712812921102035);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:553 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:112 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_9: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:114 in forward, code: return super().forward(position_ids + self.offset)
        add_5: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_9, 2);  unsqueeze_9 = None
        embedding_1: "bf16[1, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, add_5);  arg2_1 = add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:583 in forward, code: hidden_states = inputs_embeds + positions
        add_6: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:584 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        convert_element_type: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.float32);  add_6 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_7: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_1: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_8: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        convert_element_type_1: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:209 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [16384, 768])
        permute: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg6_1, view, permute);  arg6_1 = view = permute = None
        view_1: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [16, 1024, 768]);  addmm = None
        view_2: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [16, 1024, -1, 64]);  view_1 = None
        permute_1: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:229 in forward, code: key_states = self.k_proj(current_states)
        view_3: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [16384, 768])
        permute_2: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg8_1, view_3, permute_2);  arg8_1 = view_3 = permute_2 = None
        view_4: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [16, 1024, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:232 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [16, 1024, -1, 64]);  view_4 = None
        permute_4: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:230 in forward, code: value_states = self.v_proj(current_states)
        view_5: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [16384, 768])
        permute_3: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg10_1, view_5, permute_3);  arg10_1 = view_5 = permute_3 = None
        view_6: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [16, 1024, 768]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:233 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [16, 1024, -1, 64]);  view_6 = None
        permute_5: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[1024][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_3: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_4: "i64[1, 1, 1024][1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 1024][1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[1024][1]cuda:0" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_1: "i64[1, 1, 1024][1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1024, 1][1024, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_2);  unsqueeze_5 = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[16, 1, 1024, 1024][0, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(le, [16, -1, 1024, 1024]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[16, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_1, full_default);  full_default_1 = full_default = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_1, permute_4, permute_5, where, False, scale = 0.125);  permute_1 = permute_4 = permute_5 = where = None
        getitem_2: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:256 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [16, 1024, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:257 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [16384, 768]);  view_9 = None
        permute_7: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg12_1, view_10, permute_7);  arg12_1 = view_10 = permute_7 = None
        view_11: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [16, 1024, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:454 in forward, code: hidden_states = residual + hidden_states
        add_9: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1, view_11);  convert_element_type_1 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:455 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_14: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_14, [2], correction = 0, keepdim = True)
        getitem_11: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_1[0]
        getitem_12: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_12);  convert_element_type_14 = getitem_12 = None
        add_10: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_3: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_4: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, arg13_1);  mul_3 = arg13_1 = None
        add_11: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, arg14_1);  mul_4 = arg14_1 = None
        convert_element_type_15: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:474 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_12: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_15, [16384, 768])
        permute_8: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg16_1, view_12, permute_8);  arg16_1 = view_12 = permute_8 = None
        view_13: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [16, 1024, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_19: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_5: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_19, 0.5)
        mul_6: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_19, 0.7071067811865476);  convert_element_type_19 = None
        erf: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_12: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, add_12);  mul_5 = add_12 = None
        convert_element_type_20: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:476 in forward, code: hidden_states = self.fc2(hidden_states)
        view_14: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_20, [16384, 3072]);  convert_element_type_20 = None
        permute_9: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg18_1, view_14, permute_9);  arg18_1 = view_14 = permute_9 = None
        view_15: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [16, 1024, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:478 in forward, code: hidden_states = residual + hidden_states
        add_13: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_15, view_15);  convert_element_type_15 = view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:479 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_24: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_13: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_2[0]
        getitem_14: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_14);  convert_element_type_24 = getitem_14 = None
        add_14: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_8: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, arg19_1);  mul_8 = arg19_1 = None
        add_15: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, arg20_1);  mul_9 = arg20_1 = None
        convert_element_type_25: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:209 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_16: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [16384, 768])
        permute_10: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg22_1, view_16, permute_10);  arg22_1 = view_16 = permute_10 = None
        view_17: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [16, 1024, 768]);  addmm_6 = None
        view_18: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_17, [16, 1024, -1, 64]);  view_17 = None
        permute_11: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:229 in forward, code: key_states = self.k_proj(current_states)
        view_19: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [16384, 768])
        permute_12: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg24_1, view_19, permute_12);  arg24_1 = view_19 = permute_12 = None
        view_20: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [16, 1024, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:232 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_23: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_20, [16, 1024, -1, 64]);  view_20 = None
        permute_14: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:230 in forward, code: value_states = self.v_proj(current_states)
        view_21: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [16384, 768])
        permute_13: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg26_1, view_21, permute_13);  arg26_1 = view_21 = permute_13 = None
        view_22: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [16, 1024, 768]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:233 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_24: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_22, [16, 1024, -1, 64]);  view_22 = None
        permute_15: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[16, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_3, full_default_2);  full_default_3 = full_default_2 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_11, permute_14, permute_15, where_1, False, scale = 0.125);  permute_11 = permute_14 = permute_15 = where_1 = None
        getitem_15: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_16: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:256 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_25: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_16, [16, 1024, -1]);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:257 in forward, code: attn_output = self.out_proj(attn_output)
        view_26: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_25, [16384, 768]);  view_25 = None
        permute_17: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg28_1, view_26, permute_17);  arg28_1 = view_26 = permute_17 = None
        view_27: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [16, 1024, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:454 in forward, code: hidden_states = residual + hidden_states
        add_16: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_25, view_27);  convert_element_type_25 = view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:455 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_38: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_38, [2], correction = 0, keepdim = True)
        getitem_24: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_3[0]
        getitem_25: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_38, getitem_25);  convert_element_type_38 = getitem_25 = None
        add_17: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_10: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, arg29_1);  mul_10 = arg29_1 = None
        add_18: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, arg30_1);  mul_11 = arg30_1 = None
        convert_element_type_39: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:474 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_28: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_39, [16384, 768])
        permute_18: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg32_1, view_28, permute_18);  arg32_1 = view_28 = permute_18 = None
        view_29: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [16, 1024, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_43: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        mul_12: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.5)
        mul_13: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.7071067811865476);  convert_element_type_43 = None
        erf_1: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_19: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, add_19);  mul_12 = add_19 = None
        convert_element_type_44: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:476 in forward, code: hidden_states = self.fc2(hidden_states)
        view_30: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_44, [16384, 3072]);  convert_element_type_44 = None
        permute_19: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg34_1, view_30, permute_19);  arg34_1 = view_30 = permute_19 = None
        view_31: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [16, 1024, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:478 in forward, code: hidden_states = residual + hidden_states
        add_20: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_39, view_31);  convert_element_type_39 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:479 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_48: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.float32);  add_20 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem_26: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_4[0]
        getitem_27: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_27);  convert_element_type_48 = getitem_27 = None
        add_21: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_15: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_16: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, arg35_1);  mul_15 = arg35_1 = None
        add_22: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, arg36_1);  mul_16 = arg36_1 = None
        convert_element_type_49: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:209 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_32: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [16384, 768])
        permute_20: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg38_1, view_32, permute_20);  arg38_1 = view_32 = permute_20 = None
        view_33: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [16, 1024, 768]);  addmm_12 = None
        view_34: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_33, [16, 1024, -1, 64]);  view_33 = None
        permute_21: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:229 in forward, code: key_states = self.k_proj(current_states)
        view_35: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [16384, 768])
        permute_22: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg40_1, view_35, permute_22);  arg40_1 = view_35 = permute_22 = None
        view_36: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [16, 1024, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:232 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_39: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_36, [16, 1024, -1, 64]);  view_36 = None
        permute_24: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:230 in forward, code: value_states = self.v_proj(current_states)
        view_37: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [16384, 768])
        permute_23: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg42_1, view_37, permute_23);  arg42_1 = view_37 = permute_23 = None
        view_38: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [16, 1024, 768]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:233 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_40: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_38, [16, 1024, -1, 64]);  view_38 = None
        permute_25: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_5: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[16, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_5, full_default_4);  full_default_5 = full_default_4 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_21, permute_24, permute_25, where_2, False, scale = 0.125);  permute_21 = permute_24 = permute_25 = where_2 = None
        getitem_28: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_26: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:256 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_41: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_26, [16, 1024, -1]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:257 in forward, code: attn_output = self.out_proj(attn_output)
        view_42: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_41, [16384, 768]);  view_41 = None
        permute_27: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg44_1, view_42, permute_27);  arg44_1 = view_42 = permute_27 = None
        view_43: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [16, 1024, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:454 in forward, code: hidden_states = residual + hidden_states
        add_23: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_49, view_43);  convert_element_type_49 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:455 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_62: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_62, [2], correction = 0, keepdim = True)
        getitem_37: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_5[0]
        getitem_38: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, getitem_38);  convert_element_type_62 = getitem_38 = None
        add_24: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_17: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_18: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, arg45_1);  mul_17 = arg45_1 = None
        add_25: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, arg46_1);  mul_18 = arg46_1 = None
        convert_element_type_63: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:474 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_44: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_63, [16384, 768])
        permute_28: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg48_1, view_44, permute_28);  arg48_1 = view_44 = permute_28 = None
        view_45: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [16, 1024, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_67: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_19: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_20: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476);  convert_element_type_67 = None
        erf_2: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_26: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, add_26);  mul_19 = add_26 = None
        convert_element_type_68: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:476 in forward, code: hidden_states = self.fc2(hidden_states)
        view_46: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_68, [16384, 3072]);  convert_element_type_68 = None
        permute_29: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg50_1, view_46, permute_29);  arg50_1 = view_46 = permute_29 = None
        view_47: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [16, 1024, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:478 in forward, code: hidden_states = residual + hidden_states
        add_27: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_63, view_47);  convert_element_type_63 = view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:479 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_72: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_39: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_6[0]
        getitem_40: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_40);  convert_element_type_72 = getitem_40 = None
        add_28: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_22: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_23: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, arg51_1);  mul_22 = arg51_1 = None
        add_29: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, arg52_1);  mul_23 = arg52_1 = None
        convert_element_type_73: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:209 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_48: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [16384, 768])
        permute_30: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg54_1, view_48, permute_30);  arg54_1 = view_48 = permute_30 = None
        view_49: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [16, 1024, 768]);  addmm_18 = None
        view_50: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_49, [16, 1024, -1, 64]);  view_49 = None
        permute_31: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:229 in forward, code: key_states = self.k_proj(current_states)
        view_51: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [16384, 768])
        permute_32: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg56_1, view_51, permute_32);  arg56_1 = view_51 = permute_32 = None
        view_52: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [16, 1024, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:232 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_55: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [16, 1024, -1, 64]);  view_52 = None
        permute_34: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:230 in forward, code: value_states = self.v_proj(current_states)
        view_53: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [16384, 768])
        permute_33: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg58_1, view_53, permute_33);  arg58_1 = view_53 = permute_33 = None
        view_54: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [16, 1024, 768]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:233 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_56: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [16, 1024, -1, 64]);  view_54 = None
        permute_35: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_7: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[16, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_31, permute_34, permute_35, where_3, False, scale = 0.125);  permute_31 = permute_34 = permute_35 = where_3 = None
        getitem_41: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_36: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_41, [0, 2, 1, 3]);  getitem_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:256 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_57: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_36, [16, 1024, -1]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:257 in forward, code: attn_output = self.out_proj(attn_output)
        view_58: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_57, [16384, 768]);  view_57 = None
        permute_37: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg60_1, view_58, permute_37);  arg60_1 = view_58 = permute_37 = None
        view_59: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [16, 1024, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:454 in forward, code: hidden_states = residual + hidden_states
        add_30: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_73, view_59);  convert_element_type_73 = view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:455 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_86: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_86, [2], correction = 0, keepdim = True)
        getitem_50: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_7[0]
        getitem_51: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, getitem_51);  convert_element_type_86 = getitem_51 = None
        add_31: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_24: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_25: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, arg61_1);  mul_24 = arg61_1 = None
        add_32: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, arg62_1);  mul_25 = arg62_1 = None
        convert_element_type_87: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:474 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_60: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_87, [16384, 768])
        permute_38: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg64_1, view_60, permute_38);  arg64_1 = view_60 = permute_38 = None
        view_61: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [16, 1024, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_91: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        mul_26: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.5)
        mul_27: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.7071067811865476);  convert_element_type_91 = None
        erf_3: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_33: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, add_33);  mul_26 = add_33 = None
        convert_element_type_92: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_28, torch.bfloat16);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:476 in forward, code: hidden_states = self.fc2(hidden_states)
        view_62: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_92, [16384, 3072]);  convert_element_type_92 = None
        permute_39: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg66_1, view_62, permute_39);  arg66_1 = view_62 = permute_39 = None
        view_63: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [16, 1024, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:478 in forward, code: hidden_states = residual + hidden_states
        add_34: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_87, view_63);  convert_element_type_87 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:479 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_96: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.float32);  add_34 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_52: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_8[0]
        getitem_53: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_8: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_53);  convert_element_type_96 = getitem_53 = None
        add_35: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_29: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_30: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, arg67_1);  mul_29 = arg67_1 = None
        add_36: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_30, arg68_1);  mul_30 = arg68_1 = None
        convert_element_type_97: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:209 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_64: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [16384, 768])
        permute_40: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg70_1, view_64, permute_40);  arg70_1 = view_64 = permute_40 = None
        view_65: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [16, 1024, 768]);  addmm_24 = None
        view_66: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_65, [16, 1024, -1, 64]);  view_65 = None
        permute_41: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:229 in forward, code: key_states = self.k_proj(current_states)
        view_67: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [16384, 768])
        permute_42: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg72_1, view_67, permute_42);  arg72_1 = view_67 = permute_42 = None
        view_68: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [16, 1024, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:232 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_71: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_68, [16, 1024, -1, 64]);  view_68 = None
        permute_44: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:230 in forward, code: value_states = self.v_proj(current_states)
        view_69: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [16384, 768])
        permute_43: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg74_1, view_69, permute_43);  arg74_1 = view_69 = permute_43 = None
        view_70: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [16, 1024, 768]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:233 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_72: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_70, [16, 1024, -1, 64]);  view_70 = None
        permute_45: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[16, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_9, full_default_8);  full_default_9 = full_default_8 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_41, permute_44, permute_45, where_4, False, scale = 0.125);  permute_41 = permute_44 = permute_45 = where_4 = None
        getitem_54: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_46: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:256 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_73: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_46, [16, 1024, -1]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:257 in forward, code: attn_output = self.out_proj(attn_output)
        view_74: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [16384, 768]);  view_73 = None
        permute_47: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg76_1, view_74, permute_47);  arg76_1 = view_74 = permute_47 = None
        view_75: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [16, 1024, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:454 in forward, code: hidden_states = residual + hidden_states
        add_37: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_97, view_75);  convert_element_type_97 = view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:455 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_110: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True)
        getitem_63: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_9[0]
        getitem_64: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_64);  convert_element_type_110 = getitem_64 = None
        add_38: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
        rsqrt_9: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_31: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_32: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, arg77_1);  mul_31 = arg77_1 = None
        add_39: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, arg78_1);  mul_32 = arg78_1 = None
        convert_element_type_111: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:474 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_76: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_111, [16384, 768])
        permute_48: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg80_1, view_76, permute_48);  arg80_1 = view_76 = permute_48 = None
        view_77: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [16, 1024, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_115: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_33: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.5)
        mul_34: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.7071067811865476);  convert_element_type_115 = None
        erf_4: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_40: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
        convert_element_type_116: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:476 in forward, code: hidden_states = self.fc2(hidden_states)
        view_78: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [16384, 3072]);  convert_element_type_116 = None
        permute_49: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg82_1, view_78, permute_49);  arg82_1 = view_78 = permute_49 = None
        view_79: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [16, 1024, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:478 in forward, code: hidden_states = residual + hidden_states
        add_41: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_111, view_79);  convert_element_type_111 = view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:479 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_120: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_65: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_10[0]
        getitem_66: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_10: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_66);  convert_element_type_120 = getitem_66 = None
        add_42: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
        rsqrt_10: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_36: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_37: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg83_1);  mul_36 = arg83_1 = None
        add_43: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg84_1);  mul_37 = arg84_1 = None
        convert_element_type_121: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:209 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_80: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [16384, 768])
        permute_50: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg86_1, view_80, permute_50);  arg86_1 = view_80 = permute_50 = None
        view_81: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [16, 1024, 768]);  addmm_30 = None
        view_82: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [16, 1024, -1, 64]);  view_81 = None
        permute_51: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:229 in forward, code: key_states = self.k_proj(current_states)
        view_83: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [16384, 768])
        permute_52: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg88_1, view_83, permute_52);  arg88_1 = view_83 = permute_52 = None
        view_84: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [16, 1024, 768]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:232 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_87: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_84, [16, 1024, -1, 64]);  view_84 = None
        permute_54: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:230 in forward, code: value_states = self.v_proj(current_states)
        view_85: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [16384, 768])
        permute_53: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg90_1, view_85, permute_53);  arg90_1 = view_85 = permute_53 = None
        view_86: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [16, 1024, 768]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:233 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_88: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_86, [16, 1024, -1, 64]);  view_86 = None
        permute_55: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_11: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[16, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_11, full_default_10);  expand = full_default_11 = full_default_10 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_51, permute_54, permute_55, where_5, False, scale = 0.125);  permute_51 = permute_54 = permute_55 = where_5 = None
        getitem_67: "bf16[16, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "bf16[16, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_67, [0, 2, 1, 3]);  getitem_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:256 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_89: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_56, [16, 1024, -1]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:257 in forward, code: attn_output = self.out_proj(attn_output)
        view_90: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [16384, 768]);  view_89 = None
        permute_57: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg92_1, view_90, permute_57);  arg92_1 = view_90 = permute_57 = None
        view_91: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [16, 1024, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:454 in forward, code: hidden_states = residual + hidden_states
        add_44: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_121, view_91);  convert_element_type_121 = view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:455 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_134: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.float32);  add_44 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_134, [2], correction = 0, keepdim = True)
        getitem_76: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_11[0]
        getitem_77: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_11: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_77);  convert_element_type_134 = getitem_77 = None
        add_45: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_11: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_38: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_39: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, arg93_1);  mul_38 = arg93_1 = None
        add_46: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, arg94_1);  mul_39 = arg94_1 = None
        convert_element_type_135: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:474 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_92: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [16384, 768])
        permute_58: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg96_1, view_92, permute_58);  arg96_1 = view_92 = permute_58 = None
        view_93: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [16, 1024, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_139: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_40: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_41: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476);  convert_element_type_139 = None
        erf_5: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_47: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42: "f32[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, add_47);  mul_40 = add_47 = None
        convert_element_type_140: "bf16[16, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:476 in forward, code: hidden_states = self.fc2(hidden_states)
        view_94: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_140, [16384, 3072]);  convert_element_type_140 = None
        permute_59: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg98_1, view_94, permute_59);  arg98_1 = view_94 = permute_59 = None
        view_95: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [16, 1024, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:478 in forward, code: hidden_states = residual + hidden_states
        add_48: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_135, view_95);  convert_element_type_135 = view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:479 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_144: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_78: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_12[0]
        getitem_79: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1131 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_99: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(arg101_1, [-1]);  arg101_1 = None
        ne_1: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_99, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:479 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_12: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_79);  convert_element_type_144 = getitem_79 = None
        add_49: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_12: "f32[16, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_43: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_44: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, arg99_1);  mul_43 = arg99_1 = None
        add_50: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, arg100_1);  mul_44 = arg100_1 = None
        convert_element_type_145: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1125 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_96: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [16384, 768]);  convert_element_type_145 = None
        permute_60: "bf16[768, 50005][1, 768]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "bf16[768, 50008][50008, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_60, [0, 3, 0, 0]);  permute_60 = None
        mm_default: "bf16[16384, 50008][50008, 1]cuda:0" = torch.ops.aten.mm.default(view_96, constant_pad_nd_default);  view_96 = constant_pad_nd_default = None
        slice_tensor: "bf16[16384, 50005][50008, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        view_97: "bf16[16, 1024, 50005][51208192, 50008, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [16, 1024, 50005]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1131 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_98: "bf16[16384, 50005][50008, 1]cuda:0" = torch.ops.aten.reshape.default(view_97, [-1, 50005])
        convert_element_type_148: "f32[16384, 50005][50005, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_98, torch.float32);  view_98 = None
        amax: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_148, [1], True)
        sub_13: "f32[16384, 50005][50005, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_148, amax);  convert_element_type_148 = amax = None
        exp: "f32[16384, 50005][50005, 1]cuda:0" = torch.ops.aten.exp.default(sub_13)
        sum_1: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_14: "f32[16384, 50005][50005, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_13, log);  sub_13 = log = None
        convert_element_type_149: "bf16[16384, 50005][50005, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_14, torch.bfloat16);  sub_14 = None
        ne: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_99, -100)
        full_default_12: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "i64[16384][1]cuda:0" = torch.ops.aten.where.self(ne, view_99, full_default_12);  ne = full_default_12 = None
        unsqueeze_10: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather: "bf16[16384, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_149, 1, unsqueeze_10);  convert_element_type_149 = unsqueeze_10 = None
        squeeze: "bf16[16384][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[16384][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_13: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[16384][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_13);  ne_1 = neg = full_default_13 = None
        sum_3: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_7);  where_7 = None
        ne_2: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_99, -100);  view_99 = None
        sum_2: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_150: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.bfloat16);  sum_2 = None
        div: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_150);  sum_3 = convert_element_type_150 = None
        return (div, view_97)
