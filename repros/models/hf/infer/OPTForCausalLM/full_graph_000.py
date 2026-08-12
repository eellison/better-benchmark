class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 2048][2048, 1]cuda:0", arg1_1: "bf16[50272, 768][768, 1]cuda:0", arg2_1: "bf16[2050, 768][768, 1]cuda:0", arg3_1: "bf16[768][1]cuda:0", arg4_1: "bf16[768][1]cuda:0", arg5_1: "bf16[768, 768][768, 1]cuda:0", arg6_1: "bf16[768][1]cuda:0", arg7_1: "bf16[768, 768][768, 1]cuda:0", arg8_1: "bf16[768][1]cuda:0", arg9_1: "bf16[768, 768][768, 1]cuda:0", arg10_1: "bf16[768][1]cuda:0", arg11_1: "bf16[768, 768][768, 1]cuda:0", arg12_1: "bf16[768][1]cuda:0", arg13_1: "bf16[768][1]cuda:0", arg14_1: "bf16[768][1]cuda:0", arg15_1: "bf16[3072, 768][768, 1]cuda:0", arg16_1: "bf16[3072][1]cuda:0", arg17_1: "bf16[768, 3072][3072, 1]cuda:0", arg18_1: "bf16[768][1]cuda:0", arg19_1: "bf16[768][1]cuda:0", arg20_1: "bf16[768][1]cuda:0", arg21_1: "bf16[768, 768][768, 1]cuda:0", arg22_1: "bf16[768][1]cuda:0", arg23_1: "bf16[768, 768][768, 1]cuda:0", arg24_1: "bf16[768][1]cuda:0", arg25_1: "bf16[768, 768][768, 1]cuda:0", arg26_1: "bf16[768][1]cuda:0", arg27_1: "bf16[768, 768][768, 1]cuda:0", arg28_1: "bf16[768][1]cuda:0", arg29_1: "bf16[768][1]cuda:0", arg30_1: "bf16[768][1]cuda:0", arg31_1: "bf16[3072, 768][768, 1]cuda:0", arg32_1: "bf16[3072][1]cuda:0", arg33_1: "bf16[768, 3072][3072, 1]cuda:0", arg34_1: "bf16[768][1]cuda:0", arg35_1: "bf16[768][1]cuda:0", arg36_1: "bf16[768][1]cuda:0", arg37_1: "bf16[768, 768][768, 1]cuda:0", arg38_1: "bf16[768][1]cuda:0", arg39_1: "bf16[768, 768][768, 1]cuda:0", arg40_1: "bf16[768][1]cuda:0", arg41_1: "bf16[768, 768][768, 1]cuda:0", arg42_1: "bf16[768][1]cuda:0", arg43_1: "bf16[768, 768][768, 1]cuda:0", arg44_1: "bf16[768][1]cuda:0", arg45_1: "bf16[768][1]cuda:0", arg46_1: "bf16[768][1]cuda:0", arg47_1: "bf16[3072, 768][768, 1]cuda:0", arg48_1: "bf16[3072][1]cuda:0", arg49_1: "bf16[768, 3072][3072, 1]cuda:0", arg50_1: "bf16[768][1]cuda:0", arg51_1: "bf16[768][1]cuda:0", arg52_1: "bf16[768][1]cuda:0", arg53_1: "bf16[768, 768][768, 1]cuda:0", arg54_1: "bf16[768][1]cuda:0", arg55_1: "bf16[768, 768][768, 1]cuda:0", arg56_1: "bf16[768][1]cuda:0", arg57_1: "bf16[768, 768][768, 1]cuda:0", arg58_1: "bf16[768][1]cuda:0", arg59_1: "bf16[768, 768][768, 1]cuda:0", arg60_1: "bf16[768][1]cuda:0", arg61_1: "bf16[768][1]cuda:0", arg62_1: "bf16[768][1]cuda:0", arg63_1: "bf16[3072, 768][768, 1]cuda:0", arg64_1: "bf16[3072][1]cuda:0", arg65_1: "bf16[768, 3072][3072, 1]cuda:0", arg66_1: "bf16[768][1]cuda:0", arg67_1: "bf16[768][1]cuda:0", arg68_1: "bf16[768][1]cuda:0", arg69_1: "bf16[768, 768][768, 1]cuda:0", arg70_1: "bf16[768][1]cuda:0", arg71_1: "bf16[768, 768][768, 1]cuda:0", arg72_1: "bf16[768][1]cuda:0", arg73_1: "bf16[768, 768][768, 1]cuda:0", arg74_1: "bf16[768][1]cuda:0", arg75_1: "bf16[768, 768][768, 1]cuda:0", arg76_1: "bf16[768][1]cuda:0", arg77_1: "bf16[768][1]cuda:0", arg78_1: "bf16[768][1]cuda:0", arg79_1: "bf16[3072, 768][768, 1]cuda:0", arg80_1: "bf16[3072][1]cuda:0", arg81_1: "bf16[768, 3072][3072, 1]cuda:0", arg82_1: "bf16[768][1]cuda:0", arg83_1: "bf16[768][1]cuda:0", arg84_1: "bf16[768][1]cuda:0", arg85_1: "bf16[768, 768][768, 1]cuda:0", arg86_1: "bf16[768][1]cuda:0", arg87_1: "bf16[768, 768][768, 1]cuda:0", arg88_1: "bf16[768][1]cuda:0", arg89_1: "bf16[768, 768][768, 1]cuda:0", arg90_1: "bf16[768][1]cuda:0", arg91_1: "bf16[768, 768][768, 1]cuda:0", arg92_1: "bf16[768][1]cuda:0", arg93_1: "bf16[768][1]cuda:0", arg94_1: "bf16[768][1]cuda:0", arg95_1: "bf16[3072, 768][768, 1]cuda:0", arg96_1: "bf16[3072][1]cuda:0", arg97_1: "bf16[768, 3072][3072, 1]cuda:0", arg98_1: "bf16[768][1]cuda:0", arg99_1: "bf16[768][1]cuda:0", arg100_1: "bf16[768][1]cuda:0", arg101_1: "bf16[768, 768][768, 1]cuda:0", arg102_1: "bf16[768][1]cuda:0", arg103_1: "bf16[768, 768][768, 1]cuda:0", arg104_1: "bf16[768][1]cuda:0", arg105_1: "bf16[768, 768][768, 1]cuda:0", arg106_1: "bf16[768][1]cuda:0", arg107_1: "bf16[768, 768][768, 1]cuda:0", arg108_1: "bf16[768][1]cuda:0", arg109_1: "bf16[768][1]cuda:0", arg110_1: "bf16[768][1]cuda:0", arg111_1: "bf16[3072, 768][768, 1]cuda:0", arg112_1: "bf16[3072][1]cuda:0", arg113_1: "bf16[768, 3072][3072, 1]cuda:0", arg114_1: "bf16[768][1]cuda:0", arg115_1: "bf16[768][1]cuda:0", arg116_1: "bf16[768][1]cuda:0", arg117_1: "bf16[768, 768][768, 1]cuda:0", arg118_1: "bf16[768][1]cuda:0", arg119_1: "bf16[768, 768][768, 1]cuda:0", arg120_1: "bf16[768][1]cuda:0", arg121_1: "bf16[768, 768][768, 1]cuda:0", arg122_1: "bf16[768][1]cuda:0", arg123_1: "bf16[768, 768][768, 1]cuda:0", arg124_1: "bf16[768][1]cuda:0", arg125_1: "bf16[768][1]cuda:0", arg126_1: "bf16[768][1]cuda:0", arg127_1: "bf16[3072, 768][768, 1]cuda:0", arg128_1: "bf16[3072][1]cuda:0", arg129_1: "bf16[768, 3072][3072, 1]cuda:0", arg130_1: "bf16[768][1]cuda:0", arg131_1: "bf16[768][1]cuda:0", arg132_1: "bf16[768][1]cuda:0", arg133_1: "bf16[768, 768][768, 1]cuda:0", arg134_1: "bf16[768][1]cuda:0", arg135_1: "bf16[768, 768][768, 1]cuda:0", arg136_1: "bf16[768][1]cuda:0", arg137_1: "bf16[768, 768][768, 1]cuda:0", arg138_1: "bf16[768][1]cuda:0", arg139_1: "bf16[768, 768][768, 1]cuda:0", arg140_1: "bf16[768][1]cuda:0", arg141_1: "bf16[768][1]cuda:0", arg142_1: "bf16[768][1]cuda:0", arg143_1: "bf16[3072, 768][768, 1]cuda:0", arg144_1: "bf16[3072][1]cuda:0", arg145_1: "bf16[768, 3072][3072, 1]cuda:0", arg146_1: "bf16[768][1]cuda:0", arg147_1: "bf16[768][1]cuda:0", arg148_1: "bf16[768][1]cuda:0", arg149_1: "bf16[768, 768][768, 1]cuda:0", arg150_1: "bf16[768][1]cuda:0", arg151_1: "bf16[768, 768][768, 1]cuda:0", arg152_1: "bf16[768][1]cuda:0", arg153_1: "bf16[768, 768][768, 1]cuda:0", arg154_1: "bf16[768][1]cuda:0", arg155_1: "bf16[768, 768][768, 1]cuda:0", arg156_1: "bf16[768][1]cuda:0", arg157_1: "bf16[768][1]cuda:0", arg158_1: "bf16[768][1]cuda:0", arg159_1: "bf16[3072, 768][768, 1]cuda:0", arg160_1: "bf16[3072][1]cuda:0", arg161_1: "bf16[768, 3072][3072, 1]cuda:0", arg162_1: "bf16[768][1]cuda:0", arg163_1: "bf16[768][1]cuda:0", arg164_1: "bf16[768][1]cuda:0", arg165_1: "bf16[768, 768][768, 1]cuda:0", arg166_1: "bf16[768][1]cuda:0", arg167_1: "bf16[768, 768][768, 1]cuda:0", arg168_1: "bf16[768][1]cuda:0", arg169_1: "bf16[768, 768][768, 1]cuda:0", arg170_1: "bf16[768][1]cuda:0", arg171_1: "bf16[768, 768][768, 1]cuda:0", arg172_1: "bf16[768][1]cuda:0", arg173_1: "bf16[768][1]cuda:0", arg174_1: "bf16[768][1]cuda:0", arg175_1: "bf16[3072, 768][768, 1]cuda:0", arg176_1: "bf16[3072][1]cuda:0", arg177_1: "bf16[768, 3072][3072, 1]cuda:0", arg178_1: "bf16[768][1]cuda:0", arg179_1: "bf16[768][1]cuda:0", arg180_1: "bf16[768][1]cuda:0", arg181_1: "bf16[768, 768][768, 1]cuda:0", arg182_1: "bf16[768][1]cuda:0", arg183_1: "bf16[768, 768][768, 1]cuda:0", arg184_1: "bf16[768][1]cuda:0", arg185_1: "bf16[768, 768][768, 1]cuda:0", arg186_1: "bf16[768][1]cuda:0", arg187_1: "bf16[768, 768][768, 1]cuda:0", arg188_1: "bf16[768][1]cuda:0", arg189_1: "bf16[768][1]cuda:0", arg190_1: "bf16[768][1]cuda:0", arg191_1: "bf16[3072, 768][768, 1]cuda:0", arg192_1: "bf16[3072][1]cuda:0", arg193_1: "bf16[768, 3072][3072, 1]cuda:0", arg194_1: "bf16[768][1]cuda:0", arg195_1: "bf16[768][1]cuda:0", arg196_1: "bf16[768][1]cuda:0", arg197_1: "i64[4, 2048][2048, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:338 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:347 in forward, code: attention_mask = torch.ones(inputs_embeds.shape[0], seq_length, device=inputs_embeds.device)
        full_default: "f32[4, 2048][2048, 1]cuda:0" = torch.ops.aten.full.default([4, 2048], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:352 in forward, code: position_ids = (position_ids * attention_mask - 1).long()
        cumsum: "f32[4, 2048][2048, 1]cuda:0" = torch.ops.aten.cumsum.default(full_default, 1);  full_default = None
        sub: "f32[4, 2048][2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(cumsum, 1);  cumsum = None
        convert_element_type: "i64[4, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub, torch.int64);  sub = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:70 in forward, code: return super().forward(position_ids + self.offset)
        add_2: "i64[4, 2048][2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type, 2);  convert_element_type = None
        embedding_1: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, add_2);  arg2_1 = add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:368 in forward, code: hidden_states = inputs_embeds + pos_embeds.to(inputs_embeds.device)
        add_3: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_2: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub_1: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_1);  convert_element_type_2 = getitem_1 = None
        add_4: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_1: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_2: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_5: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        convert_element_type_3: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_1: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_3, [8192, 768])
        permute: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg6_1, view_1, permute);  arg6_1 = view_1 = permute = None
        view_2: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [4, 2048, 768]);  addmm = None
        mul_3: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_2, 0.125);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_3: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_3, [4, -1, 12, 64]);  mul_3 = None
        permute_1: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_4: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_3, [8192, 768])
        permute_2: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg8_1, view_4, permute_2);  arg8_1 = view_4 = permute_2 = None
        view_5: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [4, 2048, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_8: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [4, -1, 12, 64]);  view_5 = None
        permute_4: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_6: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_3, [8192, 768]);  convert_element_type_3 = None
        permute_3: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg10_1, view_6, permute_3);  arg10_1 = view_6 = permute_3 = None
        view_7: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [4, 2048, 768]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_9: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_7, [4, -1, 12, 64]);  view_7 = None
        permute_5: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_1: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_3: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[2048][1]cuda:0" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_6: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_7: "i64[1, 1, 2048][2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 1, 2048][2048, 2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[2048][1]cuda:0" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_3: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_4: "i64[1, 1, 2048][2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 2048, 1][2048, 2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_8, unsqueeze_5);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full_1, le);  full_1 = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:841 in _preprocess_mask_arguments, code: attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        full_default_1: "b8[4, 2048][2048, 1]cuda:0" = torch.ops.aten.full.default([4, 2048], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota: "i64[4][1]cuda:0" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze: "i64[4, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota, 1);  iota = None
        unsqueeze_1: "i64[4, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "i64[4, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:157 in inner_mask, code: return padding_mask[batch_idx, kv_idx]
        index: "b8[4, 1, 1, 2048][2048, 2048, 2048, 1]cuda:0" = torch.ops.aten.index.Tensor(full_default_1, [unsqueeze_2, unsqueeze_8]);  full_default_1 = unsqueeze_2 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, index);  bitwise_and = index = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.expand.default(bitwise_and_1, [4, -1, 2048, 2048]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_3, full_default_2);  full_default_3 = full_default_2 = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_1, permute_4, permute_5, where, False, scale = 1.0);  permute_1 = permute_4 = permute_5 = where = None
        getitem_2: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_10: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [4, 2048, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_11: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_10, [8192, 768]);  view_10 = None
        permute_7: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg12_1, view_11, permute_7);  arg12_1 = view_11 = permute_7 = None
        view_12: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [4, 2048, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_6: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3, view_12);  add_3 = view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_13: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_6, [-1, 768]);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_16: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_16, [1], correction = 0, keepdim = True)
        getitem_11: "f32[8192, 1][1, 1]cuda:0" = var_mean_1[0]
        getitem_12: "f32[8192, 1][1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_12);  convert_element_type_16 = getitem_12 = None
        add_7: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_4: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
        add_8: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        convert_element_type_17: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_8: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg16_1, convert_element_type_17, permute_8);  arg16_1 = convert_element_type_17 = permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_9: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg18_1, relu, permute_9);  arg18_1 = relu = permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_9: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_13, addmm_5);  view_13 = addmm_5 = None
        view_14: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_9, [4, 2048, 768]);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_24: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_14, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_13: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_2[0]
        getitem_14: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_14);  convert_element_type_24 = getitem_14 = None
        add_10: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_6: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_7: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, arg19_1);  mul_6 = arg19_1 = None
        add_11: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_7, arg20_1);  mul_7 = arg20_1 = None
        convert_element_type_25: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_15: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [8192, 768])
        permute_10: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg22_1, view_15, permute_10);  arg22_1 = view_15 = permute_10 = None
        view_16: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [4, 2048, 768]);  addmm_6 = None
        mul_8: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_16, 0.125);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_17: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_8, [4, -1, 12, 64]);  mul_8 = None
        permute_11: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_18: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [8192, 768])
        permute_12: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg24_1, view_18, permute_12);  arg24_1 = view_18 = permute_12 = None
        view_19: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [4, 2048, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_22: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_19, [4, -1, 12, 64]);  view_19 = None
        permute_14: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_20: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [8192, 768]);  convert_element_type_25 = None
        permute_13: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg26_1, view_20, permute_13);  arg26_1 = view_20 = permute_13 = None
        view_21: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [4, 2048, 768]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_23: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_21, [4, -1, 12, 64]);  view_21 = None
        permute_15: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_5: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_5, full_default_4);  full_default_5 = full_default_4 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_11, permute_14, permute_15, where_1, False, scale = 1.0);  permute_11 = permute_14 = permute_15 = where_1 = None
        getitem_15: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_16: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_24: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_16, [4, 2048, -1]);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_25: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_24, [8192, 768]);  view_24 = None
        permute_17: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg28_1, view_25, permute_17);  arg28_1 = view_25 = permute_17 = None
        view_26: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [4, 2048, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_12: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_14, view_26);  view_14 = view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_27: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_12, [-1, 768]);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_38: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_27, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_38, [1], correction = 0, keepdim = True)
        getitem_24: "f32[8192, 1][1, 1]cuda:0" = var_mean_3[0]
        getitem_25: "f32[8192, 1][1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_4: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_38, getitem_25);  convert_element_type_38 = getitem_25 = None
        add_13: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_9: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_10: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, arg29_1);  mul_9 = arg29_1 = None
        add_14: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, arg30_1);  mul_10 = arg30_1 = None
        convert_element_type_39: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_18: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg32_1, convert_element_type_39, permute_18);  arg32_1 = convert_element_type_39 = permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_1: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_10);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_19: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg34_1, relu_1, permute_19);  arg34_1 = relu_1 = permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_15: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_27, addmm_11);  view_27 = addmm_11 = None
        view_28: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_15, [4, 2048, 768]);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_46: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_28, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_46, [2], correction = 0, keepdim = True)
        getitem_26: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_4[0]
        getitem_27: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_5: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_46, getitem_27);  convert_element_type_46 = getitem_27 = None
        add_16: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_11: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_12: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, arg35_1);  mul_11 = arg35_1 = None
        add_17: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_12, arg36_1);  mul_12 = arg36_1 = None
        convert_element_type_47: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_29: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_47, [8192, 768])
        permute_20: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg38_1, view_29, permute_20);  arg38_1 = view_29 = permute_20 = None
        view_30: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [4, 2048, 768]);  addmm_12 = None
        mul_13: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_30, 0.125);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_31: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_13, [4, -1, 12, 64]);  mul_13 = None
        permute_21: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_32: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_47, [8192, 768])
        permute_22: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg40_1, view_32, permute_22);  arg40_1 = view_32 = permute_22 = None
        view_33: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [4, 2048, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_36: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_33, [4, -1, 12, 64]);  view_33 = None
        permute_24: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_34: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_47, [8192, 768]);  convert_element_type_47 = None
        permute_23: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg42_1, view_34, permute_23);  arg42_1 = view_34 = permute_23 = None
        view_35: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [4, 2048, 768]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_37: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_35, [4, -1, 12, 64]);  view_35 = None
        permute_25: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_7: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_21, permute_24, permute_25, where_2, False, scale = 1.0);  permute_21 = permute_24 = permute_25 = where_2 = None
        getitem_28: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_26: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_38: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_26, [4, 2048, -1]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_39: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_38, [8192, 768]);  view_38 = None
        permute_27: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg44_1, view_39, permute_27);  arg44_1 = view_39 = permute_27 = None
        view_40: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [4, 2048, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_18: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_28, view_40);  view_28 = view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_41: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_18, [-1, 768]);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_60: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_60, [1], correction = 0, keepdim = True)
        getitem_37: "f32[8192, 1][1, 1]cuda:0" = var_mean_5[0]
        getitem_38: "f32[8192, 1][1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_6: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_60, getitem_38);  convert_element_type_60 = getitem_38 = None
        add_19: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_14: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
        mul_15: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, arg45_1);  mul_14 = arg45_1 = None
        add_20: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, arg46_1);  mul_15 = arg46_1 = None
        convert_element_type_61: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_28: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg48_1, convert_element_type_61, permute_28);  arg48_1 = convert_element_type_61 = permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_2: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_16);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_29: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg50_1, relu_2, permute_29);  arg50_1 = relu_2 = permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_21: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_41, addmm_17);  view_41 = addmm_17 = None
        view_42: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_21, [4, 2048, 768]);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_68: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_42, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_68, [2], correction = 0, keepdim = True)
        getitem_39: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_6[0]
        getitem_40: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_7: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_68, getitem_40);  convert_element_type_68 = getitem_40 = None
        add_22: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_16: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = rsqrt_6 = None
        mul_17: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, arg51_1);  mul_16 = arg51_1 = None
        add_23: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, arg52_1);  mul_17 = arg52_1 = None
        convert_element_type_69: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_43: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_69, [8192, 768])
        permute_30: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg54_1, view_43, permute_30);  arg54_1 = view_43 = permute_30 = None
        view_44: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [4, 2048, 768]);  addmm_18 = None
        mul_18: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_44, 0.125);  view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_45: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_18, [4, -1, 12, 64]);  mul_18 = None
        permute_31: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_46: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_69, [8192, 768])
        permute_32: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg56_1, view_46, permute_32);  arg56_1 = view_46 = permute_32 = None
        view_47: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [4, 2048, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_50: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_47, [4, -1, 12, 64]);  view_47 = None
        permute_34: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_48: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_69, [8192, 768]);  convert_element_type_69 = None
        permute_33: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg58_1, view_48, permute_33);  arg58_1 = view_48 = permute_33 = None
        view_49: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [4, 2048, 768]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_51: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_49, [4, -1, 12, 64]);  view_49 = None
        permute_35: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_9, full_default_8);  full_default_9 = full_default_8 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_31, permute_34, permute_35, where_3, False, scale = 1.0);  permute_31 = permute_34 = permute_35 = where_3 = None
        getitem_41: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_36: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_41, [0, 2, 1, 3]);  getitem_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_52: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_36, [4, 2048, -1]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_53: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [8192, 768]);  view_52 = None
        permute_37: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg60_1, view_53, permute_37);  arg60_1 = view_53 = permute_37 = None
        view_54: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [4, 2048, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_24: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_42, view_54);  view_42 = view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_55: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_24, [-1, 768]);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_82: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_55, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_82, [1], correction = 0, keepdim = True)
        getitem_50: "f32[8192, 1][1, 1]cuda:0" = var_mean_7[0]
        getitem_51: "f32[8192, 1][1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_8: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_82, getitem_51);  convert_element_type_82 = getitem_51 = None
        add_25: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_19: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = rsqrt_7 = None
        mul_20: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, arg61_1);  mul_19 = arg61_1 = None
        add_26: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, arg62_1);  mul_20 = arg62_1 = None
        convert_element_type_83: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_38: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg64_1, convert_element_type_83, permute_38);  arg64_1 = convert_element_type_83 = permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_3: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_22);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_39: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg66_1, relu_3, permute_39);  arg66_1 = relu_3 = permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_27: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_55, addmm_23);  view_55 = addmm_23 = None
        view_56: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_27, [4, 2048, 768]);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_90: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_56, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True)
        getitem_52: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_8[0]
        getitem_53: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_9: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_90, getitem_53);  convert_element_type_90 = getitem_53 = None
        add_28: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_21: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = rsqrt_8 = None
        mul_22: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, arg67_1);  mul_21 = arg67_1 = None
        add_29: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, arg68_1);  mul_22 = arg68_1 = None
        convert_element_type_91: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_57: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [8192, 768])
        permute_40: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg70_1, view_57, permute_40);  arg70_1 = view_57 = permute_40 = None
        view_58: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [4, 2048, 768]);  addmm_24 = None
        mul_23: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, 0.125);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_59: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_23, [4, -1, 12, 64]);  mul_23 = None
        permute_41: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_60: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [8192, 768])
        permute_42: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg72_1, view_60, permute_42);  arg72_1 = view_60 = permute_42 = None
        view_61: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [4, 2048, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_64: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_61, [4, -1, 12, 64]);  view_61 = None
        permute_44: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_62: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [8192, 768]);  convert_element_type_91 = None
        permute_43: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg74_1, view_62, permute_43);  arg74_1 = view_62 = permute_43 = None
        view_63: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [4, 2048, 768]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_65: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_63, [4, -1, 12, 64]);  view_63 = None
        permute_45: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_11: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_11, full_default_10);  full_default_11 = full_default_10 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_41, permute_44, permute_45, where_4, False, scale = 1.0);  permute_41 = permute_44 = permute_45 = where_4 = None
        getitem_54: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_46: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_66: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_46, [4, 2048, -1]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_67: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_66, [8192, 768]);  view_66 = None
        permute_47: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg76_1, view_67, permute_47);  arg76_1 = view_67 = permute_47 = None
        view_68: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [4, 2048, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_30: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_56, view_68);  view_56 = view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_69: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_30, [-1, 768]);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_104: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_69, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_104, [1], correction = 0, keepdim = True)
        getitem_63: "f32[8192, 1][1, 1]cuda:0" = var_mean_9[0]
        getitem_64: "f32[8192, 1][1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_10: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_104, getitem_64);  convert_element_type_104 = getitem_64 = None
        add_31: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
        rsqrt_9: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_24: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = rsqrt_9 = None
        mul_25: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, arg77_1);  mul_24 = arg77_1 = None
        add_32: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, arg78_1);  mul_25 = arg78_1 = None
        convert_element_type_105: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_48: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg80_1, convert_element_type_105, permute_48);  arg80_1 = convert_element_type_105 = permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_4: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_28);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_49: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg82_1, relu_4, permute_49);  arg82_1 = relu_4 = permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_33: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_69, addmm_29);  view_69 = addmm_29 = None
        view_70: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_33, [4, 2048, 768]);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_112: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_70, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_112, [2], correction = 0, keepdim = True)
        getitem_65: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_10[0]
        getitem_66: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_11: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_112, getitem_66);  convert_element_type_112 = getitem_66 = None
        add_34: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
        rsqrt_10: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_26: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
        mul_27: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, arg83_1);  mul_26 = arg83_1 = None
        add_35: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, arg84_1);  mul_27 = arg84_1 = None
        convert_element_type_113: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_71: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_113, [8192, 768])
        permute_50: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg86_1, view_71, permute_50);  arg86_1 = view_71 = permute_50 = None
        view_72: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [4, 2048, 768]);  addmm_30 = None
        mul_28: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_72, 0.125);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_73: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_28, [4, -1, 12, 64]);  mul_28 = None
        permute_51: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_74: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_113, [8192, 768])
        permute_52: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg88_1, view_74, permute_52);  arg88_1 = view_74 = permute_52 = None
        view_75: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [4, 2048, 768]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_78: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_75, [4, -1, 12, 64]);  view_75 = None
        permute_54: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_76: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_113, [8192, 768]);  convert_element_type_113 = None
        permute_53: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg90_1, view_76, permute_53);  arg90_1 = view_76 = permute_53 = None
        view_77: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [4, 2048, 768]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_79: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_77, [4, -1, 12, 64]);  view_77 = None
        permute_55: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_13: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_51, permute_54, permute_55, where_5, False, scale = 1.0);  permute_51 = permute_54 = permute_55 = where_5 = None
        getitem_67: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_67, [0, 2, 1, 3]);  getitem_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_80: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_56, [4, 2048, -1]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_81: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_80, [8192, 768]);  view_80 = None
        permute_57: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg92_1, view_81, permute_57);  arg92_1 = view_81 = permute_57 = None
        view_82: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [4, 2048, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_36: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_70, view_82);  view_70 = view_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_83: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_36, [-1, 768]);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_126: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_83, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_126, [1], correction = 0, keepdim = True)
        getitem_76: "f32[8192, 1][1, 1]cuda:0" = var_mean_11[0]
        getitem_77: "f32[8192, 1][1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_12: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_126, getitem_77);  convert_element_type_126 = getitem_77 = None
        add_37: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_11: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_29: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = rsqrt_11 = None
        mul_30: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, arg93_1);  mul_29 = arg93_1 = None
        add_38: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_30, arg94_1);  mul_30 = arg94_1 = None
        convert_element_type_127: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_58: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg96_1, convert_element_type_127, permute_58);  arg96_1 = convert_element_type_127 = permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_5: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_34);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_59: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg98_1, relu_5, permute_59);  arg98_1 = relu_5 = permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_39: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_83, addmm_35);  view_83 = addmm_35 = None
        view_84: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_39, [4, 2048, 768]);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_134: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_84, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_134, [2], correction = 0, keepdim = True)
        getitem_78: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_12[0]
        getitem_79: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_13: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_79);  convert_element_type_134 = getitem_79 = None
        add_40: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_12: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_31: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = rsqrt_12 = None
        mul_32: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, arg99_1);  mul_31 = arg99_1 = None
        add_41: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, arg100_1);  mul_32 = arg100_1 = None
        convert_element_type_135: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_85: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [8192, 768])
        permute_60: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg102_1, view_85, permute_60);  arg102_1 = view_85 = permute_60 = None
        view_86: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [4, 2048, 768]);  addmm_36 = None
        mul_33: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_86, 0.125);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_87: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_33, [4, -1, 12, 64]);  mul_33 = None
        permute_61: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_88: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [8192, 768])
        permute_62: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_37: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg104_1, view_88, permute_62);  arg104_1 = view_88 = permute_62 = None
        view_89: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [4, 2048, 768]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_92: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [4, -1, 12, 64]);  view_89 = None
        permute_64: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_90: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [8192, 768]);  convert_element_type_135 = None
        permute_63: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_38: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg106_1, view_90, permute_63);  arg106_1 = view_90 = permute_63 = None
        view_91: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [4, 2048, 768]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_93: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_91, [4, -1, 12, 64]);  view_91 = None
        permute_65: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_15: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_15, full_default_14);  full_default_15 = full_default_14 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_61, permute_64, permute_65, where_6, False, scale = 1.0);  permute_61 = permute_64 = permute_65 = where_6 = None
        getitem_80: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_66: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_80, [0, 2, 1, 3]);  getitem_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_94: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_66, [4, 2048, -1]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_95: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [8192, 768]);  view_94 = None
        permute_67: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_39: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg108_1, view_95, permute_67);  arg108_1 = view_95 = permute_67 = None
        view_96: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [4, 2048, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_42: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_84, view_96);  view_84 = view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_97: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_42, [-1, 768]);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_148: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_97, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_148, [1], correction = 0, keepdim = True)
        getitem_89: "f32[8192, 1][1, 1]cuda:0" = var_mean_13[0]
        getitem_90: "f32[8192, 1][1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_14: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_148, getitem_90);  convert_element_type_148 = getitem_90 = None
        add_43: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_89, 1e-05);  getitem_89 = None
        rsqrt_13: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_34: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = rsqrt_13 = None
        mul_35: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, arg109_1);  mul_34 = arg109_1 = None
        add_44: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_35, arg110_1);  mul_35 = arg110_1 = None
        convert_element_type_149: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_68: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg112_1, convert_element_type_149, permute_68);  arg112_1 = convert_element_type_149 = permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_6: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_40);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_69: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_41: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg114_1, relu_6, permute_69);  arg114_1 = relu_6 = permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_45: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_97, addmm_41);  view_97 = addmm_41 = None
        view_98: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_45, [4, 2048, 768]);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_156: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_98, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_156, [2], correction = 0, keepdim = True)
        getitem_91: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_14[0]
        getitem_92: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_15: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_156, getitem_92);  convert_element_type_156 = getitem_92 = None
        add_46: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
        rsqrt_14: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_36: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_14);  sub_15 = rsqrt_14 = None
        mul_37: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg115_1);  mul_36 = arg115_1 = None
        add_47: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg116_1);  mul_37 = arg116_1 = None
        convert_element_type_157: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_99: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_157, [8192, 768])
        permute_70: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg118_1, view_99, permute_70);  arg118_1 = view_99 = permute_70 = None
        view_100: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [4, 2048, 768]);  addmm_42 = None
        mul_38: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_100, 0.125);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_101: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_38, [4, -1, 12, 64]);  mul_38 = None
        permute_71: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_102: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_157, [8192, 768])
        permute_72: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_43: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg120_1, view_102, permute_72);  arg120_1 = view_102 = permute_72 = None
        view_103: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [4, 2048, 768]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_106: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [4, -1, 12, 64]);  view_103 = None
        permute_74: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_104: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_157, [8192, 768]);  convert_element_type_157 = None
        permute_73: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg122_1, view_104, permute_73);  arg122_1 = view_104 = permute_73 = None
        view_105: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [4, 2048, 768]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_107: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_105, [4, -1, 12, 64]);  view_105 = None
        permute_75: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_17: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_17, full_default_16);  full_default_17 = full_default_16 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_71, permute_74, permute_75, where_7, False, scale = 1.0);  permute_71 = permute_74 = permute_75 = where_7 = None
        getitem_93: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_76: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_108: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_76, [4, 2048, -1]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_109: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_108, [8192, 768]);  view_108 = None
        permute_77: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_45: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg124_1, view_109, permute_77);  arg124_1 = view_109 = permute_77 = None
        view_110: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [4, 2048, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_48: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_98, view_110);  view_98 = view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_111: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_48, [-1, 768]);  add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_170: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_111, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_170, [1], correction = 0, keepdim = True)
        getitem_102: "f32[8192, 1][1, 1]cuda:0" = var_mean_15[0]
        getitem_103: "f32[8192, 1][1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_16: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_170, getitem_103);  convert_element_type_170 = getitem_103 = None
        add_49: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_15: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_39: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_15);  sub_16 = rsqrt_15 = None
        mul_40: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, arg125_1);  mul_39 = arg125_1 = None
        add_50: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_40, arg126_1);  mul_40 = arg126_1 = None
        convert_element_type_171: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_78: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg128_1, convert_element_type_171, permute_78);  arg128_1 = convert_element_type_171 = permute_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_7: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_46);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_79: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_47: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg130_1, relu_7, permute_79);  arg130_1 = relu_7 = permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_51: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_111, addmm_47);  view_111 = addmm_47 = None
        view_112: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_51, [4, 2048, 768]);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_178: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_112, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_178, [2], correction = 0, keepdim = True)
        getitem_104: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_16[0]
        getitem_105: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_17: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_178, getitem_105);  convert_element_type_178 = getitem_105 = None
        add_52: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_16: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_41: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_16);  sub_17 = rsqrt_16 = None
        mul_42: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, arg131_1);  mul_41 = arg131_1 = None
        add_53: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_42, arg132_1);  mul_42 = arg132_1 = None
        convert_element_type_179: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_113: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_179, [8192, 768])
        permute_80: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg134_1, view_113, permute_80);  arg134_1 = view_113 = permute_80 = None
        view_114: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [4, 2048, 768]);  addmm_48 = None
        mul_43: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_114, 0.125);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_115: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_43, [4, -1, 12, 64]);  mul_43 = None
        permute_81: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_116: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_179, [8192, 768])
        permute_82: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg136_1, view_116, permute_82);  arg136_1 = view_116 = permute_82 = None
        view_117: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [4, 2048, 768]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_120: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_117, [4, -1, 12, 64]);  view_117 = None
        permute_84: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_118: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_179, [8192, 768]);  convert_element_type_179 = None
        permute_83: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_50: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg138_1, view_118, permute_83);  arg138_1 = view_118 = permute_83 = None
        view_119: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [4, 2048, 768]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_121: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_119, [4, -1, 12, 64]);  view_119 = None
        permute_85: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_19: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_81, permute_84, permute_85, where_8, False, scale = 1.0);  permute_81 = permute_84 = permute_85 = where_8 = None
        getitem_106: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_86: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_122: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_86, [4, 2048, -1]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_123: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_122, [8192, 768]);  view_122 = None
        permute_87: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_51: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg140_1, view_123, permute_87);  arg140_1 = view_123 = permute_87 = None
        view_124: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [4, 2048, 768]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_54: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_112, view_124);  view_112 = view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_125: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_54, [-1, 768]);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_192: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_125, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_192, [1], correction = 0, keepdim = True)
        getitem_115: "f32[8192, 1][1, 1]cuda:0" = var_mean_17[0]
        getitem_116: "f32[8192, 1][1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_18: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_192, getitem_116);  convert_element_type_192 = getitem_116 = None
        add_55: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
        rsqrt_17: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_44: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_17);  sub_18 = rsqrt_17 = None
        mul_45: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, arg141_1);  mul_44 = arg141_1 = None
        add_56: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_45, arg142_1);  mul_45 = arg142_1 = None
        convert_element_type_193: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_88: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_52: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg144_1, convert_element_type_193, permute_88);  arg144_1 = convert_element_type_193 = permute_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_8: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_52);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_89: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_53: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg146_1, relu_8, permute_89);  arg146_1 = relu_8 = permute_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_57: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_125, addmm_53);  view_125 = addmm_53 = None
        view_126: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_57, [4, 2048, 768]);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_200: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_126, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_200, [2], correction = 0, keepdim = True)
        getitem_117: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_18[0]
        getitem_118: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_19: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_200, getitem_118);  convert_element_type_200 = getitem_118 = None
        add_58: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_117, 1e-05);  getitem_117 = None
        rsqrt_18: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_46: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = rsqrt_18 = None
        mul_47: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, arg147_1);  mul_46 = arg147_1 = None
        add_59: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_47, arg148_1);  mul_47 = arg148_1 = None
        convert_element_type_201: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_127: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_201, [8192, 768])
        permute_90: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg150_1, view_127, permute_90);  arg150_1 = view_127 = permute_90 = None
        view_128: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [4, 2048, 768]);  addmm_54 = None
        mul_48: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_128, 0.125);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_129: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_48, [4, -1, 12, 64]);  mul_48 = None
        permute_91: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_130: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_201, [8192, 768])
        permute_92: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg152_1, view_130, permute_92);  arg152_1 = view_130 = permute_92 = None
        view_131: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [4, 2048, 768]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_134: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_131, [4, -1, 12, 64]);  view_131 = None
        permute_94: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_132: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_201, [8192, 768]);  convert_element_type_201 = None
        permute_93: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_56: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg154_1, view_132, permute_93);  arg154_1 = view_132 = permute_93 = None
        view_133: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [4, 2048, 768]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_135: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [4, -1, 12, 64]);  view_133 = None
        permute_95: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_21: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_21, full_default_20);  full_default_21 = full_default_20 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_91, permute_94, permute_95, where_9, False, scale = 1.0);  permute_91 = permute_94 = permute_95 = where_9 = None
        getitem_119: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_96: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_136: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_96, [4, 2048, -1]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_137: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_136, [8192, 768]);  view_136 = None
        permute_97: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_57: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg156_1, view_137, permute_97);  arg156_1 = view_137 = permute_97 = None
        view_138: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [4, 2048, 768]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_60: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_126, view_138);  view_126 = view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_139: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_60, [-1, 768]);  add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_214: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_139, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_214, [1], correction = 0, keepdim = True)
        getitem_128: "f32[8192, 1][1, 1]cuda:0" = var_mean_19[0]
        getitem_129: "f32[8192, 1][1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_20: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_214, getitem_129);  convert_element_type_214 = getitem_129 = None
        add_61: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_19: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_49: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = rsqrt_19 = None
        mul_50: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, arg157_1);  mul_49 = arg157_1 = None
        add_62: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, arg158_1);  mul_50 = arg158_1 = None
        convert_element_type_215: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_98: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg160_1, convert_element_type_215, permute_98);  arg160_1 = convert_element_type_215 = permute_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_9: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_58);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_99: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg162_1, relu_9, permute_99);  arg162_1 = relu_9 = permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_63: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_139, addmm_59);  view_139 = addmm_59 = None
        view_140: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_63, [4, 2048, 768]);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_222: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_140, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_222, [2], correction = 0, keepdim = True)
        getitem_130: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_20[0]
        getitem_131: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_21: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_222, getitem_131);  convert_element_type_222 = getitem_131 = None
        add_64: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_20: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_51: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = rsqrt_20 = None
        mul_52: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, arg163_1);  mul_51 = arg163_1 = None
        add_65: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_52, arg164_1);  mul_52 = arg164_1 = None
        convert_element_type_223: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_141: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_223, [8192, 768])
        permute_100: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg166_1, view_141, permute_100);  arg166_1 = view_141 = permute_100 = None
        view_142: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [4, 2048, 768]);  addmm_60 = None
        mul_53: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_142, 0.125);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_143: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_53, [4, -1, 12, 64]);  mul_53 = None
        permute_101: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_144: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_223, [8192, 768])
        permute_102: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg168_1, view_144, permute_102);  arg168_1 = view_144 = permute_102 = None
        view_145: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [4, 2048, 768]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_148: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_145, [4, -1, 12, 64]);  view_145 = None
        permute_104: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_146: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_223, [8192, 768]);  convert_element_type_223 = None
        permute_103: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_62: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg170_1, view_146, permute_103);  arg170_1 = view_146 = permute_103 = None
        view_147: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [4, 2048, 768]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_149: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [4, -1, 12, 64]);  view_147 = None
        permute_105: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_23: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_23, full_default_22);  full_default_23 = full_default_22 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_101, permute_104, permute_105, where_10, False, scale = 1.0);  permute_101 = permute_104 = permute_105 = where_10 = None
        getitem_132: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_150: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_106, [4, 2048, -1]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_151: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_150, [8192, 768]);  view_150 = None
        permute_107: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_63: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg172_1, view_151, permute_107);  arg172_1 = view_151 = permute_107 = None
        view_152: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [4, 2048, 768]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_66: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_140, view_152);  view_140 = view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_153: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_66, [-1, 768]);  add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_236: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_153, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_236, [1], correction = 0, keepdim = True)
        getitem_141: "f32[8192, 1][1, 1]cuda:0" = var_mean_21[0]
        getitem_142: "f32[8192, 1][1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_22: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_236, getitem_142);  convert_element_type_236 = getitem_142 = None
        add_67: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
        rsqrt_21: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_54: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = rsqrt_21 = None
        mul_55: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, arg173_1);  mul_54 = arg173_1 = None
        add_68: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, arg174_1);  mul_55 = arg174_1 = None
        convert_element_type_237: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_108: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg176_1, convert_element_type_237, permute_108);  arg176_1 = convert_element_type_237 = permute_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_10: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_64);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_109: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg178_1, relu_10, permute_109);  arg178_1 = relu_10 = permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_69: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_153, addmm_65);  view_153 = addmm_65 = None
        view_154: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_69, [4, 2048, 768]);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_244: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_154, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_244, [2], correction = 0, keepdim = True)
        getitem_143: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_22[0]
        getitem_144: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        sub_23: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_244, getitem_144);  convert_element_type_244 = getitem_144 = None
        add_70: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_22: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_56: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = rsqrt_22 = None
        mul_57: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, arg179_1);  mul_56 = arg179_1 = None
        add_71: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, arg180_1);  mul_57 = arg180_1 = None
        convert_element_type_245: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_155: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_245, [8192, 768])
        permute_110: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg182_1, view_155, permute_110);  arg182_1 = view_155 = permute_110 = None
        view_156: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [4, 2048, 768]);  addmm_66 = None
        mul_58: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_156, 0.125);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_157: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_58, [4, -1, 12, 64]);  mul_58 = None
        permute_111: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_158: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_245, [8192, 768])
        permute_112: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg184_1, view_158, permute_112);  arg184_1 = view_158 = permute_112 = None
        view_159: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [4, 2048, 768]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_162: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_159, [4, -1, 12, 64]);  view_159 = None
        permute_114: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_160: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_245, [8192, 768]);  convert_element_type_245 = None
        permute_113: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg186_1, view_160, permute_113);  arg186_1 = view_160 = permute_113 = None
        view_161: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [4, 2048, 768]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_163: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [4, -1, 12, 64]);  view_161 = None
        permute_115: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_25: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "bf16[4, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_25, full_default_24);  expand = full_default_25 = full_default_24 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_111, permute_114, permute_115, where_11, False, scale = 1.0);  permute_111 = permute_114 = permute_115 = where_11 = None
        getitem_145: "bf16[4, 12, 2048, 64][1572864, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_116: "bf16[4, 2048, 12, 64][1572864, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_164: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_116, [4, 2048, -1]);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_165: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_164, [8192, 768]);  view_164 = None
        permute_117: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg188_1, view_165, permute_117);  arg188_1 = view_165 = permute_117 = None
        view_166: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [4, 2048, 768]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_72: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_154, view_166);  view_154 = view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_167: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_72, [-1, 768]);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_258: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_167, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_258, [1], correction = 0, keepdim = True)
        getitem_154: "f32[8192, 1][1, 1]cuda:0" = var_mean_23[0]
        getitem_155: "f32[8192, 1][1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_24: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_258, getitem_155);  convert_element_type_258 = getitem_155 = None
        add_73: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_23: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_59: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = rsqrt_23 = None
        mul_60: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, arg189_1);  mul_59 = arg189_1 = None
        add_74: "f32[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, arg190_1);  mul_60 = arg190_1 = None
        convert_element_type_259: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_118: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg192_1, convert_element_type_259, permute_118);  arg192_1 = convert_element_type_259 = permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_11: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.relu.default(addmm_70);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_119: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg194_1, relu_11, permute_119);  arg194_1 = relu_11 = permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_75: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_167, addmm_71);  view_167 = addmm_71 = None
        view_168: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_75, [4, 2048, 768]);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_266: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_168, torch.float32);  view_168 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_266, [2], correction = 0, keepdim = True)
        getitem_156: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_24[0]
        getitem_157: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[4, 2049][2049, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(arg197_1, [0, 1], -100.0);  arg197_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[4, 2048][2049, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_24: "i64[4, 2048][2048, 1]cuda:0" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_172: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [-1]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        ne_1: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_172, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_25: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_266, getitem_157);  convert_element_type_266 = getitem_157 = None
        add_76: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_24: "f32[4, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_61: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = rsqrt_24 = None
        mul_62: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, arg195_1);  mul_61 = arg195_1 = None
        add_77: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_62, arg196_1);  mul_62 = arg196_1 = None
        convert_element_type_267: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:512 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :]).contiguous()
        view_169: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_267, [8192, 768]);  convert_element_type_267 = None
        permute_120: "bf16[768, 50272][1, 768]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm: "bf16[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.mm.default(view_169, permute_120);  view_169 = permute_120 = None
        view_170: "bf16[4, 2048, 50272][102957056, 50272, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [4, 2048, 50272]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_270: "f32[4, 2048, 50272][102957056, 50272, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_170, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_171: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_270, [-1, 50272]);  convert_element_type_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(view_171, [1], True)
        sub_26: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_171, amax);  view_171 = amax = None
        exp: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.exp.default(sub_26)
        sum_1: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        ne: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_172, -100)
        full_default_26: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[8192][1]cuda:0" = torch.ops.aten.where.self(ne, view_172, full_default_26);  ne = full_default_26 = None
        unsqueeze_9: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze_9);  sub_27 = unsqueeze_9 = None
        squeeze: "f32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_27: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f32[8192][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_27);  ne_1 = neg = full_default_27 = None
        sum_3: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_13);  where_13 = None
        ne_2: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_172, -100);  view_172 = None
        sum_2: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_271: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        div: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_271);  sum_3 = convert_element_type_271 = None
        return (div, view_170)
