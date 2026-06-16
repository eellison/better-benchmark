class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[128, 128][128, 1]cuda:0", arg1_1: "bf16[50265, 1024][1024, 1]cuda:0", arg2_1: "bf16[1024, 1024][1024, 1]cuda:0", arg3_1: "bf16[1024][1]cuda:0", arg4_1: "bf16[1024][1]cuda:0", arg5_1: "bf16[1024, 1024][1024, 1]cuda:0", arg6_1: "bf16[1024][1]cuda:0", arg7_1: "bf16[1024, 1024][1024, 1]cuda:0", arg8_1: "bf16[1024][1]cuda:0", arg9_1: "bf16[1024, 1024][1024, 1]cuda:0", arg10_1: "bf16[1024][1]cuda:0", arg11_1: "bf16[1024, 1024][1024, 1]cuda:0", arg12_1: "bf16[1024][1]cuda:0", arg13_1: "bf16[1024][1]cuda:0", arg14_1: "bf16[1024][1]cuda:0", arg15_1: "bf16[4096, 1024][1024, 1]cuda:0", arg16_1: "bf16[4096][1]cuda:0", arg17_1: "bf16[1024, 4096][4096, 1]cuda:0", arg18_1: "bf16[1024][1]cuda:0", arg19_1: "bf16[1024][1]cuda:0", arg20_1: "bf16[1024][1]cuda:0", arg21_1: "bf16[1024, 1024][1024, 1]cuda:0", arg22_1: "bf16[1024][1]cuda:0", arg23_1: "bf16[1024, 1024][1024, 1]cuda:0", arg24_1: "bf16[1024][1]cuda:0", arg25_1: "bf16[1024, 1024][1024, 1]cuda:0", arg26_1: "bf16[1024][1]cuda:0", arg27_1: "bf16[1024, 1024][1024, 1]cuda:0", arg28_1: "bf16[1024][1]cuda:0", arg29_1: "bf16[1024][1]cuda:0", arg30_1: "bf16[1024][1]cuda:0", arg31_1: "bf16[4096, 1024][1024, 1]cuda:0", arg32_1: "bf16[4096][1]cuda:0", arg33_1: "bf16[1024, 4096][4096, 1]cuda:0", arg34_1: "bf16[1024][1]cuda:0", arg35_1: "bf16[1024][1]cuda:0", arg36_1: "bf16[1024][1]cuda:0", arg37_1: "bf16[1024, 1024][1024, 1]cuda:0", arg38_1: "bf16[1024][1]cuda:0", arg39_1: "bf16[1024, 1024][1024, 1]cuda:0", arg40_1: "bf16[1024][1]cuda:0", arg41_1: "bf16[1024, 1024][1024, 1]cuda:0", arg42_1: "bf16[1024][1]cuda:0", arg43_1: "bf16[1024, 1024][1024, 1]cuda:0", arg44_1: "bf16[1024][1]cuda:0", arg45_1: "bf16[1024][1]cuda:0", arg46_1: "bf16[1024][1]cuda:0", arg47_1: "bf16[4096, 1024][1024, 1]cuda:0", arg48_1: "bf16[4096][1]cuda:0", arg49_1: "bf16[1024, 4096][4096, 1]cuda:0", arg50_1: "bf16[1024][1]cuda:0", arg51_1: "bf16[1024][1]cuda:0", arg52_1: "bf16[1024][1]cuda:0", arg53_1: "bf16[1024, 1024][1024, 1]cuda:0", arg54_1: "bf16[1024][1]cuda:0", arg55_1: "bf16[1024, 1024][1024, 1]cuda:0", arg56_1: "bf16[1024][1]cuda:0", arg57_1: "bf16[1024, 1024][1024, 1]cuda:0", arg58_1: "bf16[1024][1]cuda:0", arg59_1: "bf16[1024, 1024][1024, 1]cuda:0", arg60_1: "bf16[1024][1]cuda:0", arg61_1: "bf16[1024][1]cuda:0", arg62_1: "bf16[1024][1]cuda:0", arg63_1: "bf16[4096, 1024][1024, 1]cuda:0", arg64_1: "bf16[4096][1]cuda:0", arg65_1: "bf16[1024, 4096][4096, 1]cuda:0", arg66_1: "bf16[1024][1]cuda:0", arg67_1: "bf16[1024][1]cuda:0", arg68_1: "bf16[1024][1]cuda:0", arg69_1: "bf16[1024, 1024][1024, 1]cuda:0", arg70_1: "bf16[1024][1]cuda:0", arg71_1: "bf16[1024, 1024][1024, 1]cuda:0", arg72_1: "bf16[1024][1]cuda:0", arg73_1: "bf16[1024, 1024][1024, 1]cuda:0", arg74_1: "bf16[1024][1]cuda:0", arg75_1: "bf16[1024, 1024][1024, 1]cuda:0", arg76_1: "bf16[1024][1]cuda:0", arg77_1: "bf16[1024][1]cuda:0", arg78_1: "bf16[1024][1]cuda:0", arg79_1: "bf16[4096, 1024][1024, 1]cuda:0", arg80_1: "bf16[4096][1]cuda:0", arg81_1: "bf16[1024, 4096][4096, 1]cuda:0", arg82_1: "bf16[1024][1]cuda:0", arg83_1: "bf16[1024][1]cuda:0", arg84_1: "bf16[1024][1]cuda:0", arg85_1: "bf16[1024, 1024][1024, 1]cuda:0", arg86_1: "bf16[1024][1]cuda:0", arg87_1: "bf16[1024, 1024][1024, 1]cuda:0", arg88_1: "bf16[1024][1]cuda:0", arg89_1: "bf16[1024, 1024][1024, 1]cuda:0", arg90_1: "bf16[1024][1]cuda:0", arg91_1: "bf16[1024, 1024][1024, 1]cuda:0", arg92_1: "bf16[1024][1]cuda:0", arg93_1: "bf16[1024][1]cuda:0", arg94_1: "bf16[1024][1]cuda:0", arg95_1: "bf16[4096, 1024][1024, 1]cuda:0", arg96_1: "bf16[4096][1]cuda:0", arg97_1: "bf16[1024, 4096][4096, 1]cuda:0", arg98_1: "bf16[1024][1]cuda:0", arg99_1: "bf16[1024][1]cuda:0", arg100_1: "bf16[1024][1]cuda:0", arg101_1: "bf16[1024, 1024][1024, 1]cuda:0", arg102_1: "bf16[1024][1]cuda:0", arg103_1: "bf16[1024, 1024][1024, 1]cuda:0", arg104_1: "bf16[1024][1]cuda:0", arg105_1: "bf16[1024, 1024][1024, 1]cuda:0", arg106_1: "bf16[1024][1]cuda:0", arg107_1: "bf16[1024, 1024][1024, 1]cuda:0", arg108_1: "bf16[1024][1]cuda:0", arg109_1: "bf16[1024][1]cuda:0", arg110_1: "bf16[1024][1]cuda:0", arg111_1: "bf16[4096, 1024][1024, 1]cuda:0", arg112_1: "bf16[4096][1]cuda:0", arg113_1: "bf16[1024, 4096][4096, 1]cuda:0", arg114_1: "bf16[1024][1]cuda:0", arg115_1: "bf16[1024][1]cuda:0", arg116_1: "bf16[1024][1]cuda:0", arg117_1: "bf16[1024, 1024][1024, 1]cuda:0", arg118_1: "bf16[1024][1]cuda:0", arg119_1: "bf16[1024, 1024][1024, 1]cuda:0", arg120_1: "bf16[1024][1]cuda:0", arg121_1: "bf16[1024, 1024][1024, 1]cuda:0", arg122_1: "bf16[1024][1]cuda:0", arg123_1: "bf16[1024, 1024][1024, 1]cuda:0", arg124_1: "bf16[1024][1]cuda:0", arg125_1: "bf16[1024][1]cuda:0", arg126_1: "bf16[1024][1]cuda:0", arg127_1: "bf16[4096, 1024][1024, 1]cuda:0", arg128_1: "bf16[4096][1]cuda:0", arg129_1: "bf16[1024, 4096][4096, 1]cuda:0", arg130_1: "bf16[1024][1]cuda:0", arg131_1: "bf16[1024][1]cuda:0", arg132_1: "bf16[1024][1]cuda:0", arg133_1: "bf16[1024, 1024][1024, 1]cuda:0", arg134_1: "bf16[1024][1]cuda:0", arg135_1: "bf16[1024, 1024][1024, 1]cuda:0", arg136_1: "bf16[1024][1]cuda:0", arg137_1: "bf16[1024, 1024][1024, 1]cuda:0", arg138_1: "bf16[1024][1]cuda:0", arg139_1: "bf16[1024, 1024][1024, 1]cuda:0", arg140_1: "bf16[1024][1]cuda:0", arg141_1: "bf16[1024][1]cuda:0", arg142_1: "bf16[1024][1]cuda:0", arg143_1: "bf16[4096, 1024][1024, 1]cuda:0", arg144_1: "bf16[4096][1]cuda:0", arg145_1: "bf16[1024, 4096][4096, 1]cuda:0", arg146_1: "bf16[1024][1]cuda:0", arg147_1: "bf16[1024][1]cuda:0", arg148_1: "bf16[1024][1]cuda:0", arg149_1: "bf16[1024, 1024][1024, 1]cuda:0", arg150_1: "bf16[1024][1]cuda:0", arg151_1: "bf16[1024, 1024][1024, 1]cuda:0", arg152_1: "bf16[1024][1]cuda:0", arg153_1: "bf16[1024, 1024][1024, 1]cuda:0", arg154_1: "bf16[1024][1]cuda:0", arg155_1: "bf16[1024, 1024][1024, 1]cuda:0", arg156_1: "bf16[1024][1]cuda:0", arg157_1: "bf16[1024][1]cuda:0", arg158_1: "bf16[1024][1]cuda:0", arg159_1: "bf16[4096, 1024][1024, 1]cuda:0", arg160_1: "bf16[4096][1]cuda:0", arg161_1: "bf16[1024, 4096][4096, 1]cuda:0", arg162_1: "bf16[1024][1]cuda:0", arg163_1: "bf16[1024][1]cuda:0", arg164_1: "bf16[1024][1]cuda:0", arg165_1: "bf16[1024, 1024][1024, 1]cuda:0", arg166_1: "bf16[1024][1]cuda:0", arg167_1: "bf16[1024, 1024][1024, 1]cuda:0", arg168_1: "bf16[1024][1]cuda:0", arg169_1: "bf16[1024, 1024][1024, 1]cuda:0", arg170_1: "bf16[1024][1]cuda:0", arg171_1: "bf16[1024, 1024][1024, 1]cuda:0", arg172_1: "bf16[1024][1]cuda:0", arg173_1: "bf16[1024][1]cuda:0", arg174_1: "bf16[1024][1]cuda:0", arg175_1: "bf16[4096, 1024][1024, 1]cuda:0", arg176_1: "bf16[4096][1]cuda:0", arg177_1: "bf16[1024, 4096][4096, 1]cuda:0", arg178_1: "bf16[1024][1]cuda:0", arg179_1: "bf16[1024][1]cuda:0", arg180_1: "bf16[1024][1]cuda:0", arg181_1: "bf16[1024, 1024][1024, 1]cuda:0", arg182_1: "bf16[1024][1]cuda:0", arg183_1: "bf16[1024, 1024][1024, 1]cuda:0", arg184_1: "bf16[1024][1]cuda:0", arg185_1: "bf16[1024, 1024][1024, 1]cuda:0", arg186_1: "bf16[1024][1]cuda:0", arg187_1: "bf16[1024, 1024][1024, 1]cuda:0", arg188_1: "bf16[1024][1]cuda:0", arg189_1: "bf16[1024][1]cuda:0", arg190_1: "bf16[1024][1]cuda:0", arg191_1: "bf16[4096, 1024][1024, 1]cuda:0", arg192_1: "bf16[4096][1]cuda:0", arg193_1: "bf16[1024, 4096][4096, 1]cuda:0", arg194_1: "bf16[1024][1]cuda:0", arg195_1: "bf16[1024][1]cuda:0", arg196_1: "bf16[1024][1]cuda:0", arg197_1: "i64[128, 128][128, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:631 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:634 in forward, code: inputs_embeds = inputs_embeds * self.embed_scale
        mul: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:646 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:105 in forward, code: return super().forward(position_ids)
        embedding_1: "bf16[128, 1024][1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, add);  arg2_1 = add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:674 in forward, code: hidden_states = inputs_embeds + positions
        add_5: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_6: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_1: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_7: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        convert_element_type_1: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [16384, 1024])
        permute: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg6_1, view, permute);  arg6_1 = view = permute = None
        view_1: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [128, 128, 1024]);  addmm = None
        view_2: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [128, 128, -1, 64]);  view_1 = None
        permute_1: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_3: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [16384, 1024])
        permute_2: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg8_1, view_3, permute_2);  arg8_1 = view_3 = permute_2 = None
        view_4: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [128, 128, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [128, 128, -1, 64]);  view_4 = None
        permute_4: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_5: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [16384, 1024]);  convert_element_type_1 = None
        permute_3: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg10_1, view_5, permute_3);  arg10_1 = view_5 = permute_3 = None
        view_6: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 128, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [128, 128, -1, 64]);  view_6 = None
        permute_5: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_3: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_4: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_1: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_2);  unsqueeze_5 = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[128, 1, 128, 128][0, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(le, [128, -1, 128, 128]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_1, full_default);  full_default_1 = full_default = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_1, permute_4, permute_5, where, False, scale = 0.125);  permute_1 = permute_4 = permute_5 = where = None
        getitem_2: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [128, 128, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [16384, 1024]);  view_9 = None
        permute_7: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg12_1, view_10, permute_7);  arg12_1 = view_10 = permute_7 = None
        view_11: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [128, 128, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_8: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_5, view_11);  add_5 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_14: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_14, [2], correction = 0, keepdim = True)
        getitem_11: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_1[0]
        getitem_12: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_12);  convert_element_type_14 = getitem_12 = None
        add_9: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_3: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_4: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, arg13_1);  mul_3 = arg13_1 = None
        add_10: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, arg14_1);  mul_4 = arg14_1 = None
        convert_element_type_15: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_12: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_15, [16384, 1024]);  convert_element_type_15 = None
        permute_8: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg16_1, view_12, permute_8);  arg16_1 = view_12 = permute_8 = None
        view_13: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [128, 128, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_19: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_5: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_19, 0.5)
        mul_6: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_19, 0.7071067811865476);  convert_element_type_19 = None
        erf: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_11: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, add_11);  mul_5 = add_11 = None
        convert_element_type_20: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_14: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_20, [16384, 4096]);  convert_element_type_20 = None
        permute_9: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg18_1, view_14, permute_9);  arg18_1 = view_14 = permute_9 = None
        view_15: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [128, 128, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_12: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_8, view_15);  add_8 = view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_24: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_13: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_2[0]
        getitem_14: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_14);  convert_element_type_24 = getitem_14 = None
        add_13: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_8: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, arg19_1);  mul_8 = arg19_1 = None
        add_14: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, arg20_1);  mul_9 = arg20_1 = None
        convert_element_type_25: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_16: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [16384, 1024])
        permute_10: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg22_1, view_16, permute_10);  arg22_1 = view_16 = permute_10 = None
        view_17: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [128, 128, 1024]);  addmm_6 = None
        view_18: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_17, [128, 128, -1, 64]);  view_17 = None
        permute_11: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_19: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [16384, 1024])
        permute_12: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg24_1, view_19, permute_12);  arg24_1 = view_19 = permute_12 = None
        view_20: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [128, 128, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_23: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_20, [128, 128, -1, 64]);  view_20 = None
        permute_14: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_21: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [16384, 1024]);  convert_element_type_25 = None
        permute_13: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg26_1, view_21, permute_13);  arg26_1 = view_21 = permute_13 = None
        view_22: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [128, 128, 1024]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_24: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_22, [128, 128, -1, 64]);  view_22 = None
        permute_15: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_3, full_default_2);  full_default_3 = full_default_2 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_11, permute_14, permute_15, where_1, False, scale = 0.125);  permute_11 = permute_14 = permute_15 = where_1 = None
        getitem_15: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_16: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_25: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_16, [128, 128, -1]);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_26: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_25, [16384, 1024]);  view_25 = None
        permute_17: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg28_1, view_26, permute_17);  arg28_1 = view_26 = permute_17 = None
        view_27: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [128, 128, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_15: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_12, view_27);  add_12 = view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_38: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_38, [2], correction = 0, keepdim = True)
        getitem_24: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_3[0]
        getitem_25: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_38, getitem_25);  convert_element_type_38 = getitem_25 = None
        add_16: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_10: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, arg29_1);  mul_10 = arg29_1 = None
        add_17: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, arg30_1);  mul_11 = arg30_1 = None
        convert_element_type_39: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_28: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_39, [16384, 1024]);  convert_element_type_39 = None
        permute_18: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg32_1, view_28, permute_18);  arg32_1 = view_28 = permute_18 = None
        view_29: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [128, 128, 4096]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_43: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        mul_12: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.5)
        mul_13: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.7071067811865476);  convert_element_type_43 = None
        erf_1: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_18: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, add_18);  mul_12 = add_18 = None
        convert_element_type_44: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_30: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_44, [16384, 4096]);  convert_element_type_44 = None
        permute_19: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg34_1, view_30, permute_19);  arg34_1 = view_30 = permute_19 = None
        view_31: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [128, 128, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_19: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_15, view_31);  add_15 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_48: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem_26: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_4[0]
        getitem_27: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_27);  convert_element_type_48 = getitem_27 = None
        add_20: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_15: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_16: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, arg35_1);  mul_15 = arg35_1 = None
        add_21: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, arg36_1);  mul_16 = arg36_1 = None
        convert_element_type_49: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_32: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [16384, 1024])
        permute_20: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg38_1, view_32, permute_20);  arg38_1 = view_32 = permute_20 = None
        view_33: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [128, 128, 1024]);  addmm_12 = None
        view_34: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_33, [128, 128, -1, 64]);  view_33 = None
        permute_21: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_35: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [16384, 1024])
        permute_22: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg40_1, view_35, permute_22);  arg40_1 = view_35 = permute_22 = None
        view_36: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [128, 128, 1024]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_39: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_36, [128, 128, -1, 64]);  view_36 = None
        permute_24: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_37: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_49, [16384, 1024]);  convert_element_type_49 = None
        permute_23: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg42_1, view_37, permute_23);  arg42_1 = view_37 = permute_23 = None
        view_38: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [128, 128, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_40: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_38, [128, 128, -1, 64]);  view_38 = None
        permute_25: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_5: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_5, full_default_4);  full_default_5 = full_default_4 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_21, permute_24, permute_25, where_2, False, scale = 0.125);  permute_21 = permute_24 = permute_25 = where_2 = None
        getitem_28: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_26: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_41: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_26, [128, 128, -1]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_42: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_41, [16384, 1024]);  view_41 = None
        permute_27: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg44_1, view_42, permute_27);  arg44_1 = view_42 = permute_27 = None
        view_43: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [128, 128, 1024]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_22: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_19, view_43);  add_19 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_62: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_62, [2], correction = 0, keepdim = True)
        getitem_37: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_5[0]
        getitem_38: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, getitem_38);  convert_element_type_62 = getitem_38 = None
        add_23: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_17: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_18: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, arg45_1);  mul_17 = arg45_1 = None
        add_24: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, arg46_1);  mul_18 = arg46_1 = None
        convert_element_type_63: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_44: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_63, [16384, 1024]);  convert_element_type_63 = None
        permute_28: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg48_1, view_44, permute_28);  arg48_1 = view_44 = permute_28 = None
        view_45: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [128, 128, 4096]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_67: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_19: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_20: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476);  convert_element_type_67 = None
        erf_2: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_25: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, add_25);  mul_19 = add_25 = None
        convert_element_type_68: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_46: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_68, [16384, 4096]);  convert_element_type_68 = None
        permute_29: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg50_1, view_46, permute_29);  arg50_1 = view_46 = permute_29 = None
        view_47: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [128, 128, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_26: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_22, view_47);  add_22 = view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_72: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_39: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_6[0]
        getitem_40: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_40);  convert_element_type_72 = getitem_40 = None
        add_27: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_22: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_23: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, arg51_1);  mul_22 = arg51_1 = None
        add_28: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, arg52_1);  mul_23 = arg52_1 = None
        convert_element_type_73: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_48: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [16384, 1024])
        permute_30: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg54_1, view_48, permute_30);  arg54_1 = view_48 = permute_30 = None
        view_49: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [128, 128, 1024]);  addmm_18 = None
        view_50: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_49, [128, 128, -1, 64]);  view_49 = None
        permute_31: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_51: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [16384, 1024])
        permute_32: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg56_1, view_51, permute_32);  arg56_1 = view_51 = permute_32 = None
        view_52: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [128, 128, 1024]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_55: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [128, 128, -1, 64]);  view_52 = None
        permute_34: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_53: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [16384, 1024]);  convert_element_type_73 = None
        permute_33: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg58_1, view_53, permute_33);  arg58_1 = view_53 = permute_33 = None
        view_54: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [128, 128, 1024]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_56: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [128, 128, -1, 64]);  view_54 = None
        permute_35: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_7: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_31, permute_34, permute_35, where_3, False, scale = 0.125);  permute_31 = permute_34 = permute_35 = where_3 = None
        getitem_41: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_36: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_41, [0, 2, 1, 3]);  getitem_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_57: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_36, [128, 128, -1]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_58: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_57, [16384, 1024]);  view_57 = None
        permute_37: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg60_1, view_58, permute_37);  arg60_1 = view_58 = permute_37 = None
        view_59: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [128, 128, 1024]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_29: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_26, view_59);  add_26 = view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_86: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_86, [2], correction = 0, keepdim = True)
        getitem_50: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_7[0]
        getitem_51: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, getitem_51);  convert_element_type_86 = getitem_51 = None
        add_30: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_24: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_25: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, arg61_1);  mul_24 = arg61_1 = None
        add_31: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, arg62_1);  mul_25 = arg62_1 = None
        convert_element_type_87: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_60: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_87, [16384, 1024]);  convert_element_type_87 = None
        permute_38: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg64_1, view_60, permute_38);  arg64_1 = view_60 = permute_38 = None
        view_61: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [128, 128, 4096]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_91: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        mul_26: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.5)
        mul_27: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.7071067811865476);  convert_element_type_91 = None
        erf_3: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_32: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
        convert_element_type_92: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_28, torch.bfloat16);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_62: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_92, [16384, 4096]);  convert_element_type_92 = None
        permute_39: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg66_1, view_62, permute_39);  arg66_1 = view_62 = permute_39 = None
        view_63: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [128, 128, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_33: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_29, view_63);  add_29 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_96: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_52: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_8[0]
        getitem_53: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_8: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_53);  convert_element_type_96 = getitem_53 = None
        add_34: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_29: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_30: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, arg67_1);  mul_29 = arg67_1 = None
        add_35: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_30, arg68_1);  mul_30 = arg68_1 = None
        convert_element_type_97: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_64: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [16384, 1024])
        permute_40: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg70_1, view_64, permute_40);  arg70_1 = view_64 = permute_40 = None
        view_65: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [128, 128, 1024]);  addmm_24 = None
        view_66: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_65, [128, 128, -1, 64]);  view_65 = None
        permute_41: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_67: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [16384, 1024])
        permute_42: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg72_1, view_67, permute_42);  arg72_1 = view_67 = permute_42 = None
        view_68: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [128, 128, 1024]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_71: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_68, [128, 128, -1, 64]);  view_68 = None
        permute_44: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_69: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [16384, 1024]);  convert_element_type_97 = None
        permute_43: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg74_1, view_69, permute_43);  arg74_1 = view_69 = permute_43 = None
        view_70: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [128, 128, 1024]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_72: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_70, [128, 128, -1, 64]);  view_70 = None
        permute_45: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_9, full_default_8);  full_default_9 = full_default_8 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_41, permute_44, permute_45, where_4, False, scale = 0.125);  permute_41 = permute_44 = permute_45 = where_4 = None
        getitem_54: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_46: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_73: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_46, [128, 128, -1]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_74: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [16384, 1024]);  view_73 = None
        permute_47: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg76_1, view_74, permute_47);  arg76_1 = view_74 = permute_47 = None
        view_75: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [128, 128, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_36: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_33, view_75);  add_33 = view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_110: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True)
        getitem_63: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_9[0]
        getitem_64: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_64);  convert_element_type_110 = getitem_64 = None
        add_37: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
        rsqrt_9: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_31: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_32: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, arg77_1);  mul_31 = arg77_1 = None
        add_38: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, arg78_1);  mul_32 = arg78_1 = None
        convert_element_type_111: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_76: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_111, [16384, 1024]);  convert_element_type_111 = None
        permute_48: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg80_1, view_76, permute_48);  arg80_1 = view_76 = permute_48 = None
        view_77: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [128, 128, 4096]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_115: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_33: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.5)
        mul_34: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.7071067811865476);  convert_element_type_115 = None
        erf_4: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_39: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, add_39);  mul_33 = add_39 = None
        convert_element_type_116: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_78: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [16384, 4096]);  convert_element_type_116 = None
        permute_49: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg82_1, view_78, permute_49);  arg82_1 = view_78 = permute_49 = None
        view_79: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [128, 128, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_40: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_36, view_79);  add_36 = view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_120: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_65: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_10[0]
        getitem_66: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_10: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_66);  convert_element_type_120 = getitem_66 = None
        add_41: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
        rsqrt_10: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        mul_36: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_37: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg83_1);  mul_36 = arg83_1 = None
        add_42: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg84_1);  mul_37 = arg84_1 = None
        convert_element_type_121: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_42, torch.bfloat16);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_80: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [16384, 1024])
        permute_50: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg86_1, view_80, permute_50);  arg86_1 = view_80 = permute_50 = None
        view_81: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [128, 128, 1024]);  addmm_30 = None
        view_82: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [128, 128, -1, 64]);  view_81 = None
        permute_51: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_83: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [16384, 1024])
        permute_52: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg88_1, view_83, permute_52);  arg88_1 = view_83 = permute_52 = None
        view_84: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [128, 128, 1024]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_87: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_84, [128, 128, -1, 64]);  view_84 = None
        permute_54: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_85: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [16384, 1024]);  convert_element_type_121 = None
        permute_53: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg90_1, view_85, permute_53);  arg90_1 = view_85 = permute_53 = None
        view_86: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [128, 128, 1024]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_88: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_86, [128, 128, -1, 64]);  view_86 = None
        permute_55: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_11: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_11, full_default_10);  full_default_11 = full_default_10 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_51, permute_54, permute_55, where_5, False, scale = 0.125);  permute_51 = permute_54 = permute_55 = where_5 = None
        getitem_67: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_67, [0, 2, 1, 3]);  getitem_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_89: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_56, [128, 128, -1]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_90: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [16384, 1024]);  view_89 = None
        permute_57: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg92_1, view_90, permute_57);  arg92_1 = view_90 = permute_57 = None
        view_91: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [128, 128, 1024]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_43: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_40, view_91);  add_40 = view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_134: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_134, [2], correction = 0, keepdim = True)
        getitem_76: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_11[0]
        getitem_77: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_11: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_77);  convert_element_type_134 = getitem_77 = None
        add_44: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_11: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_38: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_39: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, arg93_1);  mul_38 = arg93_1 = None
        add_45: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, arg94_1);  mul_39 = arg94_1 = None
        convert_element_type_135: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_92: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [16384, 1024]);  convert_element_type_135 = None
        permute_58: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg96_1, view_92, permute_58);  arg96_1 = view_92 = permute_58 = None
        view_93: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [128, 128, 4096]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_139: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_40: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_41: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476);  convert_element_type_139 = None
        erf_5: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_46: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, add_46);  mul_40 = add_46 = None
        convert_element_type_140: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_94: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_140, [16384, 4096]);  convert_element_type_140 = None
        permute_59: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg98_1, view_94, permute_59);  arg98_1 = view_94 = permute_59 = None
        view_95: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [128, 128, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_47: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_43, view_95);  add_43 = view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_144: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_78: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_12[0]
        getitem_79: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_12: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_79);  convert_element_type_144 = getitem_79 = None
        add_48: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_12: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_43: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_44: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, arg99_1);  mul_43 = arg99_1 = None
        add_49: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, arg100_1);  mul_44 = arg100_1 = None
        convert_element_type_145: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_96: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [16384, 1024])
        permute_60: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg102_1, view_96, permute_60);  arg102_1 = view_96 = permute_60 = None
        view_97: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [128, 128, 1024]);  addmm_36 = None
        view_98: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_97, [128, 128, -1, 64]);  view_97 = None
        permute_61: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_99: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [16384, 1024])
        permute_62: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_37: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg104_1, view_99, permute_62);  arg104_1 = view_99 = permute_62 = None
        view_100: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [128, 128, 1024]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_103: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_100, [128, 128, -1, 64]);  view_100 = None
        permute_64: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_101: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_145, [16384, 1024]);  convert_element_type_145 = None
        permute_63: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_38: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg106_1, view_101, permute_63);  arg106_1 = view_101 = permute_63 = None
        view_102: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [128, 128, 1024]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_104: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_102, [128, 128, -1, 64]);  view_102 = None
        permute_65: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_13: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_61, permute_64, permute_65, where_6, False, scale = 0.125);  permute_61 = permute_64 = permute_65 = where_6 = None
        getitem_80: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_66: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_80, [0, 2, 1, 3]);  getitem_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_66, [128, 128, -1]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_106: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_105, [16384, 1024]);  view_105 = None
        permute_67: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_39: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg108_1, view_106, permute_67);  arg108_1 = view_106 = permute_67 = None
        view_107: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [128, 128, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_50: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_47, view_107);  add_47 = view_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_158: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_158, [2], correction = 0, keepdim = True)
        getitem_89: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_13[0]
        getitem_90: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_13: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_158, getitem_90);  convert_element_type_158 = getitem_90 = None
        add_51: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_89, 1e-05);  getitem_89 = None
        rsqrt_13: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_45: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_46: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, arg109_1);  mul_45 = arg109_1 = None
        add_52: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, arg110_1);  mul_46 = arg110_1 = None
        convert_element_type_159: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_108: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_159, [16384, 1024]);  convert_element_type_159 = None
        permute_68: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg112_1, view_108, permute_68);  arg112_1 = view_108 = permute_68 = None
        view_109: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [128, 128, 4096]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_163: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None
        mul_47: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_163, 0.5)
        mul_48: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_163, 0.7071067811865476);  convert_element_type_163 = None
        erf_6: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_53: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_49: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, add_53);  mul_47 = add_53 = None
        convert_element_type_164: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_110: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_164, [16384, 4096]);  convert_element_type_164 = None
        permute_69: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_41: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg114_1, view_110, permute_69);  arg114_1 = view_110 = permute_69 = None
        view_111: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [128, 128, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_54: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_50, view_111);  add_50 = view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_168: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_168, [2], correction = 0, keepdim = True)
        getitem_91: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_14[0]
        getitem_92: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_14: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_168, getitem_92);  convert_element_type_168 = getitem_92 = None
        add_55: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
        rsqrt_14: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_50: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_51: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, arg115_1);  mul_50 = arg115_1 = None
        add_56: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, arg116_1);  mul_51 = arg116_1 = None
        convert_element_type_169: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_112: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_169, [16384, 1024])
        permute_70: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg118_1, view_112, permute_70);  arg118_1 = view_112 = permute_70 = None
        view_113: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [128, 128, 1024]);  addmm_42 = None
        view_114: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [128, 128, -1, 64]);  view_113 = None
        permute_71: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_115: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_169, [16384, 1024])
        permute_72: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_43: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg120_1, view_115, permute_72);  arg120_1 = view_115 = permute_72 = None
        view_116: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [128, 128, 1024]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_119: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_116, [128, 128, -1, 64]);  view_116 = None
        permute_74: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_117: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_169, [16384, 1024]);  convert_element_type_169 = None
        permute_73: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg122_1, view_117, permute_73);  arg122_1 = view_117 = permute_73 = None
        view_118: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [128, 128, 1024]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_120: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_118, [128, 128, -1, 64]);  view_118 = None
        permute_75: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_15: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_15, full_default_14);  full_default_15 = full_default_14 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_71, permute_74, permute_75, where_7, False, scale = 0.125);  permute_71 = permute_74 = permute_75 = where_7 = None
        getitem_93: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_76: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_121: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_76, [128, 128, -1]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_122: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_121, [16384, 1024]);  view_121 = None
        permute_77: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_45: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg124_1, view_122, permute_77);  arg124_1 = view_122 = permute_77 = None
        view_123: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [128, 128, 1024]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_57: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_54, view_123);  add_54 = view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_182: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_182, [2], correction = 0, keepdim = True)
        getitem_102: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_15[0]
        getitem_103: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_182, getitem_103);  convert_element_type_182 = getitem_103 = None
        add_58: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_15: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_52: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_53: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, arg125_1);  mul_52 = arg125_1 = None
        add_59: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, arg126_1);  mul_53 = arg126_1 = None
        convert_element_type_183: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_124: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_183, [16384, 1024]);  convert_element_type_183 = None
        permute_78: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg128_1, view_124, permute_78);  arg128_1 = view_124 = permute_78 = None
        view_125: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [128, 128, 4096]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_187: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None
        mul_54: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.5)
        mul_55: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.7071067811865476);  convert_element_type_187 = None
        erf_7: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_60: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_56: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, add_60);  mul_54 = add_60 = None
        convert_element_type_188: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_126: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_188, [16384, 4096]);  convert_element_type_188 = None
        permute_79: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_47: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg130_1, view_126, permute_79);  arg130_1 = view_126 = permute_79 = None
        view_127: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [128, 128, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_61: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_57, view_127);  add_57 = view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_192: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_192, [2], correction = 0, keepdim = True)
        getitem_104: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_16[0]
        getitem_105: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_16: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_192, getitem_105);  convert_element_type_192 = getitem_105 = None
        add_62: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_16: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_57: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_58: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg131_1);  mul_57 = arg131_1 = None
        add_63: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg132_1);  mul_58 = arg132_1 = None
        convert_element_type_193: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_128: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_193, [16384, 1024])
        permute_80: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg134_1, view_128, permute_80);  arg134_1 = view_128 = permute_80 = None
        view_129: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [128, 128, 1024]);  addmm_48 = None
        view_130: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_129, [128, 128, -1, 64]);  view_129 = None
        permute_81: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_131: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_193, [16384, 1024])
        permute_82: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg136_1, view_131, permute_82);  arg136_1 = view_131 = permute_82 = None
        view_132: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [128, 128, 1024]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_135: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_132, [128, 128, -1, 64]);  view_132 = None
        permute_84: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_133: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_193, [16384, 1024]);  convert_element_type_193 = None
        permute_83: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_50: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg138_1, view_133, permute_83);  arg138_1 = view_133 = permute_83 = None
        view_134: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [128, 128, 1024]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_136: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_134, [128, 128, -1, 64]);  view_134 = None
        permute_85: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_17: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_17, full_default_16);  full_default_17 = full_default_16 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_81, permute_84, permute_85, where_8, False, scale = 0.125);  permute_81 = permute_84 = permute_85 = where_8 = None
        getitem_106: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_86: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_137: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_86, [128, 128, -1]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_138: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [16384, 1024]);  view_137 = None
        permute_87: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_51: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg140_1, view_138, permute_87);  arg140_1 = view_138 = permute_87 = None
        view_139: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [128, 128, 1024]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_64: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_61, view_139);  add_61 = view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_206: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_206, [2], correction = 0, keepdim = True)
        getitem_115: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_17[0]
        getitem_116: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_17: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_206, getitem_116);  convert_element_type_206 = getitem_116 = None
        add_65: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
        rsqrt_17: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_59: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_60: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, arg141_1);  mul_59 = arg141_1 = None
        add_66: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, arg142_1);  mul_60 = arg142_1 = None
        convert_element_type_207: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_140: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_207, [16384, 1024]);  convert_element_type_207 = None
        permute_88: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_52: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg144_1, view_140, permute_88);  arg144_1 = view_140 = permute_88 = None
        view_141: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [128, 128, 4096]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_211: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_61: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.5)
        mul_62: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.7071067811865476);  convert_element_type_211 = None
        erf_8: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_67: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_63: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, add_67);  mul_61 = add_67 = None
        convert_element_type_212: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_142: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_212, [16384, 4096]);  convert_element_type_212 = None
        permute_89: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_53: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg146_1, view_142, permute_89);  arg146_1 = view_142 = permute_89 = None
        view_143: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [128, 128, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_68: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_64, view_143);  add_64 = view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_216: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_216, [2], correction = 0, keepdim = True)
        getitem_117: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_18[0]
        getitem_118: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_216, getitem_118);  convert_element_type_216 = getitem_118 = None
        add_69: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_117, 1e-05);  getitem_117 = None
        rsqrt_18: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_64: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_65: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, arg147_1);  mul_64 = arg147_1 = None
        add_70: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, arg148_1);  mul_65 = arg148_1 = None
        convert_element_type_217: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_144: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_217, [16384, 1024])
        permute_90: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg150_1, view_144, permute_90);  arg150_1 = view_144 = permute_90 = None
        view_145: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [128, 128, 1024]);  addmm_54 = None
        view_146: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_145, [128, 128, -1, 64]);  view_145 = None
        permute_91: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_147: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_217, [16384, 1024])
        permute_92: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg152_1, view_147, permute_92);  arg152_1 = view_147 = permute_92 = None
        view_148: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [128, 128, 1024]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_151: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_148, [128, 128, -1, 64]);  view_148 = None
        permute_94: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_149: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_217, [16384, 1024]);  convert_element_type_217 = None
        permute_93: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_56: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg154_1, view_149, permute_93);  arg154_1 = view_149 = permute_93 = None
        view_150: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [128, 128, 1024]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_152: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_150, [128, 128, -1, 64]);  view_150 = None
        permute_95: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_19: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_91, permute_94, permute_95, where_9, False, scale = 0.125);  permute_91 = permute_94 = permute_95 = where_9 = None
        getitem_119: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_96: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_153: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_96, [128, 128, -1]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_154: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_153, [16384, 1024]);  view_153 = None
        permute_97: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_57: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg156_1, view_154, permute_97);  arg156_1 = view_154 = permute_97 = None
        view_155: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [128, 128, 1024]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_71: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_68, view_155);  add_68 = view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_230: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_230, [2], correction = 0, keepdim = True)
        getitem_128: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_19[0]
        getitem_129: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_19: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_230, getitem_129);  convert_element_type_230 = getitem_129 = None
        add_72: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_19: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_66: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_67: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, arg157_1);  mul_66 = arg157_1 = None
        add_73: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, arg158_1);  mul_67 = arg158_1 = None
        convert_element_type_231: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_156: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_231, [16384, 1024]);  convert_element_type_231 = None
        permute_98: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg160_1, view_156, permute_98);  arg160_1 = view_156 = permute_98 = None
        view_157: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [128, 128, 4096]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_235: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_157, torch.float32);  view_157 = None
        mul_68: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.5)
        mul_69: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.7071067811865476);  convert_element_type_235 = None
        erf_9: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_74: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_70: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_74);  mul_68 = add_74 = None
        convert_element_type_236: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_158: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_236, [16384, 4096]);  convert_element_type_236 = None
        permute_99: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg162_1, view_158, permute_99);  arg162_1 = view_158 = permute_99 = None
        view_159: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [128, 128, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_75: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_71, view_159);  add_71 = view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_240: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_240, [2], correction = 0, keepdim = True)
        getitem_130: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_20[0]
        getitem_131: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_20: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_240, getitem_131);  convert_element_type_240 = getitem_131 = None
        add_76: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_20: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_71: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_72: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, arg163_1);  mul_71 = arg163_1 = None
        add_77: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_72, arg164_1);  mul_72 = arg164_1 = None
        convert_element_type_241: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_160: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [16384, 1024])
        permute_100: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg166_1, view_160, permute_100);  arg166_1 = view_160 = permute_100 = None
        view_161: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [128, 128, 1024]);  addmm_60 = None
        view_162: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [128, 128, -1, 64]);  view_161 = None
        permute_101: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_163: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [16384, 1024])
        permute_102: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg168_1, view_163, permute_102);  arg168_1 = view_163 = permute_102 = None
        view_164: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [128, 128, 1024]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_167: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_164, [128, 128, -1, 64]);  view_164 = None
        permute_104: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_165: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [16384, 1024]);  convert_element_type_241 = None
        permute_103: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_62: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg170_1, view_165, permute_103);  arg170_1 = view_165 = permute_103 = None
        view_166: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [128, 128, 1024]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_168: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_166, [128, 128, -1, 64]);  view_166 = None
        permute_105: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_21: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_21, full_default_20);  full_default_21 = full_default_20 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_101, permute_104, permute_105, where_10, False, scale = 0.125);  permute_101 = permute_104 = permute_105 = where_10 = None
        getitem_132: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_106, [128, 128, -1]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_170: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [16384, 1024]);  view_169 = None
        permute_107: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_63: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg172_1, view_170, permute_107);  arg172_1 = view_170 = permute_107 = None
        view_171: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [128, 128, 1024]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_78: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_75, view_171);  add_75 = view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_254: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_78, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_254, [2], correction = 0, keepdim = True)
        getitem_141: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_21[0]
        getitem_142: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_21: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_254, getitem_142);  convert_element_type_254 = getitem_142 = None
        add_79: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
        rsqrt_21: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_73: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_74: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, arg173_1);  mul_73 = arg173_1 = None
        add_80: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, arg174_1);  mul_74 = arg174_1 = None
        convert_element_type_255: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_172: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_255, [16384, 1024]);  convert_element_type_255 = None
        permute_108: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg176_1, view_172, permute_108);  arg176_1 = view_172 = permute_108 = None
        view_173: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [128, 128, 4096]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_259: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_75: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_259, 0.5)
        mul_76: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_259, 0.7071067811865476);  convert_element_type_259 = None
        erf_10: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_81: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_77: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, add_81);  mul_75 = add_81 = None
        convert_element_type_260: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_174: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_260, [16384, 4096]);  convert_element_type_260 = None
        permute_109: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg178_1, view_174, permute_109);  arg178_1 = view_174 = permute_109 = None
        view_175: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [128, 128, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_82: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_78, view_175);  add_78 = view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:367 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_264: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_82, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_264, [2], correction = 0, keepdim = True)
        getitem_143: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_22[0]
        getitem_144: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        sub_22: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_264, getitem_144);  convert_element_type_264 = getitem_144 = None
        add_83: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_22: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_78: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_79: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, arg179_1);  mul_78 = arg179_1 = None
        add_84: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, arg180_1);  mul_79 = arg180_1 = None
        convert_element_type_265: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_176: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_265, [16384, 1024])
        permute_110: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg182_1, view_176, permute_110);  arg182_1 = view_176 = permute_110 = None
        view_177: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [128, 128, 1024]);  addmm_66 = None
        view_178: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_177, [128, 128, -1, 64]);  view_177 = None
        permute_111: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: key_states = self.k_proj(current_states)
        view_179: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_265, [16384, 1024])
        permute_112: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg184_1, view_179, permute_112);  arg184_1 = view_179 = permute_112 = None
        view_180: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [128, 128, 1024]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:225 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_183: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_180, [128, 128, -1, 64]);  view_180 = None
        permute_114: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:223 in forward, code: value_states = self.v_proj(current_states)
        view_181: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_265, [16384, 1024]);  convert_element_type_265 = None
        permute_113: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg186_1, view_181, permute_113);  arg186_1 = view_181 = permute_113 = None
        view_182: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [128, 128, 1024]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:226 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_184: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_182, [128, 128, -1, 64]);  view_182 = None
        permute_115: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_23: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "bf16[128, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_23, full_default_22);  expand = full_default_23 = full_default_22 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_111, permute_114, permute_115, where_11, False, scale = 0.125);  permute_111 = permute_114 = permute_115 = where_11 = None
        getitem_145: "bf16[128, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_116: "bf16[128, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:249 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_185: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_116, [128, 128, -1]);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        view_186: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_185, [16384, 1024]);  view_185 = None
        permute_117: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg188_1, view_186, permute_117);  arg188_1 = view_186 = permute_117 = None
        view_187: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [128, 128, 1024]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_85: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_82, view_187);  add_82 = view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_278: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_278, [2], correction = 0, keepdim = True)
        getitem_154: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_23[0]
        getitem_155: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_23: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_278, getitem_155);  convert_element_type_278 = getitem_155 = None
        add_86: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_23: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_80: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_81: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, arg189_1);  mul_80 = arg189_1 = None
        add_87: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, arg190_1);  mul_81 = arg190_1 = None
        convert_element_type_279: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_188: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_279, [16384, 1024]);  convert_element_type_279 = None
        permute_118: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg192_1, view_188, permute_118);  arg192_1 = view_188 = permute_118 = None
        view_189: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [128, 128, 4096]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_283: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_189, torch.float32);  view_189 = None
        mul_82: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_283, 0.5)
        mul_83: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_283, 0.7071067811865476);  convert_element_type_283 = None
        erf_11: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_88: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_84: "f32[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, add_88);  mul_82 = add_88 = None
        convert_element_type_284: "bf16[128, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:399 in forward, code: hidden_states = self.fc2(hidden_states)
        view_190: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_284, [16384, 4096]);  convert_element_type_284 = None
        permute_119: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg194_1, view_190, permute_119);  arg194_1 = view_190 = permute_119 = None
        view_191: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [128, 128, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = residual + hidden_states
        add_89: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_85, view_191);  add_85 = view_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:694 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_288: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.float32);  add_89 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_288, [2], correction = 0, keepdim = True)
        getitem_156: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_24[0]
        getitem_157: "f32[128, 128, 1][128, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:1120 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_195: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(arg197_1, [-1]);  arg197_1 = None
        ne_1: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_195, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:694 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_24: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_288, getitem_157);  convert_element_type_288 = getitem_157 = None
        add_90: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_24: "f32[128, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_85: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_86: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, arg195_1);  mul_85 = arg195_1 = None
        add_91: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_86, arg196_1);  mul_86 = arg196_1 = None
        convert_element_type_289: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:1114 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_192: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_289, [16384, 1024]);  convert_element_type_289 = None
        permute_120: "bf16[1024, 50265][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "bf16[1024, 50272][50272, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_120, [0, 7, 0, 0]);  permute_120 = None
        mm_default: "bf16[16384, 50272][50272, 1]cuda:0" = torch.ops.aten.mm.default(view_192, constant_pad_nd_default);  view_192 = constant_pad_nd_default = None
        slice_tensor: "bf16[16384, 50265][50272, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -7);  mm_default = None
        view_193: "bf16[128, 128, 50265][6434816, 50272, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [128, 128, 50265]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:1120 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_194: "bf16[16384, 50265][50272, 1]cuda:0" = torch.ops.aten.reshape.default(view_193, [-1, 50265])
        convert_element_type_292: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_194, torch.float32);  view_194 = None
        amax: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_292, [1], True)
        sub_25: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_292, amax);  convert_element_type_292 = amax = None
        exp: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.exp.default(sub_25)
        sum_1: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_26: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        convert_element_type_293: "bf16[16384, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_26, torch.bfloat16);  sub_26 = None
        ne: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_195, -100)
        full_default_24: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[16384][1]cuda:0" = torch.ops.aten.where.self(ne, view_195, full_default_24);  ne = full_default_24 = None
        unsqueeze_9: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather: "bf16[16384, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_293, 1, unsqueeze_9);  convert_element_type_293 = unsqueeze_9 = None
        squeeze: "bf16[16384][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[16384][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_25: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "bf16[16384][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_25);  ne_1 = neg = full_default_25 = None
        sum_3: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_13);  where_13 = None
        ne_2: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_195, -100);  view_195 = None
        sum_2: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_294: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.bfloat16);  sum_2 = None
        div: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_294);  sum_3 = convert_element_type_294 = None
        return (div, view_193)
