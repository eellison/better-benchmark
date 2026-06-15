class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[64, 256][256, 1]cuda:0", arg1_1: "bf16[50265, 1024][1024, 1]cuda:0", arg2_1: "bf16[514, 1024][1024, 1]cuda:0", arg3_1: "bf16[1024][1]cuda:0", arg4_1: "bf16[1024][1]cuda:0", arg5_1: "bf16[1024, 1024][1024, 1]cuda:0", arg6_1: "bf16[1024][1]cuda:0", arg7_1: "bf16[1024, 1024][1024, 1]cuda:0", arg8_1: "bf16[1024][1]cuda:0", arg9_1: "bf16[1024, 1024][1024, 1]cuda:0", arg10_1: "bf16[1024][1]cuda:0", arg11_1: "bf16[1024, 1024][1024, 1]cuda:0", arg12_1: "bf16[1024][1]cuda:0", arg13_1: "bf16[1024][1]cuda:0", arg14_1: "bf16[1024][1]cuda:0", arg15_1: "bf16[4096, 1024][1024, 1]cuda:0", arg16_1: "bf16[4096][1]cuda:0", arg17_1: "bf16[1024, 4096][4096, 1]cuda:0", arg18_1: "bf16[1024][1]cuda:0", arg19_1: "bf16[1024][1]cuda:0", arg20_1: "bf16[1024][1]cuda:0", arg21_1: "bf16[1024, 1024][1024, 1]cuda:0", arg22_1: "bf16[1024][1]cuda:0", arg23_1: "bf16[1024, 1024][1024, 1]cuda:0", arg24_1: "bf16[1024][1]cuda:0", arg25_1: "bf16[1024, 1024][1024, 1]cuda:0", arg26_1: "bf16[1024][1]cuda:0", arg27_1: "bf16[1024, 1024][1024, 1]cuda:0", arg28_1: "bf16[1024][1]cuda:0", arg29_1: "bf16[1024][1]cuda:0", arg30_1: "bf16[1024][1]cuda:0", arg31_1: "bf16[4096, 1024][1024, 1]cuda:0", arg32_1: "bf16[4096][1]cuda:0", arg33_1: "bf16[1024, 4096][4096, 1]cuda:0", arg34_1: "bf16[1024][1]cuda:0", arg35_1: "bf16[1024][1]cuda:0", arg36_1: "bf16[1024][1]cuda:0", arg37_1: "bf16[1024, 1024][1024, 1]cuda:0", arg38_1: "bf16[1024][1]cuda:0", arg39_1: "bf16[1024, 1024][1024, 1]cuda:0", arg40_1: "bf16[1024][1]cuda:0", arg41_1: "bf16[1024, 1024][1024, 1]cuda:0", arg42_1: "bf16[1024][1]cuda:0", arg43_1: "bf16[1024, 1024][1024, 1]cuda:0", arg44_1: "bf16[1024][1]cuda:0", arg45_1: "bf16[1024][1]cuda:0", arg46_1: "bf16[1024][1]cuda:0", arg47_1: "bf16[4096, 1024][1024, 1]cuda:0", arg48_1: "bf16[4096][1]cuda:0", arg49_1: "bf16[1024, 4096][4096, 1]cuda:0", arg50_1: "bf16[1024][1]cuda:0", arg51_1: "bf16[1024][1]cuda:0", arg52_1: "bf16[1024][1]cuda:0", arg53_1: "bf16[1024, 1024][1024, 1]cuda:0", arg54_1: "bf16[1024][1]cuda:0", arg55_1: "bf16[1024, 1024][1024, 1]cuda:0", arg56_1: "bf16[1024][1]cuda:0", arg57_1: "bf16[1024, 1024][1024, 1]cuda:0", arg58_1: "bf16[1024][1]cuda:0", arg59_1: "bf16[1024, 1024][1024, 1]cuda:0", arg60_1: "bf16[1024][1]cuda:0", arg61_1: "bf16[1024][1]cuda:0", arg62_1: "bf16[1024][1]cuda:0", arg63_1: "bf16[4096, 1024][1024, 1]cuda:0", arg64_1: "bf16[4096][1]cuda:0", arg65_1: "bf16[1024, 4096][4096, 1]cuda:0", arg66_1: "bf16[1024][1]cuda:0", arg67_1: "bf16[1024][1]cuda:0", arg68_1: "bf16[1024][1]cuda:0", arg69_1: "bf16[1024, 1024][1024, 1]cuda:0", arg70_1: "bf16[1024][1]cuda:0", arg71_1: "bf16[1024, 1024][1024, 1]cuda:0", arg72_1: "bf16[1024][1]cuda:0", arg73_1: "bf16[1024, 1024][1024, 1]cuda:0", arg74_1: "bf16[1024][1]cuda:0", arg75_1: "bf16[1024, 1024][1024, 1]cuda:0", arg76_1: "bf16[1024][1]cuda:0", arg77_1: "bf16[1024][1]cuda:0", arg78_1: "bf16[1024][1]cuda:0", arg79_1: "bf16[4096, 1024][1024, 1]cuda:0", arg80_1: "bf16[4096][1]cuda:0", arg81_1: "bf16[1024, 4096][4096, 1]cuda:0", arg82_1: "bf16[1024][1]cuda:0", arg83_1: "bf16[1024][1]cuda:0", arg84_1: "bf16[1024][1]cuda:0", arg85_1: "bf16[1024, 1024][1024, 1]cuda:0", arg86_1: "bf16[1024][1]cuda:0", arg87_1: "bf16[1024, 1024][1024, 1]cuda:0", arg88_1: "bf16[1024][1]cuda:0", arg89_1: "bf16[1024, 1024][1024, 1]cuda:0", arg90_1: "bf16[1024][1]cuda:0", arg91_1: "bf16[1024, 1024][1024, 1]cuda:0", arg92_1: "bf16[1024][1]cuda:0", arg93_1: "bf16[1024][1]cuda:0", arg94_1: "bf16[1024][1]cuda:0", arg95_1: "bf16[4096, 1024][1024, 1]cuda:0", arg96_1: "bf16[4096][1]cuda:0", arg97_1: "bf16[1024, 4096][4096, 1]cuda:0", arg98_1: "bf16[1024][1]cuda:0", arg99_1: "bf16[1024][1]cuda:0", arg100_1: "bf16[1024][1]cuda:0", arg101_1: "bf16[1024, 1024][1024, 1]cuda:0", arg102_1: "bf16[1024][1]cuda:0", arg103_1: "bf16[1024, 1024][1024, 1]cuda:0", arg104_1: "bf16[1024][1]cuda:0", arg105_1: "bf16[1024, 1024][1024, 1]cuda:0", arg106_1: "bf16[1024][1]cuda:0", arg107_1: "bf16[1024, 1024][1024, 1]cuda:0", arg108_1: "bf16[1024][1]cuda:0", arg109_1: "bf16[1024][1]cuda:0", arg110_1: "bf16[1024][1]cuda:0", arg111_1: "bf16[4096, 1024][1024, 1]cuda:0", arg112_1: "bf16[4096][1]cuda:0", arg113_1: "bf16[1024, 4096][4096, 1]cuda:0", arg114_1: "bf16[1024][1]cuda:0", arg115_1: "bf16[1024][1]cuda:0", arg116_1: "bf16[1024][1]cuda:0", arg117_1: "bf16[1024, 1024][1024, 1]cuda:0", arg118_1: "bf16[1024][1]cuda:0", arg119_1: "bf16[1024, 1024][1024, 1]cuda:0", arg120_1: "bf16[1024][1]cuda:0", arg121_1: "bf16[1024, 1024][1024, 1]cuda:0", arg122_1: "bf16[1024][1]cuda:0", arg123_1: "bf16[1024, 1024][1024, 1]cuda:0", arg124_1: "bf16[1024][1]cuda:0", arg125_1: "bf16[1024][1]cuda:0", arg126_1: "bf16[1024][1]cuda:0", arg127_1: "bf16[4096, 1024][1024, 1]cuda:0", arg128_1: "bf16[4096][1]cuda:0", arg129_1: "bf16[1024, 4096][4096, 1]cuda:0", arg130_1: "bf16[1024][1]cuda:0", arg131_1: "bf16[1024][1]cuda:0", arg132_1: "bf16[1024][1]cuda:0", arg133_1: "bf16[1024, 1024][1024, 1]cuda:0", arg134_1: "bf16[1024][1]cuda:0", arg135_1: "bf16[1024, 1024][1024, 1]cuda:0", arg136_1: "bf16[1024][1]cuda:0", arg137_1: "bf16[1024, 1024][1024, 1]cuda:0", arg138_1: "bf16[1024][1]cuda:0", arg139_1: "bf16[1024, 1024][1024, 1]cuda:0", arg140_1: "bf16[1024][1]cuda:0", arg141_1: "bf16[1024][1]cuda:0", arg142_1: "bf16[1024][1]cuda:0", arg143_1: "bf16[4096, 1024][1024, 1]cuda:0", arg144_1: "bf16[4096][1]cuda:0", arg145_1: "bf16[1024, 4096][4096, 1]cuda:0", arg146_1: "bf16[1024][1]cuda:0", arg147_1: "bf16[1024][1]cuda:0", arg148_1: "bf16[1024][1]cuda:0", arg149_1: "bf16[1024, 1024][1024, 1]cuda:0", arg150_1: "bf16[1024][1]cuda:0", arg151_1: "bf16[1024, 1024][1024, 1]cuda:0", arg152_1: "bf16[1024][1]cuda:0", arg153_1: "bf16[1024, 1024][1024, 1]cuda:0", arg154_1: "bf16[1024][1]cuda:0", arg155_1: "bf16[1024, 1024][1024, 1]cuda:0", arg156_1: "bf16[1024][1]cuda:0", arg157_1: "bf16[1024][1]cuda:0", arg158_1: "bf16[1024][1]cuda:0", arg159_1: "bf16[4096, 1024][1024, 1]cuda:0", arg160_1: "bf16[4096][1]cuda:0", arg161_1: "bf16[1024, 4096][4096, 1]cuda:0", arg162_1: "bf16[1024][1]cuda:0", arg163_1: "bf16[1024][1]cuda:0", arg164_1: "bf16[1024][1]cuda:0", arg165_1: "bf16[1024, 1024][1024, 1]cuda:0", arg166_1: "bf16[1024][1]cuda:0", arg167_1: "bf16[1024, 1024][1024, 1]cuda:0", arg168_1: "bf16[1024][1]cuda:0", arg169_1: "bf16[1024, 1024][1024, 1]cuda:0", arg170_1: "bf16[1024][1]cuda:0", arg171_1: "bf16[1024, 1024][1024, 1]cuda:0", arg172_1: "bf16[1024][1]cuda:0", arg173_1: "bf16[1024][1]cuda:0", arg174_1: "bf16[1024][1]cuda:0", arg175_1: "bf16[4096, 1024][1024, 1]cuda:0", arg176_1: "bf16[4096][1]cuda:0", arg177_1: "bf16[1024, 4096][4096, 1]cuda:0", arg178_1: "bf16[1024][1]cuda:0", arg179_1: "bf16[1024][1]cuda:0", arg180_1: "bf16[1024][1]cuda:0", arg181_1: "bf16[1024, 1024][1024, 1]cuda:0", arg182_1: "bf16[1024][1]cuda:0", arg183_1: "bf16[1024, 1024][1024, 1]cuda:0", arg184_1: "bf16[1024][1]cuda:0", arg185_1: "bf16[1024, 1024][1024, 1]cuda:0", arg186_1: "bf16[1024][1]cuda:0", arg187_1: "bf16[1024, 1024][1024, 1]cuda:0", arg188_1: "bf16[1024][1]cuda:0", arg189_1: "bf16[1024][1]cuda:0", arg190_1: "bf16[1024][1]cuda:0", arg191_1: "bf16[4096, 1024][1024, 1]cuda:0", arg192_1: "bf16[4096][1]cuda:0", arg193_1: "bf16[1024, 4096][4096, 1]cuda:0", arg194_1: "bf16[1024][1]cuda:0", arg195_1: "bf16[1024][1]cuda:0", arg196_1: "bf16[1024][1]cuda:0", arg197_1: "i64[64, 256][256, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:75 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg0_1 = None
        mul: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:55 in forward, code: position_ids = torch.arange(
        iota: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:57 in forward, code: ).expand(bsz, -1)
        expand: "i64[64, 256][0, 1]cuda:0" = torch.ops.aten.expand.default(iota, [64, -1]);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:61 in forward, code: return super().forward(position_ids + self.offset)
        add: "i64[64, 256][256, 1]cuda:0" = torch.ops.aten.add.Tensor(expand, 2);  expand = None
        embedding_1: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, add);  arg2_1 = add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:543 in forward, code: hidden_states = inputs_embeds + embed_pos
        add_1: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:546 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        convert_element_type: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        _tensor_constant0: "bf16[][]cuda:0" = self._tensor_constant0;  _tensor_constant0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:546 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        sub: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_2: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_1: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_3: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        convert_element_type_1: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_1: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [16384, 1024])
        permute: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg6_1, view_1, permute);  arg6_1 = view_1 = permute = None
        view_2: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [64, 256, 1024]);  addmm = None
        mul_3: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_2, 0.125);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_9: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_3, [64, 256, 16, 64]);  mul_3 = None
        permute_5: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_1: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_10: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [1024, 256, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_3: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [16384, 1024])
        permute_1: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg8_1, view_3, permute_1);  arg8_1 = view_3 = permute_1 = None
        view_4: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [64, 256, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_7: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [64, -1, 16, 64]);  view_4 = None
        permute_3: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_2: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_11: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1024, 256, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_6: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        bmm: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_10, permute_6);  view_10 = permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_13: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [64, 16, 256, 256]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_5: "i64[256][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_3: "i64[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_5, 0);  add_5 = None
        unsqueeze_4: "i64[1, 1, 256][256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 256][256, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[256][1]cuda:0" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_1: "i64[1, 1, 256][256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 256, 1][256, 256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 256, 256][65536, 65536, 256, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_2);  unsqueeze_5 = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[64, 1, 256, 256][0, 65536, 256, 1]cuda:0" = torch.ops.aten.expand.default(le, [64, -1, 256, 256]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[64, 1, 256, 256][65536, 65536, 256, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default, full_default_1);  expand_1 = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        add_6: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_13, where);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_14: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_6, [1024, 256, 256]);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_13: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        amax: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_13, [-1], True)
        sub_1: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax);  convert_element_type_13 = amax = None
        exp: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_14: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_5: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [16384, 1024])
        permute_2: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg10_1, view_5, permute_2);  arg10_1 = view_5 = permute_2 = None
        view_6: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [64, 256, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_8: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [64, -1, 16, 64]);  view_6 = None
        permute_4: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_3: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_12: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [1024, 256, 64]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_1: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_14, view_12);  convert_element_type_14 = view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_15: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [64, 16, 256, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_7: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_5: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_16: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [64, 256, 1024]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_17: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_16, [16384, 1024]);  view_16 = None
        permute_8: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg12_1, view_17, permute_8);  arg12_1 = view_17 = permute_8 = None
        view_18: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [64, 256, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_7: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1, view_18);  convert_element_type_1 = view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_20: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_20, [2], correction = 0, keepdim = True)
        getitem_2: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_3);  convert_element_type_20 = getitem_3 = None
        add_8: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_4: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
        add_9: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        convert_element_type_21: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_19: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_21, [16384, 1024])
        permute_9: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg16_1, view_19, permute_9);  arg16_1 = view_19 = permute_9 = None
        view_20: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [64, 256, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_25: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_20, torch.float32);  view_20 = None
        mul_6: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, 0.5)
        mul_7: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, 0.7071067811865476);  convert_element_type_25 = None
        erf: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_10: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, add_10);  mul_6 = add_10 = None
        convert_element_type_26: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_21: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_26, [16384, 4096]);  convert_element_type_26 = None
        permute_10: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg18_1, view_21, permute_10);  arg18_1 = view_21 = permute_10 = None
        view_22: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [64, 256, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_11: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_21, view_22);  convert_element_type_21 = view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_30: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.float32);  add_11 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_30, [2], correction = 0, keepdim = True)
        getitem_4: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_30, getitem_5);  convert_element_type_30 = getitem_5 = None
        add_12: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_9: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
        add_13: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
        convert_element_type_31: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_23: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [16384, 1024])
        permute_11: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg22_1, view_23, permute_11);  arg22_1 = view_23 = permute_11 = None
        view_24: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [64, 256, 1024]);  addmm_6 = None
        mul_11: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_24, 0.125);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_31: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_11, [64, 256, 16, 64]);  mul_11 = None
        permute_16: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_9: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_32: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [1024, 256, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_25: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [16384, 1024])
        permute_12: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg24_1, view_25, permute_12);  arg24_1 = view_25 = permute_12 = None
        view_26: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [64, 256, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_29: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [64, -1, 16, 64]);  view_26 = None
        permute_14: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_10: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_33: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1024, 256, 64]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_17: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        bmm_2: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_32, permute_17);  view_32 = permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_35: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [64, 16, 256, 256]);  bmm_2 = None
        add_14: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_35, where);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_36: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_14, [1024, 256, 256]);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_43: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_36, torch.float32);  view_36 = None
        amax_1: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_43, [-1], True)
        sub_4: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_43, amax_1);  convert_element_type_43 = amax_1 = None
        exp_1: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_44: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_27: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [16384, 1024])
        permute_13: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg26_1, view_27, permute_13);  arg26_1 = view_27 = permute_13 = None
        view_28: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [64, 256, 1024]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_30: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_28, [64, -1, 16, 64]);  view_28 = None
        permute_15: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_11: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_34: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [1024, 256, 64]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_3: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_44, view_34);  convert_element_type_44 = view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_37: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [64, 16, 256, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_18: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_13: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_38: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [64, 256, 1024]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_39: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_38, [16384, 1024]);  view_38 = None
        permute_19: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg28_1, view_39, permute_19);  arg28_1 = view_39 = permute_19 = None
        view_40: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [64, 256, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_15: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_31, view_40);  convert_element_type_31 = view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_50: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.float32);  add_15 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_6: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_7);  convert_element_type_50 = getitem_7 = None
        add_16: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_12: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_13: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, arg29_1);  mul_12 = arg29_1 = None
        add_17: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
        convert_element_type_51: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_41: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_51, [16384, 1024])
        permute_20: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg32_1, view_41, permute_20);  arg32_1 = view_41 = permute_20 = None
        view_42: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [64, 256, 4096]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_55: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_42, torch.float32);  view_42 = None
        mul_14: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.5)
        mul_15: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.7071067811865476);  convert_element_type_55 = None
        erf_1: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_18: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, add_18);  mul_14 = add_18 = None
        convert_element_type_56: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_43: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_56, [16384, 4096]);  convert_element_type_56 = None
        permute_21: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg34_1, view_43, permute_21);  arg34_1 = view_43 = permute_21 = None
        view_44: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [64, 256, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_19: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_51, view_44);  convert_element_type_51 = view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_60: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.float32);  add_19 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_60, [2], correction = 0, keepdim = True)
        getitem_8: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_60, getitem_9);  convert_element_type_60 = getitem_9 = None
        add_20: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_17: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_18: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
        add_21: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
        convert_element_type_61: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_45: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [16384, 1024])
        permute_22: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg38_1, view_45, permute_22);  arg38_1 = view_45 = permute_22 = None
        view_46: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [64, 256, 1024]);  addmm_12 = None
        mul_19: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_46, 0.125);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_53: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [64, 256, 16, 64]);  mul_19 = None
        permute_27: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_17: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_54: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [1024, 256, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_47: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [16384, 1024])
        permute_23: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg40_1, view_47, permute_23);  arg40_1 = view_47 = permute_23 = None
        view_48: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [64, 256, 1024]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_51: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_48, [64, -1, 16, 64]);  view_48 = None
        permute_25: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_18: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        view_55: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [1024, 256, 64]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_28: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
        bmm_4: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_54, permute_28);  view_54 = permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_57: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [64, 16, 256, 256]);  bmm_4 = None
        add_22: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_57, where);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_58: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_22, [1024, 256, 256]);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_73: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_58, torch.float32);  view_58 = None
        amax_2: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_73, [-1], True)
        sub_7: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_73, amax_2);  convert_element_type_73 = amax_2 = None
        exp_2: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_74: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_49: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [16384, 1024])
        permute_24: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg42_1, view_49, permute_24);  arg42_1 = view_49 = permute_24 = None
        view_50: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [64, 256, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_52: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_50, [64, -1, 16, 64]);  view_50 = None
        permute_26: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_19: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_56: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [1024, 256, 64]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_5: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_74, view_56);  convert_element_type_74 = view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_59: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [64, 16, 256, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_29: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_21: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_60: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [64, 256, 1024]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_61: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_60, [16384, 1024]);  view_60 = None
        permute_30: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg44_1, view_61, permute_30);  arg44_1 = view_61 = permute_30 = None
        view_62: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [64, 256, 1024]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_23: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_61, view_62);  convert_element_type_61 = view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_80: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True)
        getitem_10: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_11);  convert_element_type_80 = getitem_11 = None
        add_24: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_20: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_21: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, arg45_1);  mul_20 = arg45_1 = None
        add_25: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
        convert_element_type_81: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_63: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_81, [16384, 1024])
        permute_31: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg48_1, view_63, permute_31);  arg48_1 = view_63 = permute_31 = None
        view_64: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [64, 256, 4096]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_85: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_64, torch.float32);  view_64 = None
        mul_22: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.5)
        mul_23: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.7071067811865476);  convert_element_type_85 = None
        erf_2: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_26: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, add_26);  mul_22 = add_26 = None
        convert_element_type_86: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_65: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_86, [16384, 4096]);  convert_element_type_86 = None
        permute_32: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg50_1, view_65, permute_32);  arg50_1 = view_65 = permute_32 = None
        view_66: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [64, 256, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_27: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_81, view_66);  convert_element_type_81 = view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_90: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True)
        getitem_12: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_9: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_90, getitem_13);  convert_element_type_90 = getitem_13 = None
        add_28: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_25: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_26: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, arg51_1);  mul_25 = arg51_1 = None
        add_29: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
        convert_element_type_91: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_67: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [16384, 1024])
        permute_33: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg54_1, view_67, permute_33);  arg54_1 = view_67 = permute_33 = None
        view_68: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [64, 256, 1024]);  addmm_18 = None
        mul_27: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_68, 0.125);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_75: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [64, 256, 16, 64]);  mul_27 = None
        permute_38: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_25: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_76: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [1024, 256, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_69: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [16384, 1024])
        permute_34: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg56_1, view_69, permute_34);  arg56_1 = view_69 = permute_34 = None
        view_70: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [64, 256, 1024]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_73: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_70, [64, -1, 16, 64]);  view_70 = None
        permute_36: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_26: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_77: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [1024, 256, 64]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_39: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
        bmm_6: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_76, permute_39);  view_76 = permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_79: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [64, 16, 256, 256]);  bmm_6 = None
        add_30: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_79, where);  view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_80: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_30, [1024, 256, 256]);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_103: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_80, torch.float32);  view_80 = None
        amax_3: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_103, [-1], True)
        sub_10: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_103, amax_3);  convert_element_type_103 = amax_3 = None
        exp_3: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_104: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_71: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [16384, 1024])
        permute_35: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg58_1, view_71, permute_35);  arg58_1 = view_71 = permute_35 = None
        view_72: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [64, 256, 1024]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_74: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_72, [64, -1, 16, 64]);  view_72 = None
        permute_37: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_27: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_78: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [1024, 256, 64]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_7: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_104, view_78);  convert_element_type_104 = view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_81: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [64, 16, 256, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_40: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_29: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_82: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [64, 256, 1024]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_83: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_82, [16384, 1024]);  view_82 = None
        permute_41: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg60_1, view_83, permute_41);  arg60_1 = view_83 = permute_41 = None
        view_84: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [64, 256, 1024]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_31: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_91, view_84);  convert_element_type_91 = view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_110: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.float32);  add_31 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True)
        getitem_14: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_15);  convert_element_type_110 = getitem_15 = None
        add_32: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_28: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_29: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, arg61_1);  mul_28 = arg61_1 = None
        add_33: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
        convert_element_type_111: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_85: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_111, [16384, 1024])
        permute_42: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg64_1, view_85, permute_42);  arg64_1 = view_85 = permute_42 = None
        view_86: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [64, 256, 4096]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_115: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_86, torch.float32);  view_86 = None
        mul_30: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.5)
        mul_31: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.7071067811865476);  convert_element_type_115 = None
        erf_3: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_34: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, add_34);  mul_30 = add_34 = None
        convert_element_type_116: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_87: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [16384, 4096]);  convert_element_type_116 = None
        permute_43: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg66_1, view_87, permute_43);  arg66_1 = view_87 = permute_43 = None
        view_88: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [64, 256, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_35: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_111, view_88);  convert_element_type_111 = view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_120: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_16: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_12: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_17);  convert_element_type_120 = getitem_17 = None
        add_36: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_33: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_34: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, arg67_1);  mul_33 = arg67_1 = None
        add_37: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
        convert_element_type_121: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_89: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [16384, 1024])
        permute_44: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg70_1, view_89, permute_44);  arg70_1 = view_89 = permute_44 = None
        view_90: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [64, 256, 1024]);  addmm_24 = None
        mul_35: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_90, 0.125);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_97: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_35, [64, 256, 16, 64]);  mul_35 = None
        permute_49: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_33: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_98: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [1024, 256, 64]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_91: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [16384, 1024])
        permute_45: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg72_1, view_91, permute_45);  arg72_1 = view_91 = permute_45 = None
        view_92: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [64, 256, 1024]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_95: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_92, [64, -1, 16, 64]);  view_92 = None
        permute_47: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_34: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_99: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [1024, 256, 64]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_50: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        bmm_8: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_98, permute_50);  view_98 = permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_101: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [64, 16, 256, 256]);  bmm_8 = None
        add_38: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_101, where);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_102: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_38, [1024, 256, 256]);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_133: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_102, torch.float32);  view_102 = None
        amax_4: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_133, [-1], True)
        sub_13: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_133, amax_4);  convert_element_type_133 = amax_4 = None
        exp_4: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_134: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_93: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [16384, 1024])
        permute_46: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg74_1, view_93, permute_46);  arg74_1 = view_93 = permute_46 = None
        view_94: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [64, 256, 1024]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_96: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [64, -1, 16, 64]);  view_94 = None
        permute_48: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_35: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_100: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [1024, 256, 64]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_9: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_134, view_100);  convert_element_type_134 = view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_103: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [64, 16, 256, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_51: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_37: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_104: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [64, 256, 1024]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_105: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_104, [16384, 1024]);  view_104 = None
        permute_52: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg76_1, view_105, permute_52);  arg76_1 = view_105 = permute_52 = None
        view_106: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [64, 256, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_39: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_121, view_106);  convert_element_type_121 = view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_140: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.float32);  add_39 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_140, [2], correction = 0, keepdim = True)
        getitem_18: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_140, getitem_19);  convert_element_type_140 = getitem_19 = None
        add_40: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_36: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_37: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
        add_41: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
        convert_element_type_141: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_107: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_141, [16384, 1024])
        permute_53: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg80_1, view_107, permute_53);  arg80_1 = view_107 = permute_53 = None
        view_108: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [64, 256, 4096]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_145: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        mul_38: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_145, 0.5)
        mul_39: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_145, 0.7071067811865476);  convert_element_type_145 = None
        erf_4: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_42: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, add_42);  mul_38 = add_42 = None
        convert_element_type_146: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_109: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_146, [16384, 4096]);  convert_element_type_146 = None
        permute_54: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg82_1, view_109, permute_54);  arg82_1 = view_109 = permute_54 = None
        view_110: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [64, 256, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_43: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_141, view_110);  convert_element_type_141 = view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_150: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.float32);  add_43 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_150, [2], correction = 0, keepdim = True)
        getitem_20: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_15: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_150, getitem_21);  convert_element_type_150 = getitem_21 = None
        add_44: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_41: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_42: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, arg83_1);  mul_41 = arg83_1 = None
        add_45: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
        convert_element_type_151: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_111: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [16384, 1024])
        permute_55: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg86_1, view_111, permute_55);  arg86_1 = view_111 = permute_55 = None
        view_112: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [64, 256, 1024]);  addmm_30 = None
        mul_43: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_112, 0.125);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_119: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_43, [64, 256, 16, 64]);  mul_43 = None
        permute_60: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_41: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_120: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [1024, 256, 64]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_113: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [16384, 1024])
        permute_56: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg88_1, view_113, permute_56);  arg88_1 = view_113 = permute_56 = None
        view_114: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [64, 256, 1024]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_117: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [64, -1, 16, 64]);  view_114 = None
        permute_58: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_42: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        view_121: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [1024, 256, 64]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_61: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        bmm_10: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_120, permute_61);  view_120 = permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_123: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [64, 16, 256, 256]);  bmm_10 = None
        add_46: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_123, where);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_124: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_46, [1024, 256, 256]);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_163: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_124, torch.float32);  view_124 = None
        amax_5: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_163, [-1], True)
        sub_16: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_163, amax_5);  convert_element_type_163 = amax_5 = None
        exp_5: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_164: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_115: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [16384, 1024])
        permute_57: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg90_1, view_115, permute_57);  arg90_1 = view_115 = permute_57 = None
        view_116: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [64, 256, 1024]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_118: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_116, [64, -1, 16, 64]);  view_116 = None
        permute_59: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_43: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_122: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [1024, 256, 64]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_11: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_164, view_122);  convert_element_type_164 = view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_125: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [64, 16, 256, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_62: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_45: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_126: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [64, 256, 1024]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_127: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_126, [16384, 1024]);  view_126 = None
        permute_63: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg92_1, view_127, permute_63);  arg92_1 = view_127 = permute_63 = None
        view_128: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [64, 256, 1024]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_47: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_151, view_128);  convert_element_type_151 = view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_170: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.float32);  add_47 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_170, [2], correction = 0, keepdim = True)
        getitem_22: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_170, getitem_23);  convert_element_type_170 = getitem_23 = None
        add_48: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_44: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_45: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, arg93_1);  mul_44 = arg93_1 = None
        add_49: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
        convert_element_type_171: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_129: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_171, [16384, 1024])
        permute_64: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg96_1, view_129, permute_64);  arg96_1 = view_129 = permute_64 = None
        view_130: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [64, 256, 4096]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_175: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_130, torch.float32);  view_130 = None
        mul_46: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.5)
        mul_47: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.7071067811865476);  convert_element_type_175 = None
        erf_5: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_50: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, add_50);  mul_46 = add_50 = None
        convert_element_type_176: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_131: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_176, [16384, 4096]);  convert_element_type_176 = None
        permute_65: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg98_1, view_131, permute_65);  arg98_1 = view_131 = permute_65 = None
        view_132: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [64, 256, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_51: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_171, view_132);  convert_element_type_171 = view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_180: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.float32);  add_51 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_180, [2], correction = 0, keepdim = True)
        getitem_24: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_18: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_180, getitem_25);  convert_element_type_180 = getitem_25 = None
        add_52: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_49: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_50: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, arg99_1);  mul_49 = arg99_1 = None
        add_53: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
        convert_element_type_181: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_133: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_181, [16384, 1024])
        permute_66: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg102_1, view_133, permute_66);  arg102_1 = view_133 = permute_66 = None
        view_134: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [64, 256, 1024]);  addmm_36 = None
        mul_51: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_134, 0.125);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_141: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_51, [64, 256, 16, 64]);  mul_51 = None
        permute_71: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_49: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_142: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [1024, 256, 64]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_135: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_181, [16384, 1024])
        permute_67: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_37: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg104_1, view_135, permute_67);  arg104_1 = view_135 = permute_67 = None
        view_136: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [64, 256, 1024]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_139: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_136, [64, -1, 16, 64]);  view_136 = None
        permute_69: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_50: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        view_143: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [1024, 256, 64]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_72: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
        bmm_12: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_142, permute_72);  view_142 = permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_145: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [64, 16, 256, 256]);  bmm_12 = None
        add_54: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_145, where);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_146: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_54, [1024, 256, 256]);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_193: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None
        amax_6: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_193, [-1], True)
        sub_19: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_193, amax_6);  convert_element_type_193 = amax_6 = None
        exp_6: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_194: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_137: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_181, [16384, 1024])
        permute_68: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_38: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg106_1, view_137, permute_68);  arg106_1 = view_137 = permute_68 = None
        view_138: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [64, 256, 1024]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_140: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_138, [64, -1, 16, 64]);  view_138 = None
        permute_70: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_51: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_144: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [1024, 256, 64]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_13: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_194, view_144);  convert_element_type_194 = view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_147: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [64, 16, 256, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_73: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_53: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_148: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [64, 256, 1024]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_149: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_148, [16384, 1024]);  view_148 = None
        permute_74: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_39: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg108_1, view_149, permute_74);  arg108_1 = view_149 = permute_74 = None
        view_150: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [64, 256, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_55: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_181, view_150);  convert_element_type_181 = view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_200: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_200, [2], correction = 0, keepdim = True)
        getitem_26: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_200, getitem_27);  convert_element_type_200 = getitem_27 = None
        add_56: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_52: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_53: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, arg109_1);  mul_52 = arg109_1 = None
        add_57: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, arg110_1);  mul_53 = arg110_1 = None
        convert_element_type_201: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_151: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_201, [16384, 1024])
        permute_75: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg112_1, view_151, permute_75);  arg112_1 = view_151 = permute_75 = None
        view_152: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [64, 256, 4096]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_205: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_152, torch.float32);  view_152 = None
        mul_54: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.5)
        mul_55: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.7071067811865476);  convert_element_type_205 = None
        erf_6: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_58: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, add_58);  mul_54 = add_58 = None
        convert_element_type_206: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_153: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_206, [16384, 4096]);  convert_element_type_206 = None
        permute_76: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_41: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg114_1, view_153, permute_76);  arg114_1 = view_153 = permute_76 = None
        view_154: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [64, 256, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_59: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_201, view_154);  convert_element_type_201 = view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_210: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_210, [2], correction = 0, keepdim = True)
        getitem_28: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_21: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_210, getitem_29);  convert_element_type_210 = getitem_29 = None
        add_60: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_57: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_58: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg115_1);  mul_57 = arg115_1 = None
        add_61: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg116_1);  mul_58 = arg116_1 = None
        convert_element_type_211: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_155: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_211, [16384, 1024])
        permute_77: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg118_1, view_155, permute_77);  arg118_1 = view_155 = permute_77 = None
        view_156: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [64, 256, 1024]);  addmm_42 = None
        mul_59: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_156, 0.125);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_163: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_59, [64, 256, 16, 64]);  mul_59 = None
        permute_82: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_57: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_164: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [1024, 256, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_157: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_211, [16384, 1024])
        permute_78: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_43: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg120_1, view_157, permute_78);  arg120_1 = view_157 = permute_78 = None
        view_158: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [64, 256, 1024]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_161: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_158, [64, -1, 16, 64]);  view_158 = None
        permute_80: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_58: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_165: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [1024, 256, 64]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_83: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
        bmm_14: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_164, permute_83);  view_164 = permute_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_167: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [64, 16, 256, 256]);  bmm_14 = None
        add_62: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_167, where);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_168: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_62, [1024, 256, 256]);  add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_223: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_168, torch.float32);  view_168 = None
        amax_7: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_223, [-1], True)
        sub_22: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_223, amax_7);  convert_element_type_223 = amax_7 = None
        exp_7: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_224: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_159: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_211, [16384, 1024])
        permute_79: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg122_1, view_159, permute_79);  arg122_1 = view_159 = permute_79 = None
        view_160: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [64, 256, 1024]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_162: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_160, [64, -1, 16, 64]);  view_160 = None
        permute_81: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_59: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_166: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [1024, 256, 64]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_15: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_224, view_166);  convert_element_type_224 = view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_169: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [64, 16, 256, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_84: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_61: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_170: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [64, 256, 1024]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_171: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_170, [16384, 1024]);  view_170 = None
        permute_85: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_45: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg124_1, view_171, permute_85);  arg124_1 = view_171 = permute_85 = None
        view_172: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [64, 256, 1024]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_63: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_211, view_172);  convert_element_type_211 = view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_230: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_230, [2], correction = 0, keepdim = True)
        getitem_30: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_230, getitem_31);  convert_element_type_230 = getitem_31 = None
        add_64: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_60: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_61: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, arg125_1);  mul_60 = arg125_1 = None
        add_65: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, arg126_1);  mul_61 = arg126_1 = None
        convert_element_type_231: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_173: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_231, [16384, 1024])
        permute_86: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg128_1, view_173, permute_86);  arg128_1 = view_173 = permute_86 = None
        view_174: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [64, 256, 4096]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_235: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_174, torch.float32);  view_174 = None
        mul_62: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.5)
        mul_63: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.7071067811865476);  convert_element_type_235 = None
        erf_7: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_66: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, add_66);  mul_62 = add_66 = None
        convert_element_type_236: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_175: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_236, [16384, 4096]);  convert_element_type_236 = None
        permute_87: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_47: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg130_1, view_175, permute_87);  arg130_1 = view_175 = permute_87 = None
        view_176: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [64, 256, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_67: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_231, view_176);  convert_element_type_231 = view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_240: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.float32);  add_67 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_240, [2], correction = 0, keepdim = True)
        getitem_32: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_24: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_240, getitem_33);  convert_element_type_240 = getitem_33 = None
        add_68: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_65: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_66: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, arg131_1);  mul_65 = arg131_1 = None
        add_69: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
        convert_element_type_241: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_177: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [16384, 1024])
        permute_88: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg134_1, view_177, permute_88);  arg134_1 = view_177 = permute_88 = None
        view_178: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [64, 256, 1024]);  addmm_48 = None
        mul_67: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_178, 0.125);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_185: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_67, [64, 256, 16, 64]);  mul_67 = None
        permute_93: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_65: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_186: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [1024, 256, 64]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_179: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [16384, 1024])
        permute_89: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg136_1, view_179, permute_89);  arg136_1 = view_179 = permute_89 = None
        view_180: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [64, 256, 1024]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_183: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_180, [64, -1, 16, 64]);  view_180 = None
        permute_91: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_66: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
        view_187: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [1024, 256, 64]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_94: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
        bmm_16: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_186, permute_94);  view_186 = permute_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_189: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [64, 16, 256, 256]);  bmm_16 = None
        add_70: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_189, where);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_190: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_70, [1024, 256, 256]);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_253: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_190, torch.float32);  view_190 = None
        amax_8: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_253, [-1], True)
        sub_25: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_253, amax_8);  convert_element_type_253 = amax_8 = None
        exp_8: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_254: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_181: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [16384, 1024])
        permute_90: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_50: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg138_1, view_181, permute_90);  arg138_1 = view_181 = permute_90 = None
        view_182: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [64, 256, 1024]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_184: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_182, [64, -1, 16, 64]);  view_182 = None
        permute_92: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_67: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_188: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [1024, 256, 64]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_17: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_254, view_188);  convert_element_type_254 = view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_191: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [64, 16, 256, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_95: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_69: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_192: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [64, 256, 1024]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_193: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_192, [16384, 1024]);  view_192 = None
        permute_96: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_51: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg140_1, view_193, permute_96);  arg140_1 = view_193 = permute_96 = None
        view_194: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [64, 256, 1024]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_71: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_241, view_194);  convert_element_type_241 = view_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_260: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_260, [2], correction = 0, keepdim = True)
        getitem_34: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_260, getitem_35);  convert_element_type_260 = getitem_35 = None
        add_72: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_68: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_69: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, arg141_1);  mul_68 = arg141_1 = None
        add_73: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, arg142_1);  mul_69 = arg142_1 = None
        convert_element_type_261: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_195: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_261, [16384, 1024])
        permute_97: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_52: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg144_1, view_195, permute_97);  arg144_1 = view_195 = permute_97 = None
        view_196: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [64, 256, 4096]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_265: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_196, torch.float32);  view_196 = None
        mul_70: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, 0.5)
        mul_71: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, 0.7071067811865476);  convert_element_type_265 = None
        erf_8: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_74: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_72: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, add_74);  mul_70 = add_74 = None
        convert_element_type_266: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_197: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_266, [16384, 4096]);  convert_element_type_266 = None
        permute_98: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_53: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg146_1, view_197, permute_98);  arg146_1 = view_197 = permute_98 = None
        view_198: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [64, 256, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_75: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_261, view_198);  convert_element_type_261 = view_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_270: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.float32);  add_75 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_270, [2], correction = 0, keepdim = True)
        getitem_36: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_27: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_270, getitem_37);  convert_element_type_270 = getitem_37 = None
        add_76: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_73: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_74: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, arg147_1);  mul_73 = arg147_1 = None
        add_77: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, arg148_1);  mul_74 = arg148_1 = None
        convert_element_type_271: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_199: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_271, [16384, 1024])
        permute_99: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg150_1, view_199, permute_99);  arg150_1 = view_199 = permute_99 = None
        view_200: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [64, 256, 1024]);  addmm_54 = None
        mul_75: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_200, 0.125);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_207: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [64, 256, 16, 64]);  mul_75 = None
        permute_104: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_73: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_208: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [1024, 256, 64]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_201: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_271, [16384, 1024])
        permute_100: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg152_1, view_201, permute_100);  arg152_1 = view_201 = permute_100 = None
        view_202: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [64, 256, 1024]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_205: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_202, [64, -1, 16, 64]);  view_202 = None
        permute_102: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_74: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_209: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [1024, 256, 64]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_105: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        bmm_18: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_208, permute_105);  view_208 = permute_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_211: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [64, 16, 256, 256]);  bmm_18 = None
        add_78: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_211, where);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_212: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_78, [1024, 256, 256]);  add_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_283: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        amax_9: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_283, [-1], True)
        sub_28: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_283, amax_9);  convert_element_type_283 = amax_9 = None
        exp_9: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_284: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_203: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_271, [16384, 1024])
        permute_101: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_56: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg154_1, view_203, permute_101);  arg154_1 = view_203 = permute_101 = None
        view_204: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [64, 256, 1024]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_206: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_204, [64, -1, 16, 64]);  view_204 = None
        permute_103: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_75: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        view_210: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [1024, 256, 64]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_19: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_284, view_210);  convert_element_type_284 = view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_213: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [64, 16, 256, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_106: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_77: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_214: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [64, 256, 1024]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_215: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_214, [16384, 1024]);  view_214 = None
        permute_107: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_57: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg156_1, view_215, permute_107);  arg156_1 = view_215 = permute_107 = None
        view_216: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [64, 256, 1024]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_79: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_271, view_216);  convert_element_type_271 = view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_290: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.float32);  add_79 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_290, [2], correction = 0, keepdim = True)
        getitem_38: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_290, getitem_39);  convert_element_type_290 = getitem_39 = None
        add_80: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_76: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_77: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, arg157_1);  mul_76 = arg157_1 = None
        add_81: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_77, arg158_1);  mul_77 = arg158_1 = None
        convert_element_type_291: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_217: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_291, [16384, 1024])
        permute_108: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg160_1, view_217, permute_108);  arg160_1 = view_217 = permute_108 = None
        view_218: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [64, 256, 4096]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_295: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_218, torch.float32);  view_218 = None
        mul_78: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_295, 0.5)
        mul_79: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_295, 0.7071067811865476);  convert_element_type_295 = None
        erf_9: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_82: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_80: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, add_82);  mul_78 = add_82 = None
        convert_element_type_296: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_219: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_296, [16384, 4096]);  convert_element_type_296 = None
        permute_109: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg162_1, view_219, permute_109);  arg162_1 = view_219 = permute_109 = None
        view_220: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [64, 256, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_83: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_291, view_220);  convert_element_type_291 = view_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_300: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.float32);  add_83 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_300, [2], correction = 0, keepdim = True)
        getitem_40: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_30: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_300, getitem_41);  convert_element_type_300 = getitem_41 = None
        add_84: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_81: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_82: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, arg163_1);  mul_81 = arg163_1 = None
        add_85: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, arg164_1);  mul_82 = arg164_1 = None
        convert_element_type_301: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_221: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_301, [16384, 1024])
        permute_110: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg166_1, view_221, permute_110);  arg166_1 = view_221 = permute_110 = None
        view_222: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [64, 256, 1024]);  addmm_60 = None
        mul_83: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_222, 0.125);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_229: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_83, [64, 256, 16, 64]);  mul_83 = None
        permute_115: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_81: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_230: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [1024, 256, 64]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_223: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_301, [16384, 1024])
        permute_111: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg168_1, view_223, permute_111);  arg168_1 = view_223 = permute_111 = None
        view_224: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [64, 256, 1024]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_227: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_224, [64, -1, 16, 64]);  view_224 = None
        permute_113: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_82: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_231: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [1024, 256, 64]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_116: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        bmm_20: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_230, permute_116);  view_230 = permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_233: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [64, 16, 256, 256]);  bmm_20 = None
        add_86: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_233, where);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_234: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_86, [1024, 256, 256]);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_313: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_234, torch.float32);  view_234 = None
        amax_10: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_313, [-1], True)
        sub_31: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_313, amax_10);  convert_element_type_313 = amax_10 = None
        exp_10: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_314: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_225: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_301, [16384, 1024])
        permute_112: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_62: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg170_1, view_225, permute_112);  arg170_1 = view_225 = permute_112 = None
        view_226: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [64, 256, 1024]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_228: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_226, [64, -1, 16, 64]);  view_226 = None
        permute_114: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_83: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_232: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [1024, 256, 64]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_21: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_314, view_232);  convert_element_type_314 = view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_235: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [64, 16, 256, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_117: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_85: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_236: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [64, 256, 1024]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_237: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_236, [16384, 1024]);  view_236 = None
        permute_118: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_63: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg172_1, view_237, permute_118);  arg172_1 = view_237 = permute_118 = None
        view_238: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [64, 256, 1024]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_87: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_301, view_238);  convert_element_type_301 = view_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_320: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.float32);  add_87 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_320, [2], correction = 0, keepdim = True)
        getitem_42: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_320, getitem_43);  convert_element_type_320 = getitem_43 = None
        add_88: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_84: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_85: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, arg173_1);  mul_84 = arg173_1 = None
        add_89: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, arg174_1);  mul_85 = arg174_1 = None
        convert_element_type_321: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_239: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_321, [16384, 1024])
        permute_119: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg176_1, view_239, permute_119);  arg176_1 = view_239 = permute_119 = None
        view_240: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [64, 256, 4096]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_325: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_240, torch.float32);  view_240 = None
        mul_86: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_325, 0.5)
        mul_87: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_325, 0.7071067811865476);  convert_element_type_325 = None
        erf_10: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_90: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_88: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, add_90);  mul_86 = add_90 = None
        convert_element_type_326: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_241: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_326, [16384, 4096]);  convert_element_type_326 = None
        permute_120: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg178_1, view_241, permute_120);  arg178_1 = view_241 = permute_120 = None
        view_242: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [64, 256, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_91: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_321, view_242);  convert_element_type_321 = view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_330: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.float32);  add_91 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_330, [2], correction = 0, keepdim = True)
        getitem_44: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        sub_33: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_330, getitem_45);  convert_element_type_330 = getitem_45 = None
        add_92: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_89: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_90: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, arg179_1);  mul_89 = arg179_1 = None
        add_93: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, arg180_1);  mul_90 = arg180_1 = None
        convert_element_type_331: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_243: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_331, [16384, 1024])
        permute_121: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg182_1, view_243, permute_121);  arg182_1 = view_243 = permute_121 = None
        view_244: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [64, 256, 1024]);  addmm_66 = None
        mul_91: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_244, 0.125);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_251: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_91, [64, 256, 16, 64]);  mul_91 = None
        permute_126: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_89: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_252: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [1024, 256, 64]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_245: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_331, [16384, 1024])
        permute_122: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg184_1, view_245, permute_122);  arg184_1 = view_245 = permute_122 = None
        view_246: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [64, 256, 1024]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_249: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_246, [64, -1, 16, 64]);  view_246 = None
        permute_124: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_90: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_253: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [1024, 256, 64]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_127: "bf16[1024, 64, 256][16384, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        bmm_22: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_252, permute_127);  view_252 = permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_255: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [64, 16, 256, 256]);  bmm_22 = None
        add_94: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_255, where);  view_255 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_256: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_94, [1024, 256, 256]);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        convert_element_type_343: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_256, torch.float32);  view_256 = None
        amax_11: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_343, [-1], True)
        sub_34: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_343, amax_11);  convert_element_type_343 = amax_11 = None
        exp_11: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_344: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_247: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_331, [16384, 1024])
        permute_123: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg186_1, view_247, permute_123);  arg186_1 = view_247 = permute_123 = None
        view_248: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [64, 256, 1024]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_250: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_248, [64, -1, 16, 64]);  view_248 = None
        permute_125: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_91: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_254: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [1024, 256, 64]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_23: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(convert_element_type_344, view_254);  convert_element_type_344 = view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_257: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [64, 16, 256, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_128: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_93: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_258: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [64, 256, 1024]);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_259: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_258, [16384, 1024]);  view_258 = None
        permute_129: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg188_1, view_259, permute_129);  arg188_1 = view_259 = permute_129 = None
        view_260: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [64, 256, 1024]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        add_95: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_331, view_260);  convert_element_type_331 = view_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_350: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.float32);  add_95 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_350, [2], correction = 0, keepdim = True)
        getitem_46: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_350, getitem_47);  convert_element_type_350 = getitem_47 = None
        add_96: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_92: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_93: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, arg189_1);  mul_92 = arg189_1 = None
        add_97: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, arg190_1);  mul_93 = arg190_1 = None
        convert_element_type_351: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_261: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_351, [16384, 1024])
        permute_130: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg192_1, view_261, permute_130);  arg192_1 = view_261 = permute_130 = None
        view_262: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [64, 256, 4096]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_355: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_262, torch.float32);  view_262 = None
        mul_94: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_355, 0.5)
        mul_95: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_355, 0.7071067811865476);  convert_element_type_355 = None
        erf_11: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_98: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, add_98);  mul_94 = add_98 = None
        convert_element_type_356: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_263: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_356, [16384, 4096]);  convert_element_type_356 = None
        permute_131: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg194_1, view_263, permute_131);  arg194_1 = view_263 = permute_131 = None
        view_264: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [64, 256, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        add_99: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_351, view_264);  convert_element_type_351 = view_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_360: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.float32);  add_99 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_360, [2], correction = 0, keepdim = True)
        getitem_48: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[64, 256, 1][256, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:761 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_268: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(arg197_1, [-1]);  arg197_1 = None
        ne_1: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_268, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_36: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_360, getitem_49);  convert_element_type_360 = getitem_49 = None
        add_100: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_97: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_98: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, arg195_1);  mul_97 = arg195_1 = None
        add_101: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_98, arg196_1);  mul_98 = arg196_1 = None
        convert_element_type_361: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:756 in forward, code: logits = self.output_projection(outputs[0])
        view_265: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [16384, 1024]);  convert_element_type_361 = None
        permute_132: "bf16[1024, 50265][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "bf16[1024, 50272][50272, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_132, [0, 7, 0, 0]);  permute_132 = None
        mm_default: "bf16[16384, 50272][50272, 1]cuda:0" = torch.ops.aten.mm.default(view_265, constant_pad_nd_default);  view_265 = constant_pad_nd_default = None
        slice_tensor: "bf16[16384, 50265][50272, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -7);  mm_default = None
        view_266: "bf16[64, 256, 50265][12869632, 50272, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [64, 256, 50265]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:761 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_267: "bf16[16384, 50265][50272, 1]cuda:0" = torch.ops.aten.reshape.default(view_266, [-1, 50265])
        convert_element_type_364: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_267, torch.float32);  view_267 = None
        amax_12: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_364, [1], True)
        sub_37: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_364, amax_12);  convert_element_type_364 = amax_12 = None
        exp_12: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.exp.default(sub_37)
        sum_13: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_38: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_37, log);  sub_37 = log = None
        convert_element_type_365: "bf16[16384, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_38, torch.bfloat16);  sub_38 = None
        ne: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_268, -100)
        full_default_2: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[16384][1]cuda:0" = torch.ops.aten.where.self(ne, view_268, full_default_2);  ne = full_default_2 = None
        unsqueeze_6: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "bf16[16384, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_365, 1, unsqueeze_6);  convert_element_type_365 = unsqueeze_6 = None
        squeeze: "bf16[16384][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[16384][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[16384][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        sum_15: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_2);  where_2 = None
        ne_2: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_268, -100);  view_268 = None
        sum_14: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_366: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_14, torch.bfloat16);  sum_14 = None
        div_12: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_366);  sum_15 = convert_element_type_366 = None
        return (div_12, view_266)
