class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "i64[1, 512]", arg2_1: "i64[1, 512]", arg3_1: "f16[30522, 768]", arg4_1: "f16[2, 768]", arg5_1: "f16[512, 768]", arg6_1: "f16[768]", arg7_1: "f16[768]", arg8_1: "f16[768, 768]", arg9_1: "f16[768]", arg10_1: "f16[768, 768]", arg11_1: "f16[768]", arg12_1: "f16[768, 768]", arg13_1: "f16[768]", arg14_1: "f16[768, 768]", arg15_1: "f16[768]", arg16_1: "f16[768]", arg17_1: "f16[768]", arg18_1: "f16[3072, 768]", arg19_1: "f16[3072]", arg20_1: "f16[768, 3072]", arg21_1: "f16[768]", arg22_1: "f16[768]", arg23_1: "f16[768]", arg24_1: "f16[768, 768]", arg25_1: "f16[768]", arg26_1: "f16[768, 768]", arg27_1: "f16[768]", arg28_1: "f16[768, 768]", arg29_1: "f16[768]", arg30_1: "f16[768, 768]", arg31_1: "f16[768]", arg32_1: "f16[768]", arg33_1: "f16[768]", arg34_1: "f16[3072, 768]", arg35_1: "f16[3072]", arg36_1: "f16[768, 3072]", arg37_1: "f16[768]", arg38_1: "f16[768]", arg39_1: "f16[768]", arg40_1: "f16[768, 768]", arg41_1: "f16[768]", arg42_1: "f16[768, 768]", arg43_1: "f16[768]", arg44_1: "f16[768, 768]", arg45_1: "f16[768]", arg46_1: "f16[768, 768]", arg47_1: "f16[768]", arg48_1: "f16[768]", arg49_1: "f16[768]", arg50_1: "f16[3072, 768]", arg51_1: "f16[3072]", arg52_1: "f16[768, 3072]", arg53_1: "f16[768]", arg54_1: "f16[768]", arg55_1: "f16[768]", arg56_1: "f16[768, 768]", arg57_1: "f16[768]", arg58_1: "f16[768, 768]", arg59_1: "f16[768]", arg60_1: "f16[768, 768]", arg61_1: "f16[768]", arg62_1: "f16[768, 768]", arg63_1: "f16[768]", arg64_1: "f16[768]", arg65_1: "f16[768]", arg66_1: "f16[3072, 768]", arg67_1: "f16[3072]", arg68_1: "f16[768, 3072]", arg69_1: "f16[768]", arg70_1: "f16[768]", arg71_1: "f16[768]", arg72_1: "f16[768, 768]", arg73_1: "f16[768]", arg74_1: "f16[768, 768]", arg75_1: "f16[768]", arg76_1: "f16[768, 768]", arg77_1: "f16[768]", arg78_1: "f16[768, 768]", arg79_1: "f16[768]", arg80_1: "f16[768]", arg81_1: "f16[768]", arg82_1: "f16[3072, 768]", arg83_1: "f16[3072]", arg84_1: "f16[768, 3072]", arg85_1: "f16[768]", arg86_1: "f16[768]", arg87_1: "f16[768]", arg88_1: "f16[768, 768]", arg89_1: "f16[768]", arg90_1: "f16[768, 768]", arg91_1: "f16[768]", arg92_1: "f16[768, 768]", arg93_1: "f16[768]", arg94_1: "f16[768, 768]", arg95_1: "f16[768]", arg96_1: "f16[768]", arg97_1: "f16[768]", arg98_1: "f16[3072, 768]", arg99_1: "f16[3072]", arg100_1: "f16[768, 3072]", arg101_1: "f16[768]", arg102_1: "f16[768]", arg103_1: "f16[768]", arg104_1: "f16[768, 768]", arg105_1: "f16[768]", arg106_1: "f16[768, 768]", arg107_1: "f16[768]", arg108_1: "f16[768, 768]", arg109_1: "f16[768]", arg110_1: "f16[768, 768]", arg111_1: "f16[768]", arg112_1: "f16[768]", arg113_1: "f16[768]", arg114_1: "f16[3072, 768]", arg115_1: "f16[3072]", arg116_1: "f16[768, 3072]", arg117_1: "f16[768]", arg118_1: "f16[768]", arg119_1: "f16[768]", arg120_1: "f16[768, 768]", arg121_1: "f16[768]", arg122_1: "f16[768, 768]", arg123_1: "f16[768]", arg124_1: "f16[768, 768]", arg125_1: "f16[768]", arg126_1: "f16[768, 768]", arg127_1: "f16[768]", arg128_1: "f16[768]", arg129_1: "f16[768]", arg130_1: "f16[3072, 768]", arg131_1: "f16[3072]", arg132_1: "f16[768, 3072]", arg133_1: "f16[768]", arg134_1: "f16[768]", arg135_1: "f16[768]", arg136_1: "f16[768, 768]", arg137_1: "f16[768]", arg138_1: "f16[768, 768]", arg139_1: "f16[768]", arg140_1: "f16[768, 768]", arg141_1: "f16[768]", arg142_1: "f16[768, 768]", arg143_1: "f16[768]", arg144_1: "f16[768]", arg145_1: "f16[768]", arg146_1: "f16[3072, 768]", arg147_1: "f16[3072]", arg148_1: "f16[768, 3072]", arg149_1: "f16[768]", arg150_1: "f16[768]", arg151_1: "f16[768]", arg152_1: "f16[768, 768]", arg153_1: "f16[768]", arg154_1: "f16[768, 768]", arg155_1: "f16[768]", arg156_1: "f16[768, 768]", arg157_1: "f16[768]", arg158_1: "f16[768, 768]", arg159_1: "f16[768]", arg160_1: "f16[768]", arg161_1: "f16[768]", arg162_1: "f16[3072, 768]", arg163_1: "f16[3072]", arg164_1: "f16[768, 3072]", arg165_1: "f16[768]", arg166_1: "f16[768]", arg167_1: "f16[768]", arg168_1: "f16[768, 768]", arg169_1: "f16[768]", arg170_1: "f16[768, 768]", arg171_1: "f16[768]", arg172_1: "f16[768, 768]", arg173_1: "f16[768]", arg174_1: "f16[768, 768]", arg175_1: "f16[768]", arg176_1: "f16[768]", arg177_1: "f16[768]", arg178_1: "f16[3072, 768]", arg179_1: "f16[3072]", arg180_1: "f16[768, 3072]", arg181_1: "f16[768]", arg182_1: "f16[768]", arg183_1: "f16[768]", arg184_1: "f16[768, 768]", arg185_1: "f16[768]", arg186_1: "f16[768, 768]", arg187_1: "f16[768]", arg188_1: "f16[768, 768]", arg189_1: "f16[768]", arg190_1: "f16[768, 768]", arg191_1: "f16[768]", arg192_1: "f16[768]", arg193_1: "f16[768]", arg194_1: "f16[3072, 768]", arg195_1: "f16[3072]", arg196_1: "f16[768, 3072]", arg197_1: "f16[768]", arg198_1: "f16[768]", arg199_1: "f16[768]", arg200_1: "f16[768, 768]", arg201_1: "f16[768]", arg202_1: "f16[768]", arg203_1: "f16[768]", arg204_1: "f16[30522]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:96 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand: "i64[1, 512]" = torch.ops.aten.expand.default(arg2_1, [1, -1]);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:97 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather: "i64[1, 512]" = torch.ops.aten.gather.default(expand, 1, arg1_1);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:98 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[1, 512]" = torch.ops.aten.expand.default(gather, [1, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:104 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, expand_1);  arg4_1 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:105 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:107 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, arg1_1);  arg5_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:108 in forward, code: embeddings = embeddings + position_embeddings
        add_1: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        convert_element_type: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean[0]
        getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        convert_element_type_1: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.float16);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_1, [512, 768])
        permute: "f16[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "f16[512, 768]" = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm, [1, 512, 768]);  addmm = None
        view_2: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [1, 512, -1, 64]);  view_1 = None
        permute_1: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_11: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        mul_2: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_11, 0.3535533905932738);  convert_element_type_11 = None
        expand_3: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_2, [1, 12, 512, 64]);  mul_2 = None
        view_9: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_3, [12, 512, 64]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_3: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_1, [512, 768])
        permute_2: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_1: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_3, permute_2);  arg11_1 = view_3 = permute_2 = None
        view_4: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 768]);  addmm_1 = None
        view_5: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [1, 512, -1, 64]);  view_4 = None
        permute_3: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_12: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_3, torch.float32);  permute_3 = None
        permute_6: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_12, [0, 1, 3, 2]);  convert_element_type_12 = None
        mul_3: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        expand_4: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_3, [1, 12, 64, 512]);  mul_3 = None
        view_10: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_4, [12, 64, 512]);  expand_4 = None
        bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [1, 12, 512, 512]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[512]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 512, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_2: "b8[1, 1, 512, 512]" = torch.ops.aten.expand.default(ge, [1, -1, 512, 512]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_1, full_default);  full_default_1 = full_default = None
        add_6: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = where = None
        eq: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_6, -inf)
        logical_not: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_2: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_6, [-1], True)
        sub_1: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
        exp: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        where_1: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = full_default_2 = div = None
        expand_5: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_1, [1, 12, 512, 512]);  where_1 = None
        view_12: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_5, [12, 512, 512]);  expand_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_6: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_1, [512, 768])
        permute_4: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_2: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_6, permute_4);  arg13_1 = view_6 = permute_4 = None
        view_7: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [1, 512, 768]);  addmm_2 = None
        view_8: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_7, [1, 512, -1, 64]);  view_7 = None
        permute_5: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_13: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_5, torch.float32);  permute_5 = None
        expand_6: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_13, [1, 12, 512, 64]);  convert_element_type_13 = None
        view_13: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_6, [12, 512, 64]);  expand_6 = None
        bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
        view_14: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
        convert_element_type_15: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_14, torch.float16);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_15, [0, 2, 1, 3]);  convert_element_type_15 = None
        clone_1: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_1, [1, 512, -1]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "f16[512, 768]" = torch.ops.aten.reshape.default(view_15, [512, 768]);  view_15 = None
        permute_8: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_3: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_16, permute_8);  arg15_1 = view_16 = permute_8 = None
        view_17: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_7: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_17, convert_element_type_1);  view_17 = convert_element_type_1 = None
        convert_element_type_19: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_19, [2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_19, getitem_3);  convert_element_type_19 = getitem_3 = None
        add_8: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg16_1);  mul_4 = arg16_1 = None
        add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, arg17_1);  mul_5 = arg17_1 = None
        convert_element_type_20: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_9, torch.float16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_20, [512, 768])
        permute_9: "f16[768, 3072]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_4: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg19_1, view_18, permute_9);  arg19_1 = view_18 = permute_9 = None
        view_19: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_24: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5)
        mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.7071067811865476);  convert_element_type_24 = None
        erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_10: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_10);  mul_6 = add_10 = None
        convert_element_type_25: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_8, torch.float16);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_25, [512, 3072]);  convert_element_type_25 = None
        permute_10: "f16[3072, 768]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_5: "f16[512, 768]" = torch.ops.aten.addmm.default(arg21_1, view_20, permute_10);  arg21_1 = view_20 = permute_10 = None
        view_21: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_21, convert_element_type_20);  view_21 = convert_element_type_20 = None
        convert_element_type_29: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32);  add_11 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_29, [2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_29, getitem_5);  convert_element_type_29 = getitem_5 = None
        add_12: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg22_1);  mul_9 = arg22_1 = None
        add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, arg23_1);  mul_10 = arg23_1 = None
        convert_element_type_30: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_13, torch.float16);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_22: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_30, [512, 768])
        permute_11: "f16[768, 768]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_6: "f16[512, 768]" = torch.ops.aten.addmm.default(arg25_1, view_22, permute_11);  arg25_1 = view_22 = permute_11 = None
        view_23: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [1, 512, 768]);  addmm_6 = None
        view_24: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_23, [1, 512, -1, 64]);  view_23 = None
        permute_12: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_40: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_12, torch.float32);  permute_12 = None
        mul_11: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_40, 0.3535533905932738);  convert_element_type_40 = None
        expand_7: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_11, [1, 12, 512, 64]);  mul_11 = None
        view_31: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_7, [12, 512, 64]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_25: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_30, [512, 768])
        permute_13: "f16[768, 768]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_7: "f16[512, 768]" = torch.ops.aten.addmm.default(arg27_1, view_25, permute_13);  arg27_1 = view_25 = permute_13 = None
        view_26: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 768]);  addmm_7 = None
        view_27: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_26, [1, 512, -1, 64]);  view_26 = None
        permute_14: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_41: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_14, torch.float32);  permute_14 = None
        permute_17: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_41, [0, 1, 3, 2]);  convert_element_type_41 = None
        mul_12: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_17, 0.3535533905932738);  permute_17 = None
        expand_8: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_12, [1, 12, 64, 512]);  mul_12 = None
        view_32: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_8, [12, 64, 512]);  expand_8 = None
        bmm_2: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_31, view_32);  view_31 = view_32 = None
        view_33: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 512, 512]);  bmm_2 = None
        full_default_4: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        add_14: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_33, where_2);  view_33 = where_2 = None
        eq_1: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_14, -inf)
        logical_not_2: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_default_5: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_14, [-1], True)
        sub_4: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_14, amax_1);  add_14 = amax_1 = None
        exp_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        where_3: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full_default_5, div_1);  logical_not_3 = full_default_5 = div_1 = None
        expand_9: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_3, [1, 12, 512, 512]);  where_3 = None
        view_34: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_9, [12, 512, 512]);  expand_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_28: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_30, [512, 768])
        permute_15: "f16[768, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_8: "f16[512, 768]" = torch.ops.aten.addmm.default(arg29_1, view_28, permute_15);  arg29_1 = view_28 = permute_15 = None
        view_29: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [1, 512, 768]);  addmm_8 = None
        view_30: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_29, [1, 512, -1, 64]);  view_29 = None
        permute_16: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_42: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_16, torch.float32);  permute_16 = None
        expand_10: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_42, [1, 12, 512, 64]);  convert_element_type_42 = None
        view_35: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_10, [12, 512, 64]);  expand_10 = None
        bmm_3: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
        view_36: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 512, 64]);  bmm_3 = None
        convert_element_type_44: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_36, torch.float16);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_44, [0, 2, 1, 3]);  convert_element_type_44 = None
        clone_4: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_37: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_4, [1, 512, -1]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "f16[512, 768]" = torch.ops.aten.reshape.default(view_37, [512, 768]);  view_37 = None
        permute_19: "f16[768, 768]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_9: "f16[512, 768]" = torch.ops.aten.addmm.default(arg31_1, view_38, permute_19);  arg31_1 = view_38 = permute_19 = None
        view_39: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [1, 512, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_39, convert_element_type_30);  view_39 = convert_element_type_30 = None
        convert_element_type_48: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_15, torch.float32);  add_15 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_7);  convert_element_type_48 = getitem_7 = None
        add_16: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_13: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_14: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_13, arg32_1);  mul_13 = arg32_1 = None
        add_17: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_14, arg33_1);  mul_14 = arg33_1 = None
        convert_element_type_49: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_17, torch.float16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_49, [512, 768])
        permute_20: "f16[768, 3072]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_10: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg35_1, view_40, permute_20);  arg35_1 = view_40 = permute_20 = None
        view_41: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_53: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 0.5)
        mul_16: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 0.7071067811865476);  convert_element_type_53 = None
        erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
        add_18: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_17: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_15, add_18);  mul_15 = add_18 = None
        convert_element_type_54: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_17, torch.float16);  mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_54, [512, 3072]);  convert_element_type_54 = None
        permute_21: "f16[3072, 768]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_11: "f16[512, 768]" = torch.ops.aten.addmm.default(arg37_1, view_42, permute_21);  arg37_1 = view_42 = permute_21 = None
        view_43: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_19: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_43, convert_element_type_49);  view_43 = convert_element_type_49 = None
        convert_element_type_58: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.float32);  add_19 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_58, [2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_58, getitem_9);  convert_element_type_58 = getitem_9 = None
        add_20: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg38_1);  mul_18 = arg38_1 = None
        add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, arg39_1);  mul_19 = arg39_1 = None
        convert_element_type_59: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_21, torch.float16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_44: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_59, [512, 768])
        permute_22: "f16[768, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_12: "f16[512, 768]" = torch.ops.aten.addmm.default(arg41_1, view_44, permute_22);  arg41_1 = view_44 = permute_22 = None
        view_45: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [1, 512, 768]);  addmm_12 = None
        view_46: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_45, [1, 512, -1, 64]);  view_45 = None
        permute_23: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_69: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_23, torch.float32);  permute_23 = None
        mul_20: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_69, 0.3535533905932738);  convert_element_type_69 = None
        expand_11: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_20, [1, 12, 512, 64]);  mul_20 = None
        view_53: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_11, [12, 512, 64]);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_47: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_59, [512, 768])
        permute_24: "f16[768, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_13: "f16[512, 768]" = torch.ops.aten.addmm.default(arg43_1, view_47, permute_24);  arg43_1 = view_47 = permute_24 = None
        view_48: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 768]);  addmm_13 = None
        view_49: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_48, [1, 512, -1, 64]);  view_48 = None
        permute_25: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_70: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_25, torch.float32);  permute_25 = None
        permute_28: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_70, [0, 1, 3, 2]);  convert_element_type_70 = None
        mul_21: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_28, 0.3535533905932738);  permute_28 = None
        expand_12: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_21, [1, 12, 64, 512]);  mul_21 = None
        view_54: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_12, [12, 64, 512]);  expand_12 = None
        bmm_4: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_53, view_54);  view_53 = view_54 = None
        view_55: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 512, 512]);  bmm_4 = None
        full_default_7: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        add_22: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_55, where_4);  view_55 = where_4 = None
        eq_2: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_22, -inf)
        logical_not_4: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_default_8: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_22, [-1], True)
        sub_7: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_22, amax_2);  add_22 = amax_2 = None
        exp_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        where_5: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full_default_8, div_2);  logical_not_5 = full_default_8 = div_2 = None
        expand_13: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_5, [1, 12, 512, 512]);  where_5 = None
        view_56: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_13, [12, 512, 512]);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_50: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_59, [512, 768])
        permute_26: "f16[768, 768]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_14: "f16[512, 768]" = torch.ops.aten.addmm.default(arg45_1, view_50, permute_26);  arg45_1 = view_50 = permute_26 = None
        view_51: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [1, 512, 768]);  addmm_14 = None
        view_52: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_51, [1, 512, -1, 64]);  view_51 = None
        permute_27: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_71: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_27, torch.float32);  permute_27 = None
        expand_14: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_71, [1, 12, 512, 64]);  convert_element_type_71 = None
        view_57: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_14, [12, 512, 64]);  expand_14 = None
        bmm_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_56, view_57);  view_56 = view_57 = None
        view_58: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 512, 64]);  bmm_5 = None
        convert_element_type_73: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_58, torch.float16);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_73, [0, 2, 1, 3]);  convert_element_type_73 = None
        clone_7: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_59: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_7, [1, 512, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f16[512, 768]" = torch.ops.aten.reshape.default(view_59, [512, 768]);  view_59 = None
        permute_30: "f16[768, 768]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_15: "f16[512, 768]" = torch.ops.aten.addmm.default(arg47_1, view_60, permute_30);  arg47_1 = view_60 = permute_30 = None
        view_61: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [1, 512, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_61, convert_element_type_59);  view_61 = convert_element_type_59 = None
        convert_element_type_77: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_77, [2], correction = 0, keepdim = True)
        getitem_10: "f32[1, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_77, getitem_11);  convert_element_type_77 = getitem_11 = None
        add_24: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_22: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_23: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg48_1);  mul_22 = arg48_1 = None
        add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_23, arg49_1);  mul_23 = arg49_1 = None
        convert_element_type_78: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_25, torch.float16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_78, [512, 768])
        permute_31: "f16[768, 3072]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_16: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg51_1, view_62, permute_31);  arg51_1 = view_62 = permute_31 = None
        view_63: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_82: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_24: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_82, 0.5)
        mul_25: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_82, 0.7071067811865476);  convert_element_type_82 = None
        erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_26: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_26: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_24, add_26);  mul_24 = add_26 = None
        convert_element_type_83: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_26, torch.float16);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_83, [512, 3072]);  convert_element_type_83 = None
        permute_32: "f16[3072, 768]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_17: "f16[512, 768]" = torch.ops.aten.addmm.default(arg53_1, view_64, permute_32);  arg53_1 = view_64 = permute_32 = None
        view_65: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_65, convert_element_type_78);  view_65 = convert_element_type_78 = None
        convert_element_type_87: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_87, [2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_87, getitem_13);  convert_element_type_87 = getitem_13 = None
        add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_27: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_28: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_27, arg54_1);  mul_27 = arg54_1 = None
        add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_28, arg55_1);  mul_28 = arg55_1 = None
        convert_element_type_88: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_29, torch.float16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_66: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_88, [512, 768])
        permute_33: "f16[768, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_18: "f16[512, 768]" = torch.ops.aten.addmm.default(arg57_1, view_66, permute_33);  arg57_1 = view_66 = permute_33 = None
        view_67: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [1, 512, 768]);  addmm_18 = None
        view_68: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_67, [1, 512, -1, 64]);  view_67 = None
        permute_34: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_98: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_34, torch.float32);  permute_34 = None
        mul_29: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_98, 0.3535533905932738);  convert_element_type_98 = None
        expand_15: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_29, [1, 12, 512, 64]);  mul_29 = None
        view_75: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_15, [12, 512, 64]);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_69: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_88, [512, 768])
        permute_35: "f16[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_19: "f16[512, 768]" = torch.ops.aten.addmm.default(arg59_1, view_69, permute_35);  arg59_1 = view_69 = permute_35 = None
        view_70: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 768]);  addmm_19 = None
        view_71: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_70, [1, 512, -1, 64]);  view_70 = None
        permute_36: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_99: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_36, torch.float32);  permute_36 = None
        permute_39: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_99, [0, 1, 3, 2]);  convert_element_type_99 = None
        mul_30: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_39, 0.3535533905932738);  permute_39 = None
        expand_16: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_30, [1, 12, 64, 512]);  mul_30 = None
        view_76: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_16, [12, 64, 512]);  expand_16 = None
        bmm_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_75, view_76);  view_75 = view_76 = None
        view_77: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 512, 512]);  bmm_6 = None
        full_default_10: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        add_30: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_77, where_6);  view_77 = where_6 = None
        eq_3: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_30, -inf)
        logical_not_6: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_default_11: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_30, [-1], True)
        sub_10: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_30, amax_3);  add_30 = amax_3 = None
        exp_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        where_7: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full_default_11, div_3);  logical_not_7 = full_default_11 = div_3 = None
        expand_17: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_7, [1, 12, 512, 512]);  where_7 = None
        view_78: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_17, [12, 512, 512]);  expand_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_72: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_88, [512, 768])
        permute_37: "f16[768, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_20: "f16[512, 768]" = torch.ops.aten.addmm.default(arg61_1, view_72, permute_37);  arg61_1 = view_72 = permute_37 = None
        view_73: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [1, 512, 768]);  addmm_20 = None
        view_74: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_73, [1, 512, -1, 64]);  view_73 = None
        permute_38: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_100: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_38, torch.float32);  permute_38 = None
        expand_18: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_100, [1, 12, 512, 64]);  convert_element_type_100 = None
        view_79: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_18, [12, 512, 64]);  expand_18 = None
        bmm_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
        view_80: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 512, 64]);  bmm_7 = None
        convert_element_type_102: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_80, torch.float16);  view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_102, [0, 2, 1, 3]);  convert_element_type_102 = None
        clone_10: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_81: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_10, [1, 512, -1]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_82: "f16[512, 768]" = torch.ops.aten.reshape.default(view_81, [512, 768]);  view_81 = None
        permute_41: "f16[768, 768]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_21: "f16[512, 768]" = torch.ops.aten.addmm.default(arg63_1, view_82, permute_41);  arg63_1 = view_82 = permute_41 = None
        view_83: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_83, convert_element_type_88);  view_83 = convert_element_type_88 = None
        convert_element_type_106: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32);  add_31 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_106, [2], correction = 0, keepdim = True)
        getitem_14: "f32[1, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_106, getitem_15);  convert_element_type_106 = getitem_15 = None
        add_32: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_31: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, arg64_1);  mul_31 = arg64_1 = None
        add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_32, arg65_1);  mul_32 = arg65_1 = None
        convert_element_type_107: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_33, torch.float16);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_107, [512, 768])
        permute_42: "f16[768, 3072]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_22: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg67_1, view_84, permute_42);  arg67_1 = view_84 = permute_42 = None
        view_85: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_111: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_33: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_111, 0.5)
        mul_34: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_111, 0.7071067811865476);  convert_element_type_111 = None
        erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_34: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_35: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_33, add_34);  mul_33 = add_34 = None
        convert_element_type_112: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_35, torch.float16);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_112, [512, 3072]);  convert_element_type_112 = None
        permute_43: "f16[3072, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_23: "f16[512, 768]" = torch.ops.aten.addmm.default(arg69_1, view_86, permute_43);  arg69_1 = view_86 = permute_43 = None
        view_87: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_35: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_87, convert_element_type_107);  view_87 = convert_element_type_107 = None
        convert_element_type_116: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_116, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_116, getitem_17);  convert_element_type_116 = getitem_17 = None
        add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_36: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_37: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg70_1);  mul_36 = arg70_1 = None
        add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, arg71_1);  mul_37 = arg71_1 = None
        convert_element_type_117: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_88: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_117, [512, 768])
        permute_44: "f16[768, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_24: "f16[512, 768]" = torch.ops.aten.addmm.default(arg73_1, view_88, permute_44);  arg73_1 = view_88 = permute_44 = None
        view_89: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [1, 512, 768]);  addmm_24 = None
        view_90: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_89, [1, 512, -1, 64]);  view_89 = None
        permute_45: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_127: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_45, torch.float32);  permute_45 = None
        mul_38: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_127, 0.3535533905932738);  convert_element_type_127 = None
        expand_19: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_38, [1, 12, 512, 64]);  mul_38 = None
        view_97: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_19, [12, 512, 64]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_91: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_117, [512, 768])
        permute_46: "f16[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_25: "f16[512, 768]" = torch.ops.aten.addmm.default(arg75_1, view_91, permute_46);  arg75_1 = view_91 = permute_46 = None
        view_92: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [1, 512, 768]);  addmm_25 = None
        view_93: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_92, [1, 512, -1, 64]);  view_92 = None
        permute_47: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_128: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_47, torch.float32);  permute_47 = None
        permute_50: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_128, [0, 1, 3, 2]);  convert_element_type_128 = None
        mul_39: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_50, 0.3535533905932738);  permute_50 = None
        expand_20: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_39, [1, 12, 64, 512]);  mul_39 = None
        view_98: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_20, [12, 64, 512]);  expand_20 = None
        bmm_8: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
        view_99: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 512, 512]);  bmm_8 = None
        full_default_13: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        add_38: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_99, where_8);  view_99 = where_8 = None
        eq_4: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_38, -inf)
        logical_not_8: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_default_14: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_38, [-1], True)
        sub_13: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_38, amax_4);  add_38 = amax_4 = None
        exp_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        where_9: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full_default_14, div_4);  logical_not_9 = full_default_14 = div_4 = None
        expand_21: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_9, [1, 12, 512, 512]);  where_9 = None
        view_100: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_21, [12, 512, 512]);  expand_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_94: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_117, [512, 768])
        permute_48: "f16[768, 768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_26: "f16[512, 768]" = torch.ops.aten.addmm.default(arg77_1, view_94, permute_48);  arg77_1 = view_94 = permute_48 = None
        view_95: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [1, 512, 768]);  addmm_26 = None
        view_96: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_95, [1, 512, -1, 64]);  view_95 = None
        permute_49: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_129: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_49, torch.float32);  permute_49 = None
        expand_22: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_129, [1, 12, 512, 64]);  convert_element_type_129 = None
        view_101: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_22, [12, 512, 64]);  expand_22 = None
        bmm_9: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
        view_102: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 512, 64]);  bmm_9 = None
        convert_element_type_131: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_102, torch.float16);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_131, [0, 2, 1, 3]);  convert_element_type_131 = None
        clone_13: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_103: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_13, [1, 512, -1]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "f16[512, 768]" = torch.ops.aten.reshape.default(view_103, [512, 768]);  view_103 = None
        permute_52: "f16[768, 768]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_27: "f16[512, 768]" = torch.ops.aten.addmm.default(arg79_1, view_104, permute_52);  arg79_1 = view_104 = permute_52 = None
        view_105: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [1, 512, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_105, convert_element_type_117);  view_105 = convert_element_type_117 = None
        convert_element_type_135: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_39, torch.float32);  add_39 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_135, [2], correction = 0, keepdim = True)
        getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_135, getitem_19);  convert_element_type_135 = getitem_19 = None
        add_40: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg80_1);  mul_40 = arg80_1 = None
        add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, arg81_1);  mul_41 = arg81_1 = None
        convert_element_type_136: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_41, torch.float16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_136, [512, 768])
        permute_53: "f16[768, 3072]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_28: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg83_1, view_106, permute_53);  arg83_1 = view_106 = permute_53 = None
        view_107: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_140: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_42: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_140, 0.5)
        mul_43: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_140, 0.7071067811865476);  convert_element_type_140 = None
        erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_42: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_42, add_42);  mul_42 = add_42 = None
        convert_element_type_141: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_44, torch.float16);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_141, [512, 3072]);  convert_element_type_141 = None
        permute_54: "f16[3072, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_29: "f16[512, 768]" = torch.ops.aten.addmm.default(arg85_1, view_108, permute_54);  arg85_1 = view_108 = permute_54 = None
        view_109: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_29, [1, 512, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_109, convert_element_type_136);  view_109 = convert_element_type_136 = None
        convert_element_type_145: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.float32);  add_43 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_145, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_145, getitem_21);  convert_element_type_145 = getitem_21 = None
        add_44: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_45: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_46: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg86_1);  mul_45 = arg86_1 = None
        add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, arg87_1);  mul_46 = arg87_1 = None
        convert_element_type_146: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_45, torch.float16);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_110: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_146, [512, 768])
        permute_55: "f16[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_30: "f16[512, 768]" = torch.ops.aten.addmm.default(arg89_1, view_110, permute_55);  arg89_1 = view_110 = permute_55 = None
        view_111: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [1, 512, 768]);  addmm_30 = None
        view_112: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_111, [1, 512, -1, 64]);  view_111 = None
        permute_56: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_156: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_56, torch.float32);  permute_56 = None
        mul_47: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_156, 0.3535533905932738);  convert_element_type_156 = None
        expand_23: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_47, [1, 12, 512, 64]);  mul_47 = None
        view_119: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_23, [12, 512, 64]);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_113: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_146, [512, 768])
        permute_57: "f16[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_31: "f16[512, 768]" = torch.ops.aten.addmm.default(arg91_1, view_113, permute_57);  arg91_1 = view_113 = permute_57 = None
        view_114: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_31, [1, 512, 768]);  addmm_31 = None
        view_115: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_114, [1, 512, -1, 64]);  view_114 = None
        permute_58: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_157: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_58, torch.float32);  permute_58 = None
        permute_61: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_157, [0, 1, 3, 2]);  convert_element_type_157 = None
        mul_48: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_61, 0.3535533905932738);  permute_61 = None
        expand_24: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_48, [1, 12, 64, 512]);  mul_48 = None
        view_120: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_24, [12, 64, 512]);  expand_24 = None
        bmm_10: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
        view_121: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 512, 512]);  bmm_10 = None
        full_default_16: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        add_46: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_121, where_10);  view_121 = where_10 = None
        eq_5: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_46, -inf)
        logical_not_10: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_default_17: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_46, [-1], True)
        sub_16: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_46, amax_5);  add_46 = amax_5 = None
        exp_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        where_11: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full_default_17, div_5);  logical_not_11 = full_default_17 = div_5 = None
        expand_25: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_11, [1, 12, 512, 512]);  where_11 = None
        view_122: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_25, [12, 512, 512]);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_116: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_146, [512, 768])
        permute_59: "f16[768, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_32: "f16[512, 768]" = torch.ops.aten.addmm.default(arg93_1, view_116, permute_59);  arg93_1 = view_116 = permute_59 = None
        view_117: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [1, 512, 768]);  addmm_32 = None
        view_118: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_117, [1, 512, -1, 64]);  view_117 = None
        permute_60: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_158: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_60, torch.float32);  permute_60 = None
        expand_26: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_158, [1, 12, 512, 64]);  convert_element_type_158 = None
        view_123: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_26, [12, 512, 64]);  expand_26 = None
        bmm_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
        view_124: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 512, 64]);  bmm_11 = None
        convert_element_type_160: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_124, torch.float16);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_160, [0, 2, 1, 3]);  convert_element_type_160 = None
        clone_16: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_125: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_16, [1, 512, -1]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f16[512, 768]" = torch.ops.aten.reshape.default(view_125, [512, 768]);  view_125 = None
        permute_63: "f16[768, 768]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_33: "f16[512, 768]" = torch.ops.aten.addmm.default(arg95_1, view_126, permute_63);  arg95_1 = view_126 = permute_63 = None
        view_127: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_33, [1, 512, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_47: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_127, convert_element_type_146);  view_127 = convert_element_type_146 = None
        convert_element_type_164: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_47, torch.float32);  add_47 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_164, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_164, getitem_23);  convert_element_type_164 = getitem_23 = None
        add_48: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_49: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg96_1);  mul_49 = arg96_1 = None
        add_49: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, arg97_1);  mul_50 = arg97_1 = None
        convert_element_type_165: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_49, torch.float16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_165, [512, 768])
        permute_64: "f16[768, 3072]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_34: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg99_1, view_128, permute_64);  arg99_1 = view_128 = permute_64 = None
        view_129: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_169: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_51: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_169, 0.5)
        mul_52: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_169, 0.7071067811865476);  convert_element_type_169 = None
        erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_50: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_53: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_51, add_50);  mul_51 = add_50 = None
        convert_element_type_170: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_53, torch.float16);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_170, [512, 3072]);  convert_element_type_170 = None
        permute_65: "f16[3072, 768]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_35: "f16[512, 768]" = torch.ops.aten.addmm.default(arg101_1, view_130, permute_65);  arg101_1 = view_130 = permute_65 = None
        view_131: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_35, [1, 512, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_51: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_131, convert_element_type_165);  view_131 = convert_element_type_165 = None
        convert_element_type_174: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32);  add_51 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_174, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_174, getitem_25);  convert_element_type_174 = getitem_25 = None
        add_52: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_54: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_55: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg102_1);  mul_54 = arg102_1 = None
        add_53: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_55, arg103_1);  mul_55 = arg103_1 = None
        convert_element_type_175: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_53, torch.float16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_132: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_175, [512, 768])
        permute_66: "f16[768, 768]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_36: "f16[512, 768]" = torch.ops.aten.addmm.default(arg105_1, view_132, permute_66);  arg105_1 = view_132 = permute_66 = None
        view_133: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [1, 512, 768]);  addmm_36 = None
        view_134: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_133, [1, 512, -1, 64]);  view_133 = None
        permute_67: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_185: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_67, torch.float32);  permute_67 = None
        mul_56: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_185, 0.3535533905932738);  convert_element_type_185 = None
        expand_27: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_56, [1, 12, 512, 64]);  mul_56 = None
        view_141: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_27, [12, 512, 64]);  expand_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_135: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_175, [512, 768])
        permute_68: "f16[768, 768]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_37: "f16[512, 768]" = torch.ops.aten.addmm.default(arg107_1, view_135, permute_68);  arg107_1 = view_135 = permute_68 = None
        view_136: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_37, [1, 512, 768]);  addmm_37 = None
        view_137: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_136, [1, 512, -1, 64]);  view_136 = None
        permute_69: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_186: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_69, torch.float32);  permute_69 = None
        permute_72: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_186, [0, 1, 3, 2]);  convert_element_type_186 = None
        mul_57: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_72, 0.3535533905932738);  permute_72 = None
        expand_28: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_57, [1, 12, 64, 512]);  mul_57 = None
        view_142: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_28, [12, 64, 512]);  expand_28 = None
        bmm_12: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_141, view_142);  view_141 = view_142 = None
        view_143: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [1, 12, 512, 512]);  bmm_12 = None
        full_default_19: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        add_54: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_143, where_12);  view_143 = where_12 = None
        eq_6: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_54, -inf)
        logical_not_12: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        full_default_20: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_6: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_54, [-1], True)
        sub_19: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_54, amax_6);  add_54 = amax_6 = None
        exp_6: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        where_13: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_13, full_default_20, div_6);  logical_not_13 = full_default_20 = div_6 = None
        expand_29: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_13, [1, 12, 512, 512]);  where_13 = None
        view_144: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_29, [12, 512, 512]);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_138: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_175, [512, 768])
        permute_70: "f16[768, 768]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_38: "f16[512, 768]" = torch.ops.aten.addmm.default(arg109_1, view_138, permute_70);  arg109_1 = view_138 = permute_70 = None
        view_139: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_38, [1, 512, 768]);  addmm_38 = None
        view_140: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_139, [1, 512, -1, 64]);  view_139 = None
        permute_71: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_187: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_71, torch.float32);  permute_71 = None
        expand_30: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_187, [1, 12, 512, 64]);  convert_element_type_187 = None
        view_145: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_30, [12, 512, 64]);  expand_30 = None
        bmm_13: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_144, view_145);  view_144 = view_145 = None
        view_146: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 12, 512, 64]);  bmm_13 = None
        convert_element_type_189: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_146, torch.float16);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_189, [0, 2, 1, 3]);  convert_element_type_189 = None
        clone_19: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_147: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_19, [1, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "f16[512, 768]" = torch.ops.aten.reshape.default(view_147, [512, 768]);  view_147 = None
        permute_74: "f16[768, 768]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_39: "f16[512, 768]" = torch.ops.aten.addmm.default(arg111_1, view_148, permute_74);  arg111_1 = view_148 = permute_74 = None
        view_149: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_39, [1, 512, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_55: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_149, convert_element_type_175);  view_149 = convert_element_type_175 = None
        convert_element_type_193: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_193, [2], correction = 0, keepdim = True)
        getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_193, getitem_27);  convert_element_type_193 = getitem_27 = None
        add_56: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg112_1);  mul_58 = arg112_1 = None
        add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, arg113_1);  mul_59 = arg113_1 = None
        convert_element_type_194: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_57, torch.float16);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_194, [512, 768])
        permute_75: "f16[768, 3072]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_40: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg115_1, view_150, permute_75);  arg115_1 = view_150 = permute_75 = None
        view_151: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [1, 512, 3072]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_198: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_60: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.5)
        mul_61: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.7071067811865476);  convert_element_type_198 = None
        erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_58: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_58);  mul_60 = add_58 = None
        convert_element_type_199: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_62, torch.float16);  mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_199, [512, 3072]);  convert_element_type_199 = None
        permute_76: "f16[3072, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_41: "f16[512, 768]" = torch.ops.aten.addmm.default(arg117_1, view_152, permute_76);  arg117_1 = view_152 = permute_76 = None
        view_153: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_41, [1, 512, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_153, convert_element_type_194);  view_153 = convert_element_type_194 = None
        convert_element_type_203: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_203, [2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_203, getitem_29);  convert_element_type_203 = getitem_29 = None
        add_60: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_63: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_64: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg118_1);  mul_63 = arg118_1 = None
        add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_64, arg119_1);  mul_64 = arg119_1 = None
        convert_element_type_204: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_61, torch.float16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_154: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_204, [512, 768])
        permute_77: "f16[768, 768]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_42: "f16[512, 768]" = torch.ops.aten.addmm.default(arg121_1, view_154, permute_77);  arg121_1 = view_154 = permute_77 = None
        view_155: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_42, [1, 512, 768]);  addmm_42 = None
        view_156: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_155, [1, 512, -1, 64]);  view_155 = None
        permute_78: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_214: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_78, torch.float32);  permute_78 = None
        mul_65: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_214, 0.3535533905932738);  convert_element_type_214 = None
        expand_31: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_65, [1, 12, 512, 64]);  mul_65 = None
        view_163: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_31, [12, 512, 64]);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_157: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_204, [512, 768])
        permute_79: "f16[768, 768]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_43: "f16[512, 768]" = torch.ops.aten.addmm.default(arg123_1, view_157, permute_79);  arg123_1 = view_157 = permute_79 = None
        view_158: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_43, [1, 512, 768]);  addmm_43 = None
        view_159: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_158, [1, 512, -1, 64]);  view_158 = None
        permute_80: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_215: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_80, torch.float32);  permute_80 = None
        permute_83: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_215, [0, 1, 3, 2]);  convert_element_type_215 = None
        mul_66: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_83, 0.3535533905932738);  permute_83 = None
        expand_32: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_66, [1, 12, 64, 512]);  mul_66 = None
        view_164: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_32, [12, 64, 512]);  expand_32 = None
        bmm_14: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_163, view_164);  view_163 = view_164 = None
        view_165: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_14, [1, 12, 512, 512]);  bmm_14 = None
        full_default_22: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        add_62: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_165, where_14);  view_165 = where_14 = None
        eq_7: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_62, -inf)
        logical_not_14: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        full_default_23: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_7: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_62, [-1], True)
        sub_22: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_62, amax_7);  add_62 = amax_7 = None
        exp_7: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        where_15: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_15, full_default_23, div_7);  logical_not_15 = full_default_23 = div_7 = None
        expand_33: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_15, [1, 12, 512, 512]);  where_15 = None
        view_166: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_33, [12, 512, 512]);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_160: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_204, [512, 768])
        permute_81: "f16[768, 768]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_44: "f16[512, 768]" = torch.ops.aten.addmm.default(arg125_1, view_160, permute_81);  arg125_1 = view_160 = permute_81 = None
        view_161: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_44, [1, 512, 768]);  addmm_44 = None
        view_162: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_161, [1, 512, -1, 64]);  view_161 = None
        permute_82: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_216: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_82, torch.float32);  permute_82 = None
        expand_34: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_216, [1, 12, 512, 64]);  convert_element_type_216 = None
        view_167: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_34, [12, 512, 64]);  expand_34 = None
        bmm_15: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
        view_168: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 12, 512, 64]);  bmm_15 = None
        convert_element_type_218: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_168, torch.float16);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_218, [0, 2, 1, 3]);  convert_element_type_218 = None
        clone_22: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_22, [1, 512, -1]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f16[512, 768]" = torch.ops.aten.reshape.default(view_169, [512, 768]);  view_169 = None
        permute_85: "f16[768, 768]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_45: "f16[512, 768]" = torch.ops.aten.addmm.default(arg127_1, view_170, permute_85);  arg127_1 = view_170 = permute_85 = None
        view_171: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_45, [1, 512, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_171, convert_element_type_204);  view_171 = convert_element_type_204 = None
        convert_element_type_222: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_222, [2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_222, getitem_31);  convert_element_type_222 = getitem_31 = None
        add_64: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_67: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_68: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_67, arg128_1);  mul_67 = arg128_1 = None
        add_65: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_68, arg129_1);  mul_68 = arg129_1 = None
        convert_element_type_223: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_65, torch.float16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_223, [512, 768])
        permute_86: "f16[768, 3072]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_46: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg131_1, view_172, permute_86);  arg131_1 = view_172 = permute_86 = None
        view_173: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [1, 512, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_227: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_69: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_227, 0.5)
        mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_227, 0.7071067811865476);  convert_element_type_227 = None
        erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_70);  mul_70 = None
        add_66: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_69, add_66);  mul_69 = add_66 = None
        convert_element_type_228: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_71, torch.float16);  mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_228, [512, 3072]);  convert_element_type_228 = None
        permute_87: "f16[3072, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_47: "f16[512, 768]" = torch.ops.aten.addmm.default(arg133_1, view_174, permute_87);  arg133_1 = view_174 = permute_87 = None
        view_175: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_47, [1, 512, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_175, convert_element_type_223);  view_175 = convert_element_type_223 = None
        convert_element_type_232: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_67, torch.float32);  add_67 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_232, [2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_232, getitem_33);  convert_element_type_232 = getitem_33 = None
        add_68: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_72: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg134_1);  mul_72 = arg134_1 = None
        add_69: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_73, arg135_1);  mul_73 = arg135_1 = None
        convert_element_type_233: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_69, torch.float16);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_176: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_233, [512, 768])
        permute_88: "f16[768, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_48: "f16[512, 768]" = torch.ops.aten.addmm.default(arg137_1, view_176, permute_88);  arg137_1 = view_176 = permute_88 = None
        view_177: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_48, [1, 512, 768]);  addmm_48 = None
        view_178: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_177, [1, 512, -1, 64]);  view_177 = None
        permute_89: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_243: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_89, torch.float32);  permute_89 = None
        mul_74: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_243, 0.3535533905932738);  convert_element_type_243 = None
        expand_35: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_74, [1, 12, 512, 64]);  mul_74 = None
        view_185: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_35, [12, 512, 64]);  expand_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_179: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_233, [512, 768])
        permute_90: "f16[768, 768]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_49: "f16[512, 768]" = torch.ops.aten.addmm.default(arg139_1, view_179, permute_90);  arg139_1 = view_179 = permute_90 = None
        view_180: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_49, [1, 512, 768]);  addmm_49 = None
        view_181: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_180, [1, 512, -1, 64]);  view_180 = None
        permute_91: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_244: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_91, torch.float32);  permute_91 = None
        permute_94: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_244, [0, 1, 3, 2]);  convert_element_type_244 = None
        mul_75: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_94, 0.3535533905932738);  permute_94 = None
        expand_36: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_75, [1, 12, 64, 512]);  mul_75 = None
        view_186: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_36, [12, 64, 512]);  expand_36 = None
        bmm_16: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_185, view_186);  view_185 = view_186 = None
        view_187: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [1, 12, 512, 512]);  bmm_16 = None
        full_default_25: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_25, full_default_24);  full_default_25 = full_default_24 = None
        add_70: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_187, where_16);  view_187 = where_16 = None
        eq_8: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_70, -inf)
        logical_not_16: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        full_default_26: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_8: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_70, [-1], True)
        sub_25: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_70, amax_8);  add_70 = amax_8 = None
        exp_8: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        where_17: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_17, full_default_26, div_8);  logical_not_17 = full_default_26 = div_8 = None
        expand_37: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_17, [1, 12, 512, 512]);  where_17 = None
        view_188: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_37, [12, 512, 512]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_182: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_233, [512, 768])
        permute_92: "f16[768, 768]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_50: "f16[512, 768]" = torch.ops.aten.addmm.default(arg141_1, view_182, permute_92);  arg141_1 = view_182 = permute_92 = None
        view_183: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_50, [1, 512, 768]);  addmm_50 = None
        view_184: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_183, [1, 512, -1, 64]);  view_183 = None
        permute_93: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_245: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_93, torch.float32);  permute_93 = None
        expand_38: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_245, [1, 12, 512, 64]);  convert_element_type_245 = None
        view_189: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_38, [12, 512, 64]);  expand_38 = None
        bmm_17: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_188, view_189);  view_188 = view_189 = None
        view_190: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 12, 512, 64]);  bmm_17 = None
        convert_element_type_247: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_190, torch.float16);  view_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_247, [0, 2, 1, 3]);  convert_element_type_247 = None
        clone_25: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_191: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_25, [1, 512, -1]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "f16[512, 768]" = torch.ops.aten.reshape.default(view_191, [512, 768]);  view_191 = None
        permute_96: "f16[768, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_51: "f16[512, 768]" = torch.ops.aten.addmm.default(arg143_1, view_192, permute_96);  arg143_1 = view_192 = permute_96 = None
        view_193: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_51, [1, 512, 768]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_193, convert_element_type_233);  view_193 = convert_element_type_233 = None
        convert_element_type_251: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_251, [2], correction = 0, keepdim = True)
        getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_251, getitem_35);  convert_element_type_251 = getitem_35 = None
        add_72: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_76: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_77: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_76, arg144_1);  mul_76 = arg144_1 = None
        add_73: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_77, arg145_1);  mul_77 = arg145_1 = None
        convert_element_type_252: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_73, torch.float16);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_252, [512, 768])
        permute_97: "f16[768, 3072]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_52: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg147_1, view_194, permute_97);  arg147_1 = view_194 = permute_97 = None
        view_195: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [1, 512, 3072]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_256: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_78: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_256, 0.5)
        mul_79: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_256, 0.7071067811865476);  convert_element_type_256 = None
        erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_74: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_80: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_78, add_74);  mul_78 = add_74 = None
        convert_element_type_257: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_80, torch.float16);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_257, [512, 3072]);  convert_element_type_257 = None
        permute_98: "f16[3072, 768]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_53: "f16[512, 768]" = torch.ops.aten.addmm.default(arg149_1, view_196, permute_98);  arg149_1 = view_196 = permute_98 = None
        view_197: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_53, [1, 512, 768]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_75: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_197, convert_element_type_252);  view_197 = convert_element_type_252 = None
        convert_element_type_261: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_75, torch.float32);  add_75 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_261, [2], correction = 0, keepdim = True)
        getitem_36: "f32[1, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_27: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_261, getitem_37);  convert_element_type_261 = getitem_37 = None
        add_76: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_82: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, arg150_1);  mul_81 = arg150_1 = None
        add_77: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, arg151_1);  mul_82 = arg151_1 = None
        convert_element_type_262: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_77, torch.float16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_198: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_262, [512, 768])
        permute_99: "f16[768, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_54: "f16[512, 768]" = torch.ops.aten.addmm.default(arg153_1, view_198, permute_99);  arg153_1 = view_198 = permute_99 = None
        view_199: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_54, [1, 512, 768]);  addmm_54 = None
        view_200: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_199, [1, 512, -1, 64]);  view_199 = None
        permute_100: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_272: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_100, torch.float32);  permute_100 = None
        mul_83: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_272, 0.3535533905932738);  convert_element_type_272 = None
        expand_39: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_83, [1, 12, 512, 64]);  mul_83 = None
        view_207: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_39, [12, 512, 64]);  expand_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_201: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_262, [512, 768])
        permute_101: "f16[768, 768]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_55: "f16[512, 768]" = torch.ops.aten.addmm.default(arg155_1, view_201, permute_101);  arg155_1 = view_201 = permute_101 = None
        view_202: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_55, [1, 512, 768]);  addmm_55 = None
        view_203: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_202, [1, 512, -1, 64]);  view_202 = None
        permute_102: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_273: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_102, torch.float32);  permute_102 = None
        permute_105: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_273, [0, 1, 3, 2]);  convert_element_type_273 = None
        mul_84: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_105, 0.3535533905932738);  permute_105 = None
        expand_40: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_84, [1, 12, 64, 512]);  mul_84 = None
        view_208: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_40, [12, 64, 512]);  expand_40 = None
        bmm_18: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_207, view_208);  view_207 = view_208 = None
        view_209: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_18, [1, 12, 512, 512]);  bmm_18 = None
        full_default_28: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_28, full_default_27);  full_default_28 = full_default_27 = None
        add_78: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_209, where_18);  view_209 = where_18 = None
        eq_9: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_78, -inf)
        logical_not_18: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        full_default_29: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_9: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
        sub_28: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_78, amax_9);  add_78 = amax_9 = None
        exp_9: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        where_19: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_19, full_default_29, div_9);  logical_not_19 = full_default_29 = div_9 = None
        expand_41: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_19, [1, 12, 512, 512]);  where_19 = None
        view_210: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_41, [12, 512, 512]);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_204: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_262, [512, 768])
        permute_103: "f16[768, 768]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_56: "f16[512, 768]" = torch.ops.aten.addmm.default(arg157_1, view_204, permute_103);  arg157_1 = view_204 = permute_103 = None
        view_205: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_56, [1, 512, 768]);  addmm_56 = None
        view_206: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_205, [1, 512, -1, 64]);  view_205 = None
        permute_104: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_274: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_104, torch.float32);  permute_104 = None
        expand_42: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_274, [1, 12, 512, 64]);  convert_element_type_274 = None
        view_211: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_42, [12, 512, 64]);  expand_42 = None
        bmm_19: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_210, view_211);  view_210 = view_211 = None
        view_212: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 12, 512, 64]);  bmm_19 = None
        convert_element_type_276: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_212, torch.float16);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_276, [0, 2, 1, 3]);  convert_element_type_276 = None
        clone_28: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_213: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_28, [1, 512, -1]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "f16[512, 768]" = torch.ops.aten.reshape.default(view_213, [512, 768]);  view_213 = None
        permute_107: "f16[768, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_57: "f16[512, 768]" = torch.ops.aten.addmm.default(arg159_1, view_214, permute_107);  arg159_1 = view_214 = permute_107 = None
        view_215: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_57, [1, 512, 768]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_79: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_215, convert_element_type_262);  view_215 = convert_element_type_262 = None
        convert_element_type_280: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32);  add_79 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_280, [2], correction = 0, keepdim = True)
        getitem_38: "f32[1, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_280, getitem_39);  convert_element_type_280 = getitem_39 = None
        add_80: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_85: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_86: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg160_1);  mul_85 = arg160_1 = None
        add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, arg161_1);  mul_86 = arg161_1 = None
        convert_element_type_281: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_81, torch.float16);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_281, [512, 768])
        permute_108: "f16[768, 3072]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_58: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg163_1, view_216, permute_108);  arg163_1 = view_216 = permute_108 = None
        view_217: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [1, 512, 3072]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_285: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_87: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_285, 0.5)
        mul_88: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_285, 0.7071067811865476);  convert_element_type_285 = None
        erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_82: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_89: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_87, add_82);  mul_87 = add_82 = None
        convert_element_type_286: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_89, torch.float16);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_286, [512, 3072]);  convert_element_type_286 = None
        permute_109: "f16[3072, 768]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_59: "f16[512, 768]" = torch.ops.aten.addmm.default(arg165_1, view_218, permute_109);  arg165_1 = view_218 = permute_109 = None
        view_219: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_59, [1, 512, 768]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_219, convert_element_type_281);  view_219 = convert_element_type_281 = None
        convert_element_type_290: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32);  add_83 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_290, [2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_290, getitem_41);  convert_element_type_290 = getitem_41 = None
        add_84: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_90: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_91: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg166_1);  mul_90 = arg166_1 = None
        add_85: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_91, arg167_1);  mul_91 = arg167_1 = None
        convert_element_type_291: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_85, torch.float16);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_220: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_291, [512, 768])
        permute_110: "f16[768, 768]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_60: "f16[512, 768]" = torch.ops.aten.addmm.default(arg169_1, view_220, permute_110);  arg169_1 = view_220 = permute_110 = None
        view_221: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_60, [1, 512, 768]);  addmm_60 = None
        view_222: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_221, [1, 512, -1, 64]);  view_221 = None
        permute_111: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_301: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_111, torch.float32);  permute_111 = None
        mul_92: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_301, 0.3535533905932738);  convert_element_type_301 = None
        expand_43: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_92, [1, 12, 512, 64]);  mul_92 = None
        view_229: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_43, [12, 512, 64]);  expand_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_223: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_291, [512, 768])
        permute_112: "f16[768, 768]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_61: "f16[512, 768]" = torch.ops.aten.addmm.default(arg171_1, view_223, permute_112);  arg171_1 = view_223 = permute_112 = None
        view_224: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_61, [1, 512, 768]);  addmm_61 = None
        view_225: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_224, [1, 512, -1, 64]);  view_224 = None
        permute_113: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_302: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_113, torch.float32);  permute_113 = None
        permute_116: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_302, [0, 1, 3, 2]);  convert_element_type_302 = None
        mul_93: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_116, 0.3535533905932738);  permute_116 = None
        expand_44: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_93, [1, 12, 64, 512]);  mul_93 = None
        view_230: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_44, [12, 64, 512]);  expand_44 = None
        bmm_20: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_229, view_230);  view_229 = view_230 = None
        view_231: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [1, 12, 512, 512]);  bmm_20 = None
        full_default_31: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_31, full_default_30);  full_default_31 = full_default_30 = None
        add_86: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_231, where_20);  view_231 = where_20 = None
        eq_10: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_86, -inf)
        logical_not_20: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        full_default_32: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_10: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_86, [-1], True)
        sub_31: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_86, amax_10);  add_86 = amax_10 = None
        exp_10: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        where_21: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_21, full_default_32, div_10);  logical_not_21 = full_default_32 = div_10 = None
        expand_45: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_21, [1, 12, 512, 512]);  where_21 = None
        view_232: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_45, [12, 512, 512]);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_226: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_291, [512, 768])
        permute_114: "f16[768, 768]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_62: "f16[512, 768]" = torch.ops.aten.addmm.default(arg173_1, view_226, permute_114);  arg173_1 = view_226 = permute_114 = None
        view_227: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_62, [1, 512, 768]);  addmm_62 = None
        view_228: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_227, [1, 512, -1, 64]);  view_227 = None
        permute_115: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_303: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_115, torch.float32);  permute_115 = None
        expand_46: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_303, [1, 12, 512, 64]);  convert_element_type_303 = None
        view_233: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_46, [12, 512, 64]);  expand_46 = None
        bmm_21: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_232, view_233);  view_232 = view_233 = None
        view_234: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 12, 512, 64]);  bmm_21 = None
        convert_element_type_305: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_234, torch.float16);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_305, [0, 2, 1, 3]);  convert_element_type_305 = None
        clone_31: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_31, [1, 512, -1]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "f16[512, 768]" = torch.ops.aten.reshape.default(view_235, [512, 768]);  view_235 = None
        permute_118: "f16[768, 768]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        addmm_63: "f16[512, 768]" = torch.ops.aten.addmm.default(arg175_1, view_236, permute_118);  arg175_1 = view_236 = permute_118 = None
        view_237: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_63, [1, 512, 768]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_237, convert_element_type_291);  view_237 = convert_element_type_291 = None
        convert_element_type_309: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_87, torch.float32);  add_87 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_309, [2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_309, getitem_43);  convert_element_type_309 = getitem_43 = None
        add_88: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_94: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_95: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_94, arg176_1);  mul_94 = arg176_1 = None
        add_89: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_95, arg177_1);  mul_95 = arg177_1 = None
        convert_element_type_310: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_89, torch.float16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_310, [512, 768])
        permute_119: "f16[768, 3072]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_64: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg179_1, view_238, permute_119);  arg179_1 = view_238 = permute_119 = None
        view_239: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [1, 512, 3072]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_314: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_96: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_314, 0.5)
        mul_97: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_314, 0.7071067811865476);  convert_element_type_314 = None
        erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_90: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_98: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_96, add_90);  mul_96 = add_90 = None
        convert_element_type_315: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_98, torch.float16);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_315, [512, 3072]);  convert_element_type_315 = None
        permute_120: "f16[3072, 768]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_65: "f16[512, 768]" = torch.ops.aten.addmm.default(arg181_1, view_240, permute_120);  arg181_1 = view_240 = permute_120 = None
        view_241: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_65, [1, 512, 768]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_91: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_241, convert_element_type_310);  view_241 = convert_element_type_310 = None
        convert_element_type_319: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_91, torch.float32);  add_91 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_319, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_319, getitem_45);  convert_element_type_319 = getitem_45 = None
        add_92: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, arg182_1);  mul_99 = arg182_1 = None
        add_93: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_100, arg183_1);  mul_100 = arg183_1 = None
        convert_element_type_320: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_93, torch.float16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_242: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_320, [512, 768])
        permute_121: "f16[768, 768]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_66: "f16[512, 768]" = torch.ops.aten.addmm.default(arg185_1, view_242, permute_121);  arg185_1 = view_242 = permute_121 = None
        view_243: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_66, [1, 512, 768]);  addmm_66 = None
        view_244: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_243, [1, 512, -1, 64]);  view_243 = None
        permute_122: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_330: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_122, torch.float32);  permute_122 = None
        mul_101: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_330, 0.3535533905932738);  convert_element_type_330 = None
        expand_47: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_101, [1, 12, 512, 64]);  mul_101 = None
        view_251: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_47, [12, 512, 64]);  expand_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_245: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_320, [512, 768])
        permute_123: "f16[768, 768]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_67: "f16[512, 768]" = torch.ops.aten.addmm.default(arg187_1, view_245, permute_123);  arg187_1 = view_245 = permute_123 = None
        view_246: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, [1, 512, 768]);  addmm_67 = None
        view_247: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_246, [1, 512, -1, 64]);  view_246 = None
        permute_124: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_331: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_124, torch.float32);  permute_124 = None
        permute_127: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_331, [0, 1, 3, 2]);  convert_element_type_331 = None
        mul_102: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_127, 0.3535533905932738);  permute_127 = None
        expand_48: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_102, [1, 12, 64, 512]);  mul_102 = None
        view_252: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_48, [12, 64, 512]);  expand_48 = None
        bmm_22: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_251, view_252);  view_251 = view_252 = None
        view_253: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [1, 12, 512, 512]);  bmm_22 = None
        full_default_34: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_34, full_default_33);  expand_2 = full_default_34 = full_default_33 = None
        add_94: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_253, where_22);  view_253 = where_22 = None
        eq_11: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_94, -inf)
        logical_not_22: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        full_default_35: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_11: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_94, [-1], True)
        sub_34: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_94, amax_11);  add_94 = amax_11 = None
        exp_11: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        where_23: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_23, full_default_35, div_11);  logical_not_23 = full_default_35 = div_11 = None
        expand_49: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_23, [1, 12, 512, 512]);  where_23 = None
        view_254: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_49, [12, 512, 512]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_248: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_320, [512, 768])
        permute_125: "f16[768, 768]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_68: "f16[512, 768]" = torch.ops.aten.addmm.default(arg189_1, view_248, permute_125);  arg189_1 = view_248 = permute_125 = None
        view_249: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_68, [1, 512, 768]);  addmm_68 = None
        view_250: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_249, [1, 512, -1, 64]);  view_249 = None
        permute_126: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_332: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_126, torch.float32);  permute_126 = None
        expand_50: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_332, [1, 12, 512, 64]);  convert_element_type_332 = None
        view_255: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_50, [12, 512, 64]);  expand_50 = None
        bmm_23: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_254, view_255);  view_254 = view_255 = None
        view_256: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 12, 512, 64]);  bmm_23 = None
        convert_element_type_334: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_256, torch.float16);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_334, [0, 2, 1, 3]);  convert_element_type_334 = None
        clone_34: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_257: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_34, [1, 512, -1]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_258: "f16[512, 768]" = torch.ops.aten.reshape.default(view_257, [512, 768]);  view_257 = None
        permute_129: "f16[768, 768]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_69: "f16[512, 768]" = torch.ops.aten.addmm.default(arg191_1, view_258, permute_129);  arg191_1 = view_258 = permute_129 = None
        view_259: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_69, [1, 512, 768]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_95: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_259, convert_element_type_320);  view_259 = convert_element_type_320 = None
        convert_element_type_338: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_95, torch.float32);  add_95 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_338, [2], correction = 0, keepdim = True)
        getitem_46: "f32[1, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_338, getitem_47);  convert_element_type_338 = getitem_47 = None
        add_96: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_104: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_103, arg192_1);  mul_103 = arg192_1 = None
        add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_104, arg193_1);  mul_104 = arg193_1 = None
        convert_element_type_339: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_97, torch.float16);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_339, [512, 768])
        permute_130: "f16[768, 3072]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_70: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg195_1, view_260, permute_130);  arg195_1 = view_260 = permute_130 = None
        view_261: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [1, 512, 3072]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_343: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_105: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_343, 0.5)
        mul_106: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_343, 0.7071067811865476);  convert_element_type_343 = None
        erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
        add_98: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_107: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_105, add_98);  mul_105 = add_98 = None
        convert_element_type_344: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_107, torch.float16);  mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_344, [512, 3072]);  convert_element_type_344 = None
        permute_131: "f16[3072, 768]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        addmm_71: "f16[512, 768]" = torch.ops.aten.addmm.default(arg197_1, view_262, permute_131);  arg197_1 = view_262 = permute_131 = None
        view_263: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_71, [1, 512, 768]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_99: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_263, convert_element_type_339);  view_263 = convert_element_type_339 = None
        convert_element_type_348: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_99, torch.float32);  add_99 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_348, [2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_348, getitem_49);  convert_element_type_348 = getitem_49 = None
        add_100: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_108: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_109: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, arg198_1);  mul_108 = arg198_1 = None
        add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_109, arg199_1);  mul_109 = arg199_1 = None
        convert_element_type_349: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_101, torch.float16);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        view_264: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_349, [512, 768]);  convert_element_type_349 = None
        permute_132: "f16[768, 768]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_72: "f16[512, 768]" = torch.ops.aten.addmm.default(arg201_1, view_264, permute_132);  arg201_1 = view_264 = permute_132 = None
        view_265: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_72, [1, 512, 768]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_353: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_353, 0.5)
        mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_353, 0.7071067811865476);  convert_element_type_353 = None
        erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_102: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, add_102);  mul_110 = add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_default: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_112, torch.float32);  mul_112 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem_50: "f32[1, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_37: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_51);  convert_element_type_default = getitem_51 = None
        add_103: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = rsqrt_25 = None
        mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, arg202_1);  mul_113 = arg202_1 = None
        add_104: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_114, arg203_1);  mul_114 = arg203_1 = None
        convert_element_type_356: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_104, torch.float16);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        view_266: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_356, [512, 768]);  convert_element_type_356 = None
        permute_133: "f16[768, 30522]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_73: "f16[512, 30522]" = torch.ops.aten.addmm.default(arg204_1, view_266, permute_133);  arg204_1 = view_266 = permute_133 = None
        view_267: "f16[1, 512, 30522]" = torch.ops.aten.reshape.default(addmm_73, [1, 512, 30522]);  addmm_73 = None
        return (view_267,)
