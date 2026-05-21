class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "i64[1, 512]", arg2_1: "i64[1, 512]", arg3_1: "f16[30000, 128]", arg4_1: "f16[2, 128]", arg5_1: "f16[512, 128]", arg6_1: "f16[128]", arg7_1: "f16[128]", arg8_1: "f16[768, 128]", arg9_1: "f16[768]", arg10_1: "f16[768, 768]", arg11_1: "f16[768]", arg12_1: "f16[768, 768]", arg13_1: "f16[768]", arg14_1: "f16[768, 768]", arg15_1: "f16[768]", arg16_1: "f16[768, 768]", arg17_1: "f16[768]", arg18_1: "f16[768]", arg19_1: "f16[768]", arg20_1: "f16[3072, 768]", arg21_1: "f16[3072]", arg22_1: "f16[768, 3072]", arg23_1: "f16[768]", arg24_1: "f16[768]", arg25_1: "f16[768]", arg26_1: "f16[128, 768]", arg27_1: "f16[128]", arg28_1: "f16[128]", arg29_1: "f16[128]", arg30_1: "f16[30000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:101 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f16[1, 512, 128]" = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:94 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand: "i64[1, 512]" = torch.ops.aten.expand.default(arg2_1, [1, -1]);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:95 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather: "i64[1, 512]" = torch.ops.aten.gather.default(expand, 1, arg1_1);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:96 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[1, 512]" = torch.ops.aten.expand.default(gather, [1, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:102 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f16[1, 512, 128]" = torch.ops.aten.embedding.default(arg4_1, expand_1);  arg4_1 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:103 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f16[1, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:105 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f16[1, 512, 128]" = torch.ops.aten.embedding.default(arg5_1, arg1_1);  arg5_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:106 in forward, code: embeddings = embeddings + position_embeddings
        add_1: "f16[1, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:108 in forward, code: embeddings = self.LayerNorm(embeddings)
        convert_element_type: "f32[1, 512, 128]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean[0]
        getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        convert_element_type_1: "f16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(add_3, torch.float16);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:276 in forward, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        view: "f16[512, 128]" = torch.ops.aten.reshape.default(convert_element_type_1, [512, 128]);  convert_element_type_1 = None
        permute: "f16[128, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "f16[512, 768]" = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm, [1, 512, 768]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_2: "f16[512, 768]" = torch.ops.aten.reshape.default(view_1, [512, 768])
        permute_1: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_1: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_2, permute_1);  view_2 = permute_1 = None
        view_3: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 768]);  addmm_1 = None
        view_4: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_3, [1, 512, -1, 64]);  view_3 = None
        permute_2: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_14: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_2, torch.float32);  permute_2 = None
        mul_2: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_14, 0.3535533905932738);  convert_element_type_14 = None
        expand_3: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_2, [1, 12, 512, 64]);  mul_2 = None
        view_11: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_3, [12, 512, 64]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_5: "f16[512, 768]" = torch.ops.aten.reshape.default(view_1, [512, 768])
        permute_3: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_2: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_5, permute_3);  view_5 = permute_3 = None
        view_6: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [1, 512, 768]);  addmm_2 = None
        view_7: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_6, [1, 512, -1, 64]);  view_6 = None
        permute_4: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_15: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None
        permute_7: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_15, [0, 1, 3, 2]);  convert_element_type_15 = None
        mul_3: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_7, 0.3535533905932738);  permute_7 = None
        expand_4: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_3, [1, 12, 64, 512]);  mul_3 = None
        view_12: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_4, [12, 64, 512]);  expand_4 = None
        bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_11, view_12);  view_11 = view_12 = None
        view_13: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [1, 12, 512, 512]);  bmm = None

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
        add_6: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_13, where);  view_13 = where = None
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
        view_14: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_5, [12, 512, 512]);  expand_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_8: "f16[512, 768]" = torch.ops.aten.reshape.default(view_1, [512, 768])
        permute_5: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_3: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_8, permute_5);  view_8 = permute_5 = None
        view_9: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 768]);  addmm_3 = None
        view_10: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_9, [1, 512, -1, 64]);  view_9 = None
        permute_6: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_16: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_6, torch.float32);  permute_6 = None
        expand_6: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_16, [1, 12, 512, 64]);  convert_element_type_16 = None
        view_15: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_6, [12, 512, 64]);  expand_6 = None
        bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_14, view_15);  view_14 = view_15 = None
        view_16: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
        convert_element_type_18: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_16, torch.float16);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_18, [0, 2, 1, 3]);  convert_element_type_18 = None
        clone_1: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_17: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_1, [1, 512, -1]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_18: "f16[512, 768]" = torch.ops.aten.reshape.default(view_17, [512, 768]);  view_17 = None
        permute_9: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_4: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_18, permute_9);  view_18 = permute_9 = None
        view_19: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 768]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_7: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_1, view_19);  view_1 = view_19 = None
        convert_element_type_22: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_22, [2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_22, getitem_3);  convert_element_type_22 = getitem_3 = None
        add_8: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg18_1);  mul_4 = None
        add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, arg19_1);  mul_5 = None
        convert_element_type_23: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_9, torch.float16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_20: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_23, [512, 768])
        permute_10: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_5: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_20, permute_10);  view_20 = permute_10 = None
        view_21: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 3072]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
        pow_1: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 3.0)
        mul_7: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_10: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_21, mul_7);  view_21 = mul_7 = None
        mul_8: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_10, 0.7978845608028654);  add_10 = None
        tanh: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_11: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_9: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_11);  mul_6 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_22: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_9, [512, 3072]);  mul_9 = None
        permute_11: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_6: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_22, permute_11);  view_22 = permute_11 = None
        view_23: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [1, 512, 768]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_12: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_23, convert_element_type_23);  view_23 = convert_element_type_23 = None
        convert_element_type_30: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_12, torch.float32);  add_12 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_30, [2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_30, getitem_5);  convert_element_type_30 = getitem_5 = None
        add_13: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg24_1);  mul_10 = None
        add_14: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, arg25_1);  mul_11 = None
        convert_element_type_31: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_14, torch.float16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_24: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_31, [512, 768])
        permute_12: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_7: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_24, permute_12);  view_24 = permute_12 = None
        view_25: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 768]);  addmm_7 = None
        view_26: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_25, [1, 512, -1, 64]);  view_25 = None
        permute_13: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_41: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_13, torch.float32);  permute_13 = None
        mul_12: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_41, 0.3535533905932738);  convert_element_type_41 = None
        expand_7: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_12, [1, 12, 512, 64]);  mul_12 = None
        view_33: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_7, [12, 512, 64]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_27: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_31, [512, 768])
        permute_14: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_8: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_27, permute_14);  view_27 = permute_14 = None
        view_28: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [1, 512, 768]);  addmm_8 = None
        view_29: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_28, [1, 512, -1, 64]);  view_28 = None
        permute_15: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_42: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_15, torch.float32);  permute_15 = None
        permute_18: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_42, [0, 1, 3, 2]);  convert_element_type_42 = None
        mul_13: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_18, 0.3535533905932738);  permute_18 = None
        expand_8: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_13, [1, 12, 64, 512]);  mul_13 = None
        view_34: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_8, [12, 64, 512]);  expand_8 = None
        bmm_2: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_33, view_34);  view_33 = view_34 = None
        view_35: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 512, 512]);  bmm_2 = None
        full_default_4: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        add_15: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_35, where_2);  view_35 = where_2 = None
        eq_1: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_15, -inf)
        logical_not_2: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_default_5: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_15, [-1], True)
        sub_4: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_15, amax_1);  add_15 = amax_1 = None
        exp_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        where_3: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full_default_5, div_1);  logical_not_3 = full_default_5 = div_1 = None
        expand_9: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_3, [1, 12, 512, 512]);  where_3 = None
        view_36: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_9, [12, 512, 512]);  expand_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_30: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_31, [512, 768])
        permute_16: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_9: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_30, permute_16);  view_30 = permute_16 = None
        view_31: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [1, 512, 768]);  addmm_9 = None
        view_32: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_31, [1, 512, -1, 64]);  view_31 = None
        permute_17: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_43: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_17, torch.float32);  permute_17 = None
        expand_10: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_43, [1, 12, 512, 64]);  convert_element_type_43 = None
        view_37: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_10, [12, 512, 64]);  expand_10 = None
        bmm_3: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
        view_38: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 512, 64]);  bmm_3 = None
        convert_element_type_45: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_38, torch.float16);  view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_45, [0, 2, 1, 3]);  convert_element_type_45 = None
        clone_3: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_39: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_3, [1, 512, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_40: "f16[512, 768]" = torch.ops.aten.reshape.default(view_39, [512, 768]);  view_39 = None
        permute_20: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_10: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_40, permute_20);  view_40 = permute_20 = None
        view_41: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 768]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_16: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_31, view_41);  convert_element_type_31 = view_41 = None
        convert_element_type_49: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_49, [2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_49, getitem_7);  convert_element_type_49 = getitem_7 = None
        add_17: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_14: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_15: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg18_1);  mul_14 = None
        add_18: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_15, arg19_1);  mul_15 = None
        convert_element_type_50: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_18, torch.float16);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_42: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_50, [512, 768])
        permute_21: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_11: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_42, permute_21);  view_42 = permute_21 = None
        view_43: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 3072]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_16: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
        pow_2: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
        mul_17: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_19: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_43, mul_17);  view_43 = mul_17 = None
        mul_18: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_19, 0.7978845608028654);  add_19 = None
        tanh_1: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_20: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_19: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_16, add_20);  mul_16 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_44: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_19, [512, 3072]);  mul_19 = None
        permute_22: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_12: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_44, permute_22);  view_44 = permute_22 = None
        view_45: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [1, 512, 768]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_21: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_45, convert_element_type_50);  view_45 = convert_element_type_50 = None
        convert_element_type_57: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_21, torch.float32);  add_21 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_57, [2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_57, getitem_9);  convert_element_type_57 = getitem_9 = None
        add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_20: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_21: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg24_1);  mul_20 = None
        add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_21, arg25_1);  mul_21 = None
        convert_element_type_58: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_23, torch.float16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_46: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_58, [512, 768])
        permute_23: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_13: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_46, permute_23);  view_46 = permute_23 = None
        view_47: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 768]);  addmm_13 = None
        view_48: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_47, [1, 512, -1, 64]);  view_47 = None
        permute_24: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_68: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_24, torch.float32);  permute_24 = None
        mul_22: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_68, 0.3535533905932738);  convert_element_type_68 = None
        expand_11: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_22, [1, 12, 512, 64]);  mul_22 = None
        view_55: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_11, [12, 512, 64]);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_49: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_58, [512, 768])
        permute_25: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_14: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_49, permute_25);  view_49 = permute_25 = None
        view_50: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [1, 512, 768]);  addmm_14 = None
        view_51: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_50, [1, 512, -1, 64]);  view_50 = None
        permute_26: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_69: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_26, torch.float32);  permute_26 = None
        permute_29: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_69, [0, 1, 3, 2]);  convert_element_type_69 = None
        mul_23: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_29, 0.3535533905932738);  permute_29 = None
        expand_12: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_23, [1, 12, 64, 512]);  mul_23 = None
        view_56: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_12, [12, 64, 512]);  expand_12 = None
        bmm_4: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
        view_57: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 512, 512]);  bmm_4 = None
        full_default_7: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        add_24: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_57, where_4);  view_57 = where_4 = None
        eq_2: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_24, -inf)
        logical_not_4: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_default_8: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_24, [-1], True)
        sub_7: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_24, amax_2);  add_24 = amax_2 = None
        exp_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        where_5: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full_default_8, div_2);  logical_not_5 = full_default_8 = div_2 = None
        expand_13: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_5, [1, 12, 512, 512]);  where_5 = None
        view_58: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_13, [12, 512, 512]);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_52: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_58, [512, 768])
        permute_27: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_15: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_52, permute_27);  view_52 = permute_27 = None
        view_53: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [1, 512, 768]);  addmm_15 = None
        view_54: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_53, [1, 512, -1, 64]);  view_53 = None
        permute_28: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_70: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_28, torch.float32);  permute_28 = None
        expand_14: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_70, [1, 12, 512, 64]);  convert_element_type_70 = None
        view_59: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_14, [12, 512, 64]);  expand_14 = None
        bmm_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
        view_60: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 512, 64]);  bmm_5 = None
        convert_element_type_72: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_60, torch.float16);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_72, [0, 2, 1, 3]);  convert_element_type_72 = None
        clone_5: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_61: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_5, [1, 512, -1]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_62: "f16[512, 768]" = torch.ops.aten.reshape.default(view_61, [512, 768]);  view_61 = None
        permute_31: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_16: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_62, permute_31);  view_62 = permute_31 = None
        view_63: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 768]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_25: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_58, view_63);  convert_element_type_58 = view_63 = None
        convert_element_type_76: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_76, [2], correction = 0, keepdim = True)
        getitem_10: "f32[1, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_76, getitem_11);  convert_element_type_76 = getitem_11 = None
        add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg18_1);  mul_24 = None
        add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, arg19_1);  mul_25 = None
        convert_element_type_77: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_27, torch.float16);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_64: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_77, [512, 768])
        permute_32: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_17: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_64, permute_32);  view_64 = permute_32 = None
        view_65: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 3072]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_26: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
        pow_3: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
        mul_27: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_28: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_65, mul_27);  view_65 = mul_27 = None
        mul_28: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_28, 0.7978845608028654);  add_28 = None
        tanh_2: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_29: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_29: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_29);  mul_26 = add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_66: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_29, [512, 3072]);  mul_29 = None
        permute_33: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_18: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_66, permute_33);  view_66 = permute_33 = None
        view_67: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [1, 512, 768]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_30: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_67, convert_element_type_77);  view_67 = convert_element_type_77 = None
        convert_element_type_84: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_84, [2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_84, getitem_13);  convert_element_type_84 = getitem_13 = None
        add_31: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_30: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_31: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg24_1);  mul_30 = None
        add_32: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_31, arg25_1);  mul_31 = None
        convert_element_type_85: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_32, torch.float16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_68: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_85, [512, 768])
        permute_34: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_19: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_68, permute_34);  view_68 = permute_34 = None
        view_69: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 768]);  addmm_19 = None
        view_70: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_69, [1, 512, -1, 64]);  view_69 = None
        permute_35: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_95: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_35, torch.float32);  permute_35 = None
        mul_32: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_95, 0.3535533905932738);  convert_element_type_95 = None
        expand_15: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_32, [1, 12, 512, 64]);  mul_32 = None
        view_77: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_15, [12, 512, 64]);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_71: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_85, [512, 768])
        permute_36: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_20: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_71, permute_36);  view_71 = permute_36 = None
        view_72: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [1, 512, 768]);  addmm_20 = None
        view_73: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_72, [1, 512, -1, 64]);  view_72 = None
        permute_37: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_96: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_37, torch.float32);  permute_37 = None
        permute_40: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_96, [0, 1, 3, 2]);  convert_element_type_96 = None
        mul_33: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_40, 0.3535533905932738);  permute_40 = None
        expand_16: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_33, [1, 12, 64, 512]);  mul_33 = None
        view_78: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_16, [12, 64, 512]);  expand_16 = None
        bmm_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_77, view_78);  view_77 = view_78 = None
        view_79: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 512, 512]);  bmm_6 = None
        full_default_10: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        add_33: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_79, where_6);  view_79 = where_6 = None
        eq_3: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_33, -inf)
        logical_not_6: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_default_11: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_33, [-1], True)
        sub_10: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_33, amax_3);  add_33 = amax_3 = None
        exp_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        where_7: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full_default_11, div_3);  logical_not_7 = full_default_11 = div_3 = None
        expand_17: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_7, [1, 12, 512, 512]);  where_7 = None
        view_80: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_17, [12, 512, 512]);  expand_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_74: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_85, [512, 768])
        permute_38: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_21: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_74, permute_38);  view_74 = permute_38 = None
        view_75: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 768]);  addmm_21 = None
        view_76: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_75, [1, 512, -1, 64]);  view_75 = None
        permute_39: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_97: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_39, torch.float32);  permute_39 = None
        expand_18: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_97, [1, 12, 512, 64]);  convert_element_type_97 = None
        view_81: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_18, [12, 512, 64]);  expand_18 = None
        bmm_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = view_81 = None
        view_82: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 512, 64]);  bmm_7 = None
        convert_element_type_99: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_82, torch.float16);  view_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_41: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_99, [0, 2, 1, 3]);  convert_element_type_99 = None
        clone_7: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_83: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_7, [1, 512, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_84: "f16[512, 768]" = torch.ops.aten.reshape.default(view_83, [512, 768]);  view_83 = None
        permute_42: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_22: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_84, permute_42);  view_84 = permute_42 = None
        view_85: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 768]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_34: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_85, view_85);  convert_element_type_85 = view_85 = None
        convert_element_type_103: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32);  add_34 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_103, [2], correction = 0, keepdim = True)
        getitem_14: "f32[1, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_103, getitem_15);  convert_element_type_103 = getitem_15 = None
        add_35: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_35: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg18_1);  mul_34 = None
        add_36: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, arg19_1);  mul_35 = None
        convert_element_type_104: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_36, torch.float16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_86: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_104, [512, 768])
        permute_43: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_23: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_86, permute_43);  view_86 = permute_43 = None
        view_87: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 3072]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
        pow_4: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 3.0)
        mul_37: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_37: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_87, mul_37);  view_87 = mul_37 = None
        mul_38: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_37, 0.7978845608028654);  add_37 = None
        tanh_3: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_38: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_39: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_38);  mul_36 = add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_88: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_39, [512, 3072]);  mul_39 = None
        permute_44: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_24: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_88, permute_44);  view_88 = permute_44 = None
        view_89: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [1, 512, 768]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_39: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_89, convert_element_type_104);  view_89 = convert_element_type_104 = None
        convert_element_type_111: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_39, torch.float32);  add_39 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_111, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_111, getitem_17);  convert_element_type_111 = getitem_17 = None
        add_40: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg24_1);  mul_40 = None
        add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, arg25_1);  mul_41 = None
        convert_element_type_112: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_41, torch.float16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_90: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_112, [512, 768])
        permute_45: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_25: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_90, permute_45);  view_90 = permute_45 = None
        view_91: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [1, 512, 768]);  addmm_25 = None
        view_92: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_91, [1, 512, -1, 64]);  view_91 = None
        permute_46: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_122: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_46, torch.float32);  permute_46 = None
        mul_42: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_122, 0.3535533905932738);  convert_element_type_122 = None
        expand_19: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_42, [1, 12, 512, 64]);  mul_42 = None
        view_99: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_19, [12, 512, 64]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_93: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_112, [512, 768])
        permute_47: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_26: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_93, permute_47);  view_93 = permute_47 = None
        view_94: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [1, 512, 768]);  addmm_26 = None
        view_95: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_94, [1, 512, -1, 64]);  view_94 = None
        permute_48: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_123: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_48, torch.float32);  permute_48 = None
        permute_51: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_123, [0, 1, 3, 2]);  convert_element_type_123 = None
        mul_43: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_51, 0.3535533905932738);  permute_51 = None
        expand_20: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_43, [1, 12, 64, 512]);  mul_43 = None
        view_100: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_20, [12, 64, 512]);  expand_20 = None
        bmm_8: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
        view_101: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 512, 512]);  bmm_8 = None
        full_default_13: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        add_42: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_101, where_8);  view_101 = where_8 = None
        eq_4: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_42, -inf)
        logical_not_8: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_default_14: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_42, [-1], True)
        sub_13: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_42, amax_4);  add_42 = amax_4 = None
        exp_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        where_9: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full_default_14, div_4);  logical_not_9 = full_default_14 = div_4 = None
        expand_21: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_9, [1, 12, 512, 512]);  where_9 = None
        view_102: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_21, [12, 512, 512]);  expand_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_96: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_112, [512, 768])
        permute_49: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_27: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_96, permute_49);  view_96 = permute_49 = None
        view_97: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [1, 512, 768]);  addmm_27 = None
        view_98: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_97, [1, 512, -1, 64]);  view_97 = None
        permute_50: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_124: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_50, torch.float32);  permute_50 = None
        expand_22: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_124, [1, 12, 512, 64]);  convert_element_type_124 = None
        view_103: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_22, [12, 512, 64]);  expand_22 = None
        bmm_9: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = view_103 = None
        view_104: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 512, 64]);  bmm_9 = None
        convert_element_type_126: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_104, torch.float16);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_52: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_126, [0, 2, 1, 3]);  convert_element_type_126 = None
        clone_9: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_9, [1, 512, -1]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_106: "f16[512, 768]" = torch.ops.aten.reshape.default(view_105, [512, 768]);  view_105 = None
        permute_53: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_28: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_106, permute_53);  view_106 = permute_53 = None
        view_107: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 768]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_43: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_112, view_107);  convert_element_type_112 = view_107 = None
        convert_element_type_130: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.float32);  add_43 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_130, [2], correction = 0, keepdim = True)
        getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_130, getitem_19);  convert_element_type_130 = getitem_19 = None
        add_44: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_44: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_45: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg18_1);  mul_44 = None
        add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, arg19_1);  mul_45 = None
        convert_element_type_131: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_45, torch.float16);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_108: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_131, [512, 768])
        permute_54: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_29: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_108, permute_54);  view_108 = permute_54 = None
        view_109: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_29, [1, 512, 3072]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        pow_5: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
        mul_47: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_46: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_109, mul_47);  view_109 = mul_47 = None
        mul_48: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_4: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_47: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_49: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_47);  mul_46 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_110: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_49, [512, 3072]);  mul_49 = None
        permute_55: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_30: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_110, permute_55);  view_110 = permute_55 = None
        view_111: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [1, 512, 768]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_48: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_111, convert_element_type_131);  view_111 = convert_element_type_131 = None
        convert_element_type_138: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_138, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_138, getitem_21);  convert_element_type_138 = getitem_21 = None
        add_49: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg24_1);  mul_50 = None
        add_50: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, arg25_1);  mul_51 = None
        convert_element_type_139: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_50, torch.float16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_112: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_139, [512, 768])
        permute_56: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_31: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_112, permute_56);  view_112 = permute_56 = None
        view_113: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_31, [1, 512, 768]);  addmm_31 = None
        view_114: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_113, [1, 512, -1, 64]);  view_113 = None
        permute_57: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_149: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_57, torch.float32);  permute_57 = None
        mul_52: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_149, 0.3535533905932738);  convert_element_type_149 = None
        expand_23: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_52, [1, 12, 512, 64]);  mul_52 = None
        view_121: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_23, [12, 512, 64]);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_115: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_139, [512, 768])
        permute_58: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_32: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_115, permute_58);  view_115 = permute_58 = None
        view_116: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [1, 512, 768]);  addmm_32 = None
        view_117: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_116, [1, 512, -1, 64]);  view_116 = None
        permute_59: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_150: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_59, torch.float32);  permute_59 = None
        permute_62: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_150, [0, 1, 3, 2]);  convert_element_type_150 = None
        mul_53: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_62, 0.3535533905932738);  permute_62 = None
        expand_24: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_53, [1, 12, 64, 512]);  mul_53 = None
        view_122: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_24, [12, 64, 512]);  expand_24 = None
        bmm_10: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_121, view_122);  view_121 = view_122 = None
        view_123: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 512, 512]);  bmm_10 = None
        full_default_16: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        add_51: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_123, where_10);  view_123 = where_10 = None
        eq_5: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_51, -inf)
        logical_not_10: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_default_17: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_51, [-1], True)
        sub_16: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_51, amax_5);  add_51 = amax_5 = None
        exp_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        where_11: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full_default_17, div_5);  logical_not_11 = full_default_17 = div_5 = None
        expand_25: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_11, [1, 12, 512, 512]);  where_11 = None
        view_124: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_25, [12, 512, 512]);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_118: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_139, [512, 768])
        permute_60: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_33: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_118, permute_60);  view_118 = permute_60 = None
        view_119: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_33, [1, 512, 768]);  addmm_33 = None
        view_120: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_119, [1, 512, -1, 64]);  view_119 = None
        permute_61: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_151: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_61, torch.float32);  permute_61 = None
        expand_26: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_151, [1, 12, 512, 64]);  convert_element_type_151 = None
        view_125: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_26, [12, 512, 64]);  expand_26 = None
        bmm_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
        view_126: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 512, 64]);  bmm_11 = None
        convert_element_type_153: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_126, torch.float16);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_153, [0, 2, 1, 3]);  convert_element_type_153 = None
        clone_11: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_127: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_11, [1, 512, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_128: "f16[512, 768]" = torch.ops.aten.reshape.default(view_127, [512, 768]);  view_127 = None
        permute_64: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_34: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_128, permute_64);  view_128 = permute_64 = None
        view_129: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 768]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_52: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_139, view_129);  convert_element_type_139 = view_129 = None
        convert_element_type_157: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_52, torch.float32);  add_52 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_157, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_157, getitem_23);  convert_element_type_157 = getitem_23 = None
        add_53: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_54: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_55: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg18_1);  mul_54 = None
        add_54: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_55, arg19_1);  mul_55 = None
        convert_element_type_158: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_54, torch.float16);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_130: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_158, [512, 768])
        permute_65: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_35: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_130, permute_65);  view_130 = permute_65 = None
        view_131: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_35, [1, 512, 3072]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
        pow_6: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
        mul_57: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_55: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_131, mul_57);  view_131 = mul_57 = None
        mul_58: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_5: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_56: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_59: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_56, add_56);  mul_56 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_132: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_59, [512, 3072]);  mul_59 = None
        permute_66: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_36: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_132, permute_66);  view_132 = permute_66 = None
        view_133: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [1, 512, 768]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_57: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_133, convert_element_type_158);  view_133 = convert_element_type_158 = None
        convert_element_type_165: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_165, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_165, getitem_25);  convert_element_type_165 = getitem_25 = None
        add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_61: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, arg24_1);  mul_60 = None
        add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, arg25_1);  mul_61 = None
        convert_element_type_166: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.float16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_134: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_166, [512, 768])
        permute_67: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_37: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_134, permute_67);  view_134 = permute_67 = None
        view_135: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_37, [1, 512, 768]);  addmm_37 = None
        view_136: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_135, [1, 512, -1, 64]);  view_135 = None
        permute_68: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_176: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_68, torch.float32);  permute_68 = None
        mul_62: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_176, 0.3535533905932738);  convert_element_type_176 = None
        expand_27: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_62, [1, 12, 512, 64]);  mul_62 = None
        view_143: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_27, [12, 512, 64]);  expand_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_137: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_166, [512, 768])
        permute_69: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_38: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_137, permute_69);  view_137 = permute_69 = None
        view_138: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_38, [1, 512, 768]);  addmm_38 = None
        view_139: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_138, [1, 512, -1, 64]);  view_138 = None
        permute_70: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_177: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_70, torch.float32);  permute_70 = None
        permute_73: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_177, [0, 1, 3, 2]);  convert_element_type_177 = None
        mul_63: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_73, 0.3535533905932738);  permute_73 = None
        expand_28: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_63, [1, 12, 64, 512]);  mul_63 = None
        view_144: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_28, [12, 64, 512]);  expand_28 = None
        bmm_12: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_143, view_144);  view_143 = view_144 = None
        view_145: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [1, 12, 512, 512]);  bmm_12 = None
        full_default_19: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        add_60: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_145, where_12);  view_145 = where_12 = None
        eq_6: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_60, -inf)
        logical_not_12: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        full_default_20: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_6: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_60, [-1], True)
        sub_19: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_60, amax_6);  add_60 = amax_6 = None
        exp_6: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        where_13: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_13, full_default_20, div_6);  logical_not_13 = full_default_20 = div_6 = None
        expand_29: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_13, [1, 12, 512, 512]);  where_13 = None
        view_146: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_29, [12, 512, 512]);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_140: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_166, [512, 768])
        permute_71: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_39: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_140, permute_71);  view_140 = permute_71 = None
        view_141: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_39, [1, 512, 768]);  addmm_39 = None
        view_142: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_141, [1, 512, -1, 64]);  view_141 = None
        permute_72: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_178: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_72, torch.float32);  permute_72 = None
        expand_30: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_178, [1, 12, 512, 64]);  convert_element_type_178 = None
        view_147: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_30, [12, 512, 64]);  expand_30 = None
        bmm_13: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_146, view_147);  view_146 = view_147 = None
        view_148: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 12, 512, 64]);  bmm_13 = None
        convert_element_type_180: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_148, torch.float16);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_74: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_180, [0, 2, 1, 3]);  convert_element_type_180 = None
        clone_13: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_149: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_13, [1, 512, -1]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_150: "f16[512, 768]" = torch.ops.aten.reshape.default(view_149, [512, 768]);  view_149 = None
        permute_75: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_40: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_150, permute_75);  view_150 = permute_75 = None
        view_151: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_40, [1, 512, 768]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_61: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_166, view_151);  convert_element_type_166 = view_151 = None
        convert_element_type_184: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_61, torch.float32);  add_61 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_184, [2], correction = 0, keepdim = True)
        getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_184, getitem_27);  convert_element_type_184 = getitem_27 = None
        add_62: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_64: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg18_1);  mul_64 = None
        add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, arg19_1);  mul_65 = None
        convert_element_type_185: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_63, torch.float16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_152: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_185, [512, 768])
        permute_76: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_41: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_152, permute_76);  view_152 = permute_76 = None
        view_153: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_41, [1, 512, 3072]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_66: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
        pow_7: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 3.0)
        mul_67: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_64: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_153, mul_67);  view_153 = mul_67 = None
        mul_68: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_64, 0.7978845608028654);  add_64 = None
        tanh_6: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_65: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_69: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_66, add_65);  mul_66 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_154: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_69, [512, 3072]);  mul_69 = None
        permute_77: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_42: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_154, permute_77);  view_154 = permute_77 = None
        view_155: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_42, [1, 512, 768]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_66: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_155, convert_element_type_185);  view_155 = convert_element_type_185 = None
        convert_element_type_192: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_66, torch.float32);  add_66 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_192, [2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_192, getitem_29);  convert_element_type_192 = getitem_29 = None
        add_67: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_70: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_71: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg24_1);  mul_70 = None
        add_68: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_71, arg25_1);  mul_71 = None
        convert_element_type_193: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_68, torch.float16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_156: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_193, [512, 768])
        permute_78: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_43: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_156, permute_78);  view_156 = permute_78 = None
        view_157: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_43, [1, 512, 768]);  addmm_43 = None
        view_158: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_157, [1, 512, -1, 64]);  view_157 = None
        permute_79: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_203: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_79, torch.float32);  permute_79 = None
        mul_72: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_203, 0.3535533905932738);  convert_element_type_203 = None
        expand_31: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_72, [1, 12, 512, 64]);  mul_72 = None
        view_165: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_31, [12, 512, 64]);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_159: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_193, [512, 768])
        permute_80: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_44: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_159, permute_80);  view_159 = permute_80 = None
        view_160: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_44, [1, 512, 768]);  addmm_44 = None
        view_161: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_160, [1, 512, -1, 64]);  view_160 = None
        permute_81: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_204: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_81, torch.float32);  permute_81 = None
        permute_84: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_204, [0, 1, 3, 2]);  convert_element_type_204 = None
        mul_73: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_84, 0.3535533905932738);  permute_84 = None
        expand_32: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_73, [1, 12, 64, 512]);  mul_73 = None
        view_166: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_32, [12, 64, 512]);  expand_32 = None
        bmm_14: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_165, view_166);  view_165 = view_166 = None
        view_167: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_14, [1, 12, 512, 512]);  bmm_14 = None
        full_default_22: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        add_69: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_167, where_14);  view_167 = where_14 = None
        eq_7: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_69, -inf)
        logical_not_14: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        full_default_23: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_7: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_69, [-1], True)
        sub_22: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_69, amax_7);  add_69 = amax_7 = None
        exp_7: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        where_15: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_15, full_default_23, div_7);  logical_not_15 = full_default_23 = div_7 = None
        expand_33: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_15, [1, 12, 512, 512]);  where_15 = None
        view_168: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_33, [12, 512, 512]);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_162: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_193, [512, 768])
        permute_82: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_45: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_162, permute_82);  view_162 = permute_82 = None
        view_163: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_45, [1, 512, 768]);  addmm_45 = None
        view_164: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_163, [1, 512, -1, 64]);  view_163 = None
        permute_83: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_205: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_83, torch.float32);  permute_83 = None
        expand_34: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_205, [1, 12, 512, 64]);  convert_element_type_205 = None
        view_169: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_34, [12, 512, 64]);  expand_34 = None
        bmm_15: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
        view_170: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 12, 512, 64]);  bmm_15 = None
        convert_element_type_207: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_170, torch.float16);  view_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_85: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_207, [0, 2, 1, 3]);  convert_element_type_207 = None
        clone_15: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_171: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_15, [1, 512, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_172: "f16[512, 768]" = torch.ops.aten.reshape.default(view_171, [512, 768]);  view_171 = None
        permute_86: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_46: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_172, permute_86);  view_172 = permute_86 = None
        view_173: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_46, [1, 512, 768]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_70: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_193, view_173);  convert_element_type_193 = view_173 = None
        convert_element_type_211: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_70, torch.float32);  add_70 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_211, [2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_211, getitem_31);  convert_element_type_211 = getitem_31 = None
        add_71: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_75: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg18_1);  mul_74 = None
        add_72: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, arg19_1);  mul_75 = None
        convert_element_type_212: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_72, torch.float16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_174: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_212, [512, 768])
        permute_87: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_47: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_174, permute_87);  view_174 = permute_87 = None
        view_175: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_47, [1, 512, 3072]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
        pow_8: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 3.0)
        mul_77: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_73: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_175, mul_77);  view_175 = mul_77 = None
        mul_78: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_7: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_74: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_79: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_74);  mul_76 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_176: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_79, [512, 3072]);  mul_79 = None
        permute_88: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_48: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_176, permute_88);  view_176 = permute_88 = None
        view_177: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_48, [1, 512, 768]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_75: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_177, convert_element_type_212);  view_177 = convert_element_type_212 = None
        convert_element_type_219: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_75, torch.float32);  add_75 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_219, [2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_219, getitem_33);  convert_element_type_219 = getitem_33 = None
        add_76: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_80: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg24_1);  mul_80 = None
        add_77: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, arg25_1);  mul_81 = None
        convert_element_type_220: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_77, torch.float16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_178: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_220, [512, 768])
        permute_89: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_49: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_178, permute_89);  view_178 = permute_89 = None
        view_179: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_49, [1, 512, 768]);  addmm_49 = None
        view_180: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_179, [1, 512, -1, 64]);  view_179 = None
        permute_90: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_230: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_90, torch.float32);  permute_90 = None
        mul_82: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_230, 0.3535533905932738);  convert_element_type_230 = None
        expand_35: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_82, [1, 12, 512, 64]);  mul_82 = None
        view_187: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_35, [12, 512, 64]);  expand_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_181: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_220, [512, 768])
        permute_91: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_50: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_181, permute_91);  view_181 = permute_91 = None
        view_182: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_50, [1, 512, 768]);  addmm_50 = None
        view_183: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_182, [1, 512, -1, 64]);  view_182 = None
        permute_92: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_231: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_92, torch.float32);  permute_92 = None
        permute_95: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_231, [0, 1, 3, 2]);  convert_element_type_231 = None
        mul_83: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_95, 0.3535533905932738);  permute_95 = None
        expand_36: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_83, [1, 12, 64, 512]);  mul_83 = None
        view_188: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_36, [12, 64, 512]);  expand_36 = None
        bmm_16: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_187, view_188);  view_187 = view_188 = None
        view_189: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [1, 12, 512, 512]);  bmm_16 = None
        full_default_25: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_25, full_default_24);  full_default_25 = full_default_24 = None
        add_78: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_189, where_16);  view_189 = where_16 = None
        eq_8: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_78, -inf)
        logical_not_16: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        full_default_26: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_8: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
        sub_25: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_78, amax_8);  add_78 = amax_8 = None
        exp_8: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        where_17: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_17, full_default_26, div_8);  logical_not_17 = full_default_26 = div_8 = None
        expand_37: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_17, [1, 12, 512, 512]);  where_17 = None
        view_190: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_37, [12, 512, 512]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_184: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_220, [512, 768])
        permute_93: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_51: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_184, permute_93);  view_184 = permute_93 = None
        view_185: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_51, [1, 512, 768]);  addmm_51 = None
        view_186: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_185, [1, 512, -1, 64]);  view_185 = None
        permute_94: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_232: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_94, torch.float32);  permute_94 = None
        expand_38: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_232, [1, 12, 512, 64]);  convert_element_type_232 = None
        view_191: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_38, [12, 512, 64]);  expand_38 = None
        bmm_17: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
        view_192: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 12, 512, 64]);  bmm_17 = None
        convert_element_type_234: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_192, torch.float16);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_96: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_234, [0, 2, 1, 3]);  convert_element_type_234 = None
        clone_17: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_193: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_17, [1, 512, -1]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_194: "f16[512, 768]" = torch.ops.aten.reshape.default(view_193, [512, 768]);  view_193 = None
        permute_97: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_52: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_194, permute_97);  view_194 = permute_97 = None
        view_195: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_52, [1, 512, 768]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_79: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_220, view_195);  convert_element_type_220 = view_195 = None
        convert_element_type_238: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32);  add_79 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_238, [2], correction = 0, keepdim = True)
        getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_238, getitem_35);  convert_element_type_238 = getitem_35 = None
        add_80: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_84: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_85: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, arg18_1);  mul_84 = None
        add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_85, arg19_1);  mul_85 = None
        convert_element_type_239: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_81, torch.float16);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_196: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_239, [512, 768])
        permute_98: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_53: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_196, permute_98);  view_196 = permute_98 = None
        view_197: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_53, [1, 512, 3072]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_86: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
        pow_9: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
        mul_87: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_82: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_197, mul_87);  view_197 = mul_87 = None
        mul_88: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_82, 0.7978845608028654);  add_82 = None
        tanh_8: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_83: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_89: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_86, add_83);  mul_86 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_198: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_89, [512, 3072]);  mul_89 = None
        permute_99: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_54: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_198, permute_99);  view_198 = permute_99 = None
        view_199: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_54, [1, 512, 768]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_84: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_199, convert_element_type_239);  view_199 = convert_element_type_239 = None
        convert_element_type_246: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_84, torch.float32);  add_84 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_246, [2], correction = 0, keepdim = True)
        getitem_36: "f32[1, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_27: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_246, getitem_37);  convert_element_type_246 = getitem_37 = None
        add_85: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_90: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_91: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg24_1);  mul_90 = None
        add_86: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_91, arg25_1);  mul_91 = None
        convert_element_type_247: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_86, torch.float16);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_200: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_247, [512, 768])
        permute_100: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_55: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_200, permute_100);  view_200 = permute_100 = None
        view_201: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_55, [1, 512, 768]);  addmm_55 = None
        view_202: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_201, [1, 512, -1, 64]);  view_201 = None
        permute_101: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_257: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_101, torch.float32);  permute_101 = None
        mul_92: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_257, 0.3535533905932738);  convert_element_type_257 = None
        expand_39: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_92, [1, 12, 512, 64]);  mul_92 = None
        view_209: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_39, [12, 512, 64]);  expand_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_203: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_247, [512, 768])
        permute_102: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_56: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_203, permute_102);  view_203 = permute_102 = None
        view_204: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_56, [1, 512, 768]);  addmm_56 = None
        view_205: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_204, [1, 512, -1, 64]);  view_204 = None
        permute_103: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_258: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_103, torch.float32);  permute_103 = None
        permute_106: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_258, [0, 1, 3, 2]);  convert_element_type_258 = None
        mul_93: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_106, 0.3535533905932738);  permute_106 = None
        expand_40: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_93, [1, 12, 64, 512]);  mul_93 = None
        view_210: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_40, [12, 64, 512]);  expand_40 = None
        bmm_18: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_209, view_210);  view_209 = view_210 = None
        view_211: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_18, [1, 12, 512, 512]);  bmm_18 = None
        full_default_28: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_28, full_default_27);  full_default_28 = full_default_27 = None
        add_87: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_211, where_18);  view_211 = where_18 = None
        eq_9: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_87, -inf)
        logical_not_18: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        full_default_29: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_9: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_87, [-1], True)
        sub_28: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_87, amax_9);  add_87 = amax_9 = None
        exp_9: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        where_19: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_19, full_default_29, div_9);  logical_not_19 = full_default_29 = div_9 = None
        expand_41: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_19, [1, 12, 512, 512]);  where_19 = None
        view_212: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_41, [12, 512, 512]);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_206: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_247, [512, 768])
        permute_104: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_57: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_206, permute_104);  view_206 = permute_104 = None
        view_207: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_57, [1, 512, 768]);  addmm_57 = None
        view_208: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_207, [1, 512, -1, 64]);  view_207 = None
        permute_105: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_259: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_105, torch.float32);  permute_105 = None
        expand_42: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_259, [1, 12, 512, 64]);  convert_element_type_259 = None
        view_213: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_42, [12, 512, 64]);  expand_42 = None
        bmm_19: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_212, view_213);  view_212 = view_213 = None
        view_214: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 12, 512, 64]);  bmm_19 = None
        convert_element_type_261: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_214, torch.float16);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_261, [0, 2, 1, 3]);  convert_element_type_261 = None
        clone_19: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_215: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_19, [1, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_216: "f16[512, 768]" = torch.ops.aten.reshape.default(view_215, [512, 768]);  view_215 = None
        permute_108: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_58: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_216, permute_108);  view_216 = permute_108 = None
        view_217: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_58, [1, 512, 768]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_88: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_247, view_217);  convert_element_type_247 = view_217 = None
        convert_element_type_265: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_88, torch.float32);  add_88 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_265, [2], correction = 0, keepdim = True)
        getitem_38: "f32[1, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_265, getitem_39);  convert_element_type_265 = getitem_39 = None
        add_89: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_94: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_95: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_94, arg18_1);  mul_94 = None
        add_90: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_95, arg19_1);  mul_95 = None
        convert_element_type_266: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_90, torch.float16);  add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_218: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_266, [512, 768])
        permute_109: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_59: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_218, permute_109);  view_218 = permute_109 = None
        view_219: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_59, [1, 512, 3072]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_96: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
        pow_10: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 3.0)
        mul_97: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_91: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_219, mul_97);  view_219 = mul_97 = None
        mul_98: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_91, 0.7978845608028654);  add_91 = None
        tanh_9: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_92: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_99: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_96, add_92);  mul_96 = add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_220: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_99, [512, 3072]);  mul_99 = None
        permute_110: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_60: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_220, permute_110);  view_220 = permute_110 = None
        view_221: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_60, [1, 512, 768]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_93: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_221, convert_element_type_266);  view_221 = convert_element_type_266 = None
        convert_element_type_273: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_93, torch.float32);  add_93 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_273, [2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_273, getitem_41);  convert_element_type_273 = getitem_41 = None
        add_94: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_100, arg24_1);  mul_100 = None
        add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_101, arg25_1);  mul_101 = None
        convert_element_type_274: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_95, torch.float16);  add_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_222: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_274, [512, 768])
        permute_111: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_61: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_222, permute_111);  view_222 = permute_111 = None
        view_223: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_61, [1, 512, 768]);  addmm_61 = None
        view_224: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_223, [1, 512, -1, 64]);  view_223 = None
        permute_112: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_284: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_112, torch.float32);  permute_112 = None
        mul_102: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_284, 0.3535533905932738);  convert_element_type_284 = None
        expand_43: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_102, [1, 12, 512, 64]);  mul_102 = None
        view_231: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_43, [12, 512, 64]);  expand_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_225: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_274, [512, 768])
        permute_113: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_62: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_225, permute_113);  view_225 = permute_113 = None
        view_226: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_62, [1, 512, 768]);  addmm_62 = None
        view_227: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_226, [1, 512, -1, 64]);  view_226 = None
        permute_114: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_285: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_114, torch.float32);  permute_114 = None
        permute_117: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_285, [0, 1, 3, 2]);  convert_element_type_285 = None
        mul_103: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_117, 0.3535533905932738);  permute_117 = None
        expand_44: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_103, [1, 12, 64, 512]);  mul_103 = None
        view_232: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_44, [12, 64, 512]);  expand_44 = None
        bmm_20: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_231, view_232);  view_231 = view_232 = None
        view_233: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [1, 12, 512, 512]);  bmm_20 = None
        full_default_31: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_31, full_default_30);  full_default_31 = full_default_30 = None
        add_96: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_233, where_20);  view_233 = where_20 = None
        eq_10: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_96, -inf)
        logical_not_20: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        full_default_32: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_10: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_96, [-1], True)
        sub_31: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_96, amax_10);  add_96 = amax_10 = None
        exp_10: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        where_21: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_21, full_default_32, div_10);  logical_not_21 = full_default_32 = div_10 = None
        expand_45: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_21, [1, 12, 512, 512]);  where_21 = None
        view_234: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_45, [12, 512, 512]);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_228: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_274, [512, 768])
        permute_115: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_63: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_228, permute_115);  view_228 = permute_115 = None
        view_229: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_63, [1, 512, 768]);  addmm_63 = None
        view_230: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_229, [1, 512, -1, 64]);  view_229 = None
        permute_116: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_286: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_116, torch.float32);  permute_116 = None
        expand_46: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_286, [1, 12, 512, 64]);  convert_element_type_286 = None
        view_235: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_46, [12, 512, 64]);  expand_46 = None
        bmm_21: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_234, view_235);  view_234 = view_235 = None
        view_236: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 12, 512, 64]);  bmm_21 = None
        convert_element_type_288: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_236, torch.float16);  view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_118: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_288, [0, 2, 1, 3]);  convert_element_type_288 = None
        clone_21: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_237: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_21, [1, 512, -1]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_238: "f16[512, 768]" = torch.ops.aten.reshape.default(view_237, [512, 768]);  view_237 = None
        permute_119: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_64: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_238, permute_119);  view_238 = permute_119 = None
        view_239: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_64, [1, 512, 768]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_97: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_274, view_239);  convert_element_type_274 = view_239 = None
        convert_element_type_292: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_97, torch.float32);  add_97 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_292, [2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_292, getitem_43);  convert_element_type_292 = getitem_43 = None
        add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_104: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_105: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_104, arg18_1);  mul_104 = None
        add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_105, arg19_1);  mul_105 = None
        convert_element_type_293: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_99, torch.float16);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_240: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_293, [512, 768])
        permute_120: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_65: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_240, permute_120);  view_240 = permute_120 = None
        view_241: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_65, [1, 512, 3072]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_106: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
        pow_11: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 3.0)
        mul_107: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_100: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_241, mul_107);  view_241 = mul_107 = None
        mul_108: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_101: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_109: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_106, add_101);  mul_106 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_242: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_109, [512, 3072]);  mul_109 = None
        permute_121: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_66: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_242, permute_121);  view_242 = permute_121 = None
        view_243: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_66, [1, 512, 768]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_102: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_243, convert_element_type_293);  view_243 = convert_element_type_293 = None
        convert_element_type_300: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_102, torch.float32);  add_102 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_300, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_300, getitem_45);  convert_element_type_300 = getitem_45 = None
        add_103: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, arg24_1);  mul_110 = None
        add_104: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_111, arg25_1);  mul_111 = None
        convert_element_type_301: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_104, torch.float16);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_244: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_301, [512, 768])
        permute_122: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_67: "f16[512, 768]" = torch.ops.aten.addmm.default(arg11_1, view_244, permute_122);  arg11_1 = view_244 = permute_122 = None
        view_245: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, [1, 512, 768]);  addmm_67 = None
        view_246: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_245, [1, 512, -1, 64]);  view_245 = None
        permute_123: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_311: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_123, torch.float32);  permute_123 = None
        mul_112: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_311, 0.3535533905932738);  convert_element_type_311 = None
        expand_47: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_112, [1, 12, 512, 64]);  mul_112 = None
        view_253: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_47, [12, 512, 64]);  expand_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_247: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_301, [512, 768])
        permute_124: "f16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_68: "f16[512, 768]" = torch.ops.aten.addmm.default(arg13_1, view_247, permute_124);  arg13_1 = view_247 = permute_124 = None
        view_248: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_68, [1, 512, 768]);  addmm_68 = None
        view_249: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_248, [1, 512, -1, 64]);  view_248 = None
        permute_125: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_312: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_125, torch.float32);  permute_125 = None
        permute_128: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_312, [0, 1, 3, 2]);  convert_element_type_312 = None
        mul_113: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_128, 0.3535533905932738);  permute_128 = None
        expand_48: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_113, [1, 12, 64, 512]);  mul_113 = None
        view_254: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_48, [12, 64, 512]);  expand_48 = None
        bmm_22: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_253, view_254);  view_253 = view_254 = None
        view_255: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [1, 12, 512, 512]);  bmm_22 = None
        full_default_34: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_34, full_default_33);  expand_2 = full_default_34 = full_default_33 = None
        add_105: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_255, where_22);  view_255 = where_22 = None
        eq_11: "b8[1, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_105, -inf)
        logical_not_22: "b8[1, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[1, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[1, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        full_default_35: "f32[1, 12, 512, 512]" = torch.ops.aten.full.default([1, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_11: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_105, [-1], True)
        sub_34: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_105, amax_11);  add_105 = amax_11 = None
        exp_11: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        where_23: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_23, full_default_35, div_11);  logical_not_23 = full_default_35 = div_11 = None
        expand_49: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_23, [1, 12, 512, 512]);  where_23 = None
        view_256: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_49, [12, 512, 512]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_250: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_301, [512, 768])
        permute_126: "f16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_69: "f16[512, 768]" = torch.ops.aten.addmm.default(arg15_1, view_250, permute_126);  arg15_1 = view_250 = permute_126 = None
        view_251: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_69, [1, 512, 768]);  addmm_69 = None
        view_252: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_251, [1, 512, -1, 64]);  view_251 = None
        permute_127: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_313: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_127, torch.float32);  permute_127 = None
        expand_50: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_313, [1, 12, 512, 64]);  convert_element_type_313 = None
        view_257: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_50, [12, 512, 64]);  expand_50 = None
        bmm_23: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_256, view_257);  view_256 = view_257 = None
        view_258: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 12, 512, 64]);  bmm_23 = None
        convert_element_type_315: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_258, torch.float16);  view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_129: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_315, [0, 2, 1, 3]);  convert_element_type_315 = None
        clone_23: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_259: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_23, [1, 512, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        view_260: "f16[512, 768]" = torch.ops.aten.reshape.default(view_259, [512, 768]);  view_259 = None
        permute_130: "f16[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_70: "f16[512, 768]" = torch.ops.aten.addmm.default(arg17_1, view_260, permute_130);  arg17_1 = view_260 = permute_130 = None
        view_261: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_70, [1, 512, 768]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        add_106: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_301, view_261);  convert_element_type_301 = view_261 = None
        convert_element_type_319: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_106, torch.float32);  add_106 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_319, [2], correction = 0, keepdim = True)
        getitem_46: "f32[1, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_319, getitem_47);  convert_element_type_319 = getitem_47 = None
        add_107: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_114, arg18_1);  mul_114 = arg18_1 = None
        add_108: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_115, arg19_1);  mul_115 = arg19_1 = None
        convert_element_type_320: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_108, torch.float16);  add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_262: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_320, [512, 768])
        permute_131: "f16[768, 3072]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_71: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg21_1, view_262, permute_131);  arg21_1 = view_262 = permute_131 = None
        view_263: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_71, [1, 512, 3072]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
        pow_12: "f16[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 3.0)
        mul_117: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_109: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_263, mul_117);  view_263 = mul_117 = None
        mul_118: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_109, 0.7978845608028654);  add_109 = None
        tanh_11: "f16[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_110: "f16[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_119: "f16[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_116, add_110);  mul_116 = add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_264: "f16[512, 3072]" = torch.ops.aten.reshape.default(mul_119, [512, 3072]);  mul_119 = None
        permute_132: "f16[3072, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_72: "f16[512, 768]" = torch.ops.aten.addmm.default(arg23_1, view_264, permute_132);  arg23_1 = view_264 = permute_132 = None
        view_265: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_72, [1, 512, 768]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_111: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_265, convert_element_type_320);  view_265 = convert_element_type_320 = None
        convert_element_type_327: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_111, torch.float32);  add_111 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_327, [2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_327, getitem_49);  convert_element_type_327 = getitem_49 = None
        add_112: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_120: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_120, arg24_1);  mul_120 = arg24_1 = None
        add_113: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_121, arg25_1);  mul_121 = arg25_1 = None
        convert_element_type_328: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_113, torch.float16);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:538 in forward, code: hidden_states = self.dense(hidden_states)
        view_266: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_328, [512, 768]);  convert_element_type_328 = None
        permute_133: "f16[768, 128]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_73: "f16[512, 128]" = torch.ops.aten.addmm.default(arg27_1, view_266, permute_133);  arg27_1 = view_266 = permute_133 = None
        view_267: "f16[1, 512, 128]" = torch.ops.aten.reshape.default(addmm_73, [1, 512, 128]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_122: "f16[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
        pow_13: "f16[1, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_267, 3.0)
        mul_123: "f16[1, 512, 128]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_114: "f16[1, 512, 128]" = torch.ops.aten.add.Tensor(view_267, mul_123);  view_267 = mul_123 = None
        mul_124: "f16[1, 512, 128]" = torch.ops.aten.mul.Tensor(add_114, 0.7978845608028654);  add_114 = None
        tanh_12: "f16[1, 512, 128]" = torch.ops.aten.tanh.default(mul_124);  mul_124 = None
        add_115: "f16[1, 512, 128]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_125: "f16[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_122, add_115);  mul_122 = add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:540 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_332: "f32[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_125, torch.float32);  mul_125 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_332, [2], correction = 0, keepdim = True)
        getitem_50: "f32[1, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_37: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_332, getitem_51);  convert_element_type_332 = getitem_51 = None
        add_116: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_126: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = rsqrt_25 = None
        mul_127: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_126, arg28_1);  mul_126 = arg28_1 = None
        add_117: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(mul_127, arg29_1);  mul_127 = arg29_1 = None
        convert_element_type_333: "f16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(add_117, torch.float16);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:541 in forward, code: hidden_states = self.decoder(hidden_states)
        view_268: "f16[512, 128]" = torch.ops.aten.reshape.default(convert_element_type_333, [512, 128]);  convert_element_type_333 = None
        permute_134: "f16[128, 30000]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_74: "f16[512, 30000]" = torch.ops.aten.addmm.default(arg30_1, view_268, permute_134);  arg30_1 = view_268 = permute_134 = None
        view_269: "f16[1, 512, 30000]" = torch.ops.aten.reshape.default(addmm_74, [1, 512, 30000]);  addmm_74 = None
        return (view_269,)
