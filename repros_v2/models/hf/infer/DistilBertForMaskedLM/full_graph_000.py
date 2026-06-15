class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[256, 128][128, 1]cuda:0", arg1_1: "bf16[30522, 768][768, 1]cuda:0", arg2_1: "i64[1, 512][512, 1]cuda:0", arg3_1: "bf16[512, 768][768, 1]cuda:0", arg4_1: "bf16[768][1]cuda:0", arg5_1: "bf16[768][1]cuda:0", arg6_1: "bf16[768, 768][768, 1]cuda:0", arg7_1: "bf16[768][1]cuda:0", arg8_1: "bf16[768, 768][768, 1]cuda:0", arg9_1: "bf16[768][1]cuda:0", arg10_1: "bf16[768, 768][768, 1]cuda:0", arg11_1: "bf16[768][1]cuda:0", arg12_1: "bf16[768, 768][768, 1]cuda:0", arg13_1: "bf16[768][1]cuda:0", arg14_1: "bf16[768][1]cuda:0", arg15_1: "bf16[768][1]cuda:0", arg16_1: "bf16[3072, 768][768, 1]cuda:0", arg17_1: "bf16[3072][1]cuda:0", arg18_1: "bf16[768, 3072][3072, 1]cuda:0", arg19_1: "bf16[768][1]cuda:0", arg20_1: "bf16[768][1]cuda:0", arg21_1: "bf16[768][1]cuda:0", arg22_1: "bf16[768, 768][768, 1]cuda:0", arg23_1: "bf16[768][1]cuda:0", arg24_1: "bf16[768, 768][768, 1]cuda:0", arg25_1: "bf16[768][1]cuda:0", arg26_1: "bf16[768, 768][768, 1]cuda:0", arg27_1: "bf16[768][1]cuda:0", arg28_1: "bf16[768, 768][768, 1]cuda:0", arg29_1: "bf16[768][1]cuda:0", arg30_1: "bf16[768][1]cuda:0", arg31_1: "bf16[768][1]cuda:0", arg32_1: "bf16[3072, 768][768, 1]cuda:0", arg33_1: "bf16[3072][1]cuda:0", arg34_1: "bf16[768, 3072][3072, 1]cuda:0", arg35_1: "bf16[768][1]cuda:0", arg36_1: "bf16[768][1]cuda:0", arg37_1: "bf16[768][1]cuda:0", arg38_1: "bf16[768, 768][768, 1]cuda:0", arg39_1: "bf16[768][1]cuda:0", arg40_1: "bf16[768, 768][768, 1]cuda:0", arg41_1: "bf16[768][1]cuda:0", arg42_1: "bf16[768, 768][768, 1]cuda:0", arg43_1: "bf16[768][1]cuda:0", arg44_1: "bf16[768, 768][768, 1]cuda:0", arg45_1: "bf16[768][1]cuda:0", arg46_1: "bf16[768][1]cuda:0", arg47_1: "bf16[768][1]cuda:0", arg48_1: "bf16[3072, 768][768, 1]cuda:0", arg49_1: "bf16[3072][1]cuda:0", arg50_1: "bf16[768, 3072][3072, 1]cuda:0", arg51_1: "bf16[768][1]cuda:0", arg52_1: "bf16[768][1]cuda:0", arg53_1: "bf16[768][1]cuda:0", arg54_1: "bf16[768, 768][768, 1]cuda:0", arg55_1: "bf16[768][1]cuda:0", arg56_1: "bf16[768, 768][768, 1]cuda:0", arg57_1: "bf16[768][1]cuda:0", arg58_1: "bf16[768, 768][768, 1]cuda:0", arg59_1: "bf16[768][1]cuda:0", arg60_1: "bf16[768, 768][768, 1]cuda:0", arg61_1: "bf16[768][1]cuda:0", arg62_1: "bf16[768][1]cuda:0", arg63_1: "bf16[768][1]cuda:0", arg64_1: "bf16[3072, 768][768, 1]cuda:0", arg65_1: "bf16[3072][1]cuda:0", arg66_1: "bf16[768, 3072][3072, 1]cuda:0", arg67_1: "bf16[768][1]cuda:0", arg68_1: "bf16[768][1]cuda:0", arg69_1: "bf16[768][1]cuda:0", arg70_1: "bf16[768, 768][768, 1]cuda:0", arg71_1: "bf16[768][1]cuda:0", arg72_1: "bf16[768, 768][768, 1]cuda:0", arg73_1: "bf16[768][1]cuda:0", arg74_1: "bf16[768, 768][768, 1]cuda:0", arg75_1: "bf16[768][1]cuda:0", arg76_1: "bf16[768, 768][768, 1]cuda:0", arg77_1: "bf16[768][1]cuda:0", arg78_1: "bf16[768][1]cuda:0", arg79_1: "bf16[768][1]cuda:0", arg80_1: "bf16[3072, 768][768, 1]cuda:0", arg81_1: "bf16[3072][1]cuda:0", arg82_1: "bf16[768, 3072][3072, 1]cuda:0", arg83_1: "bf16[768][1]cuda:0", arg84_1: "bf16[768][1]cuda:0", arg85_1: "bf16[768][1]cuda:0", arg86_1: "bf16[768, 768][768, 1]cuda:0", arg87_1: "bf16[768][1]cuda:0", arg88_1: "bf16[768, 768][768, 1]cuda:0", arg89_1: "bf16[768][1]cuda:0", arg90_1: "bf16[768, 768][768, 1]cuda:0", arg91_1: "bf16[768][1]cuda:0", arg92_1: "bf16[768, 768][768, 1]cuda:0", arg93_1: "bf16[768][1]cuda:0", arg94_1: "bf16[768][1]cuda:0", arg95_1: "bf16[768][1]cuda:0", arg96_1: "bf16[3072, 768][768, 1]cuda:0", arg97_1: "bf16[3072][1]cuda:0", arg98_1: "bf16[768, 3072][3072, 1]cuda:0", arg99_1: "bf16[768][1]cuda:0", arg100_1: "bf16[768][1]cuda:0", arg101_1: "bf16[768][1]cuda:0", arg102_1: "bf16[768, 768][768, 1]cuda:0", arg103_1: "bf16[768][1]cuda:0", arg104_1: "bf16[768][1]cuda:0", arg105_1: "bf16[768][1]cuda:0", arg106_1: "bf16[30522][1]cuda:0", arg107_1: "i64[256, 128][128, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        embedding: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:112 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_1: "i64[1, 128][512, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:117 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        embedding_1: "bf16[1, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg3_1, slice_1);  arg3_1 = slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:119 in forward, code: embeddings = inputs_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:120 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        convert_element_type: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_1: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[256, 1, 128, 128][0, 128, 1, 0]cuda:0" = torch.ops.aten.expand.default(ge, [256, -1, 128, 128]);  ge = expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor = None
        scalar_tensor_1: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_1 = None
        full_default: "bf16[256, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:120 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        sub: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [32768, 768])
        permute: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        view_1: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [256, 128, 768]);  addmm = None
        view_2: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [256, 128, -1, 64]);  view_1 = None
        permute_1: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_2: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        expand_1: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_2, [256, 12, 128, 64]);  mul_2 = None
        clone_1: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [3072, 128, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_3: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [32768, 768])
        permute_2: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg9_1, view_3, permute_2);  arg9_1 = view_3 = permute_2 = None
        view_4: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [256, 128, 768]);  addmm_1 = None
        view_5: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [256, 128, -1, 64]);  view_4 = None
        permute_3: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_6: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_3: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        expand_2: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_3, [256, 12, 64, 128]);  mul_3 = None
        clone_2: "bf16[256, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_10: "bf16[3072, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [3072, 64, 128]);  clone_2 = None
        bmm: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [256, 12, 128, 128]);  bmm = None
        eq: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_11, -inf)
        logical_not: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_1: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 12, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_13: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        amax: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_13, [-1], True)
        sub_1: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax);  convert_element_type_13 = amax = None
        exp: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_14: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where_1: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_1, full_default_1, convert_element_type_14);  logical_not_1 = full_default_1 = convert_element_type_14 = None
        expand_3: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_1, [256, 12, 128, 128]);  where_1 = None
        view_12: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_3, [3072, 128, 128]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_6: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [32768, 768])
        permute_4: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg11_1, view_6, permute_4);  arg11_1 = view_6 = permute_4 = None
        view_7: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [256, 128, 768]);  addmm_2 = None
        view_8: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_7, [256, 128, -1, 64]);  view_7 = None
        permute_5: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [256, 12, 128, 64]);  permute_5 = None
        clone_3: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_13: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [3072, 128, 64]);  clone_3 = None
        bmm_1: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
        view_14: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [256, 12, 128, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[256, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_4: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [256, 128, -1]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_16: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [32768, 768]);  view_15 = None
        permute_8: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg13_1, view_16, permute_8);  arg13_1 = view_16 = permute_8 = None
        view_17: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [256, 128, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_6: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_17, convert_element_type_1);  view_17 = convert_element_type_1 = None
        convert_element_type_20: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.float32);  add_6 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_20, [2], correction = 0, keepdim = True)
        getitem_2: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_3);  convert_element_type_20 = getitem_3 = None
        add_7: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_4: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, arg14_1);  mul_4 = arg14_1 = None
        add_8: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, arg15_1);  mul_5 = arg15_1 = None
        convert_element_type_21: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_18: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_21, [32768, 768])
        permute_9: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg17_1, view_18, permute_9);  arg17_1 = view_18 = permute_9 = None
        view_19: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [256, 128, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_25: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_6: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, 0.5)
        mul_7: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, 0.7071067811865476);  convert_element_type_25 = None
        erf: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_9: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, add_9);  mul_6 = add_9 = None
        convert_element_type_26: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_20: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_26, [32768, 3072]);  convert_element_type_26 = None
        permute_10: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg19_1, view_20, permute_10);  arg19_1 = view_20 = permute_10 = None
        view_21: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [256, 128, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_10: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_21, convert_element_type_21);  view_21 = convert_element_type_21 = None
        convert_element_type_30: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.float32);  add_10 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_30, [2], correction = 0, keepdim = True)
        getitem_4: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_2: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_2 = None
        scalar_tensor_3: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_3 = None
        full_default_2: "bf16[256, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        sub_3: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_30, getitem_5);  convert_element_type_30 = getitem_5 = None
        add_11: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_9: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, arg20_1);  mul_9 = arg20_1 = None
        add_12: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        convert_element_type_31: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_22: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [32768, 768])
        permute_11: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg23_1, view_22, permute_11);  arg23_1 = view_22 = permute_11 = None
        view_23: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [256, 128, 768]);  addmm_6 = None
        view_24: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [256, 128, -1, 64]);  view_23 = None
        permute_12: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_11: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
        expand_5: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_11, [256, 12, 128, 64]);  mul_11 = None
        clone_6: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_31: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [3072, 128, 64]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_25: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [32768, 768])
        permute_13: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg25_1, view_25, permute_13);  arg25_1 = view_25 = permute_13 = None
        view_26: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [256, 128, 768]);  addmm_7 = None
        view_27: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [256, 128, -1, 64]);  view_26 = None
        permute_14: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_17: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        mul_12: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_17, 0.3535533905932738);  permute_17 = None
        expand_6: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_12, [256, 12, 64, 128]);  mul_12 = None
        clone_7: "bf16[256, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_32: "bf16[3072, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [3072, 64, 128]);  clone_7 = None
        bmm_2: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_31, view_32);  view_31 = view_32 = None
        view_33: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [256, 12, 128, 128]);  bmm_2 = None
        eq_1: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_33, -inf)
        logical_not_2: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_default_3: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 12, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_43: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        amax_1: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_43, [-1], True)
        sub_4: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_43, amax_1);  convert_element_type_43 = amax_1 = None
        exp_1: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_44: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        where_3: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_3, full_default_3, convert_element_type_44);  logical_not_3 = full_default_3 = convert_element_type_44 = None
        expand_7: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_3, [256, 12, 128, 128]);  where_3 = None
        view_34: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_7, [3072, 128, 128]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_28: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [32768, 768])
        permute_15: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg27_1, view_28, permute_15);  arg27_1 = view_28 = permute_15 = None
        view_29: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [256, 128, 768]);  addmm_8 = None
        view_30: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [256, 128, -1, 64]);  view_29 = None
        permute_16: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_16, [256, 12, 128, 64]);  permute_16 = None
        clone_8: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_35: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [3072, 128, 64]);  clone_8 = None
        bmm_3: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
        view_36: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [256, 12, 128, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[256, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_9: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_37: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [256, 128, -1]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_38: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [32768, 768]);  view_37 = None
        permute_19: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg29_1, view_38, permute_19);  arg29_1 = view_38 = permute_19 = None
        view_39: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [256, 128, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_14: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_39, convert_element_type_31);  view_39 = convert_element_type_31 = None
        convert_element_type_50: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_6: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_7);  convert_element_type_50 = getitem_7 = None
        add_15: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_13: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_14: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
        add_16: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_14, arg31_1);  mul_14 = arg31_1 = None
        convert_element_type_51: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_40: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_51, [32768, 768])
        permute_20: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg33_1, view_40, permute_20);  arg33_1 = view_40 = permute_20 = None
        view_41: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [256, 128, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_55: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_15: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.5)
        mul_16: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.7071067811865476);  convert_element_type_55 = None
        erf_1: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
        add_17: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_17: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, add_17);  mul_15 = add_17 = None
        convert_element_type_56: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_42: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_56, [32768, 3072]);  convert_element_type_56 = None
        permute_21: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg35_1, view_42, permute_21);  arg35_1 = view_42 = permute_21 = None
        view_43: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [256, 128, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_18: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_43, convert_element_type_51);  view_43 = convert_element_type_51 = None
        convert_element_type_60: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_18, torch.float32);  add_18 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_60, [2], correction = 0, keepdim = True)
        getitem_8: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_4: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_4 = None
        scalar_tensor_5: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_5 = None
        full_default_4: "bf16[256, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        sub_6: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_60, getitem_9);  convert_element_type_60 = getitem_9 = None
        add_19: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_18: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_19: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
        add_20: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, arg37_1);  mul_19 = arg37_1 = None
        convert_element_type_61: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_44: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [32768, 768])
        permute_22: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg39_1, view_44, permute_22);  arg39_1 = view_44 = permute_22 = None
        view_45: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [256, 128, 768]);  addmm_12 = None
        view_46: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [256, 128, -1, 64]);  view_45 = None
        permute_23: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_20: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_23, 0.3535533905932738);  permute_23 = None
        expand_9: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_20, [256, 12, 128, 64]);  mul_20 = None
        clone_11: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_53: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [3072, 128, 64]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_47: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [32768, 768])
        permute_24: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg41_1, view_47, permute_24);  arg41_1 = view_47 = permute_24 = None
        view_48: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [256, 128, 768]);  addmm_13 = None
        view_49: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_48, [256, 128, -1, 64]);  view_48 = None
        permute_25: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_28: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        mul_21: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_28, 0.3535533905932738);  permute_28 = None
        expand_10: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_21, [256, 12, 64, 128]);  mul_21 = None
        clone_12: "bf16[256, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_54: "bf16[3072, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [3072, 64, 128]);  clone_12 = None
        bmm_4: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_53, view_54);  view_53 = view_54 = None
        view_55: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [256, 12, 128, 128]);  bmm_4 = None
        eq_2: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_55, -inf)
        logical_not_4: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_default_5: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 12, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_73: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None
        amax_2: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_73, [-1], True)
        sub_7: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_73, amax_2);  convert_element_type_73 = amax_2 = None
        exp_2: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_74: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        where_5: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_5, full_default_5, convert_element_type_74);  logical_not_5 = full_default_5 = convert_element_type_74 = None
        expand_11: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_5, [256, 12, 128, 128]);  where_5 = None
        view_56: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_11, [3072, 128, 128]);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_50: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [32768, 768])
        permute_26: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg43_1, view_50, permute_26);  arg43_1 = view_50 = permute_26 = None
        view_51: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [256, 128, 768]);  addmm_14 = None
        view_52: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_51, [256, 128, -1, 64]);  view_51 = None
        permute_27: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_12: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_27, [256, 12, 128, 64]);  permute_27 = None
        clone_13: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_57: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [3072, 128, 64]);  clone_13 = None
        bmm_5: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_56, view_57);  view_56 = view_57 = None
        view_58: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [256, 12, 128, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[256, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_14: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_59: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [256, 128, -1]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_60: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [32768, 768]);  view_59 = None
        permute_30: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg45_1, view_60, permute_30);  arg45_1 = view_60 = permute_30 = None
        view_61: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [256, 128, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_22: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_61, convert_element_type_61);  view_61 = convert_element_type_61 = None
        convert_element_type_80: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.float32);  add_22 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True)
        getitem_10: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_11);  convert_element_type_80 = getitem_11 = None
        add_23: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_22: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_23: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, arg46_1);  mul_22 = arg46_1 = None
        add_24: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, arg47_1);  mul_23 = arg47_1 = None
        convert_element_type_81: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_62: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_81, [32768, 768])
        permute_31: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg49_1, view_62, permute_31);  arg49_1 = view_62 = permute_31 = None
        view_63: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [256, 128, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_85: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_24: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.5)
        mul_25: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.7071067811865476);  convert_element_type_85 = None
        erf_2: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_25: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_26: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, add_25);  mul_24 = add_25 = None
        convert_element_type_86: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_64: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_86, [32768, 3072]);  convert_element_type_86 = None
        permute_32: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg51_1, view_64, permute_32);  arg51_1 = view_64 = permute_32 = None
        view_65: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [256, 128, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_26: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_65, convert_element_type_81);  view_65 = convert_element_type_81 = None
        convert_element_type_90: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.float32);  add_26 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True)
        getitem_12: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_6: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_6 = None
        scalar_tensor_7: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_7 = None
        full_default_6: "bf16[256, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        sub_9: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_90, getitem_13);  convert_element_type_90 = getitem_13 = None
        add_27: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_27: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_28: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, arg52_1);  mul_27 = arg52_1 = None
        add_28: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_28, arg53_1);  mul_28 = arg53_1 = None
        convert_element_type_91: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_66: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [32768, 768])
        permute_33: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg55_1, view_66, permute_33);  arg55_1 = view_66 = permute_33 = None
        view_67: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [256, 128, 768]);  addmm_18 = None
        view_68: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [256, 128, -1, 64]);  view_67 = None
        permute_34: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_29: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_34, 0.3535533905932738);  permute_34 = None
        expand_13: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_29, [256, 12, 128, 64]);  mul_29 = None
        clone_16: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_75: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [3072, 128, 64]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_69: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [32768, 768])
        permute_35: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg57_1, view_69, permute_35);  arg57_1 = view_69 = permute_35 = None
        view_70: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [256, 128, 768]);  addmm_19 = None
        view_71: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_70, [256, 128, -1, 64]);  view_70 = None
        permute_36: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_39: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        mul_30: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_39, 0.3535533905932738);  permute_39 = None
        expand_14: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_30, [256, 12, 64, 128]);  mul_30 = None
        clone_17: "bf16[256, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_76: "bf16[3072, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [3072, 64, 128]);  clone_17 = None
        bmm_6: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_75, view_76);  view_75 = view_76 = None
        view_77: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [256, 12, 128, 128]);  bmm_6 = None
        eq_3: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_77, -inf)
        logical_not_6: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_default_7: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 12, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_103: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        amax_3: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_103, [-1], True)
        sub_10: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_103, amax_3);  convert_element_type_103 = amax_3 = None
        exp_3: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_104: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        where_7: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_7, full_default_7, convert_element_type_104);  logical_not_7 = full_default_7 = convert_element_type_104 = None
        expand_15: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_7, [256, 12, 128, 128]);  where_7 = None
        view_78: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_15, [3072, 128, 128]);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_72: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_91, [32768, 768])
        permute_37: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg59_1, view_72, permute_37);  arg59_1 = view_72 = permute_37 = None
        view_73: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [256, 128, 768]);  addmm_20 = None
        view_74: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [256, 128, -1, 64]);  view_73 = None
        permute_38: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_16: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_38, [256, 12, 128, 64]);  permute_38 = None
        clone_18: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_79: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [3072, 128, 64]);  clone_18 = None
        bmm_7: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
        view_80: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [256, 12, 128, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[256, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_19: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_81: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [256, 128, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_82: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [32768, 768]);  view_81 = None
        permute_41: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg61_1, view_82, permute_41);  arg61_1 = view_82 = permute_41 = None
        view_83: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [256, 128, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_30: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_83, convert_element_type_91);  view_83 = convert_element_type_91 = None
        convert_element_type_110: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True)
        getitem_14: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_15);  convert_element_type_110 = getitem_15 = None
        add_31: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_31: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_32: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, arg62_1);  mul_31 = arg62_1 = None
        add_32: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, arg63_1);  mul_32 = arg63_1 = None
        convert_element_type_111: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_84: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_111, [32768, 768])
        permute_42: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg65_1, view_84, permute_42);  arg65_1 = view_84 = permute_42 = None
        view_85: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [256, 128, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_115: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_33: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.5)
        mul_34: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.7071067811865476);  convert_element_type_115 = None
        erf_3: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_33: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_35: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, add_33);  mul_33 = add_33 = None
        convert_element_type_116: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_86: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [32768, 3072]);  convert_element_type_116 = None
        permute_43: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg67_1, view_86, permute_43);  arg67_1 = view_86 = permute_43 = None
        view_87: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [256, 128, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_34: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_87, convert_element_type_111);  view_87 = convert_element_type_111 = None
        convert_element_type_120: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.float32);  add_34 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_16: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_8: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_8 = None
        scalar_tensor_9: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_9 = None
        full_default_8: "bf16[256, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        sub_12: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_17);  convert_element_type_120 = getitem_17 = None
        add_35: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_36: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_37: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg68_1);  mul_36 = arg68_1 = None
        add_36: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg69_1);  mul_37 = arg69_1 = None
        convert_element_type_121: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_88: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [32768, 768])
        permute_44: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg71_1, view_88, permute_44);  arg71_1 = view_88 = permute_44 = None
        view_89: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [256, 128, 768]);  addmm_24 = None
        view_90: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [256, 128, -1, 64]);  view_89 = None
        permute_45: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_38: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_45, 0.3535533905932738);  permute_45 = None
        expand_17: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_38, [256, 12, 128, 64]);  mul_38 = None
        clone_21: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_97: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [3072, 128, 64]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_91: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [32768, 768])
        permute_46: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg73_1, view_91, permute_46);  arg73_1 = view_91 = permute_46 = None
        view_92: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [256, 128, 768]);  addmm_25 = None
        view_93: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_92, [256, 128, -1, 64]);  view_92 = None
        permute_47: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_50: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        mul_39: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_50, 0.3535533905932738);  permute_50 = None
        expand_18: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_39, [256, 12, 64, 128]);  mul_39 = None
        clone_22: "bf16[256, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_98: "bf16[3072, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [3072, 64, 128]);  clone_22 = None
        bmm_8: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
        view_99: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [256, 12, 128, 128]);  bmm_8 = None
        eq_4: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_99, -inf)
        logical_not_8: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_default_9: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 12, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_133: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        amax_4: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_133, [-1], True)
        sub_13: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_133, amax_4);  convert_element_type_133 = amax_4 = None
        exp_4: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_134: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        where_9: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_9, full_default_9, convert_element_type_134);  logical_not_9 = full_default_9 = convert_element_type_134 = None
        expand_19: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_9, [256, 12, 128, 128]);  where_9 = None
        view_100: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_19, [3072, 128, 128]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_94: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_121, [32768, 768])
        permute_48: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg75_1, view_94, permute_48);  arg75_1 = view_94 = permute_48 = None
        view_95: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [256, 128, 768]);  addmm_26 = None
        view_96: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_95, [256, 128, -1, 64]);  view_95 = None
        permute_49: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_20: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_49, [256, 12, 128, 64]);  permute_49 = None
        clone_23: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_101: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [3072, 128, 64]);  clone_23 = None
        bmm_9: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
        view_102: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [256, 12, 128, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "bf16[256, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_24: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_103: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [256, 128, -1]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_104: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [32768, 768]);  view_103 = None
        permute_52: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg77_1, view_104, permute_52);  arg77_1 = view_104 = permute_52 = None
        view_105: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [256, 128, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_38: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_105, convert_element_type_121);  view_105 = convert_element_type_121 = None
        convert_element_type_140: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.float32);  add_38 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_140, [2], correction = 0, keepdim = True)
        getitem_18: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_140, getitem_19);  convert_element_type_140 = getitem_19 = None
        add_39: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_40: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_41: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, arg78_1);  mul_40 = arg78_1 = None
        add_40: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, arg79_1);  mul_41 = arg79_1 = None
        convert_element_type_141: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_106: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_141, [32768, 768])
        permute_53: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg81_1, view_106, permute_53);  arg81_1 = view_106 = permute_53 = None
        view_107: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [256, 128, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_145: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_42: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_145, 0.5)
        mul_43: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_145, 0.7071067811865476);  convert_element_type_145 = None
        erf_4: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_41: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_44: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, add_41);  mul_42 = add_41 = None
        convert_element_type_146: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_108: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_146, [32768, 3072]);  convert_element_type_146 = None
        permute_54: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg83_1, view_108, permute_54);  arg83_1 = view_108 = permute_54 = None
        view_109: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [256, 128, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_42: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_109, convert_element_type_141);  view_109 = convert_element_type_141 = None
        convert_element_type_150: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_42, torch.float32);  add_42 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_150, [2], correction = 0, keepdim = True)
        getitem_20: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_10: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_10 = None
        scalar_tensor_11: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_11 = None
        full_default_10: "bf16[256, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        sub_15: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_150, getitem_21);  convert_element_type_150 = getitem_21 = None
        add_43: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_45: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_46: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, arg84_1);  mul_45 = arg84_1 = None
        add_44: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, arg85_1);  mul_46 = arg85_1 = None
        convert_element_type_151: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_110: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [32768, 768])
        permute_55: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg87_1, view_110, permute_55);  arg87_1 = view_110 = permute_55 = None
        view_111: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [256, 128, 768]);  addmm_30 = None
        view_112: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [256, 128, -1, 64]);  view_111 = None
        permute_56: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_47: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_56, 0.3535533905932738);  permute_56 = None
        expand_21: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_47, [256, 12, 128, 64]);  mul_47 = None
        clone_26: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_119: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [3072, 128, 64]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_113: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [32768, 768])
        permute_57: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg89_1, view_113, permute_57);  arg89_1 = view_113 = permute_57 = None
        view_114: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [256, 128, 768]);  addmm_31 = None
        view_115: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [256, 128, -1, 64]);  view_114 = None
        permute_58: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_61: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        mul_48: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_61, 0.3535533905932738);  permute_61 = None
        expand_22: "bf16[256, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_48, [256, 12, 64, 128]);  mul_48 = None
        clone_27: "bf16[256, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_120: "bf16[3072, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [3072, 64, 128]);  clone_27 = None
        bmm_10: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
        view_121: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [256, 12, 128, 128]);  bmm_10 = None
        eq_5: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_121, -inf)
        logical_not_10: "b8[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_default_11: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([256, 12, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_163: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_121, torch.float32);  view_121 = None
        amax_5: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_163, [-1], True)
        sub_16: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_163, amax_5);  convert_element_type_163 = amax_5 = None
        exp_5: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[256, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_164: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        where_11: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(logical_not_11, full_default_11, convert_element_type_164);  logical_not_11 = full_default_11 = convert_element_type_164 = None
        expand_23: "bf16[256, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_11, [256, 12, 128, 128]);  where_11 = None
        view_122: "bf16[3072, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_23, [3072, 128, 128]);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_116: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_151, [32768, 768])
        permute_59: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg91_1, view_116, permute_59);  arg91_1 = view_116 = permute_59 = None
        view_117: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [256, 128, 768]);  addmm_32 = None
        view_118: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_117, [256, 128, -1, 64]);  view_117 = None
        permute_60: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_24: "bf16[256, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_60, [256, 12, 128, 64]);  permute_60 = None
        clone_28: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_123: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [3072, 128, 64]);  clone_28 = None
        bmm_11: "bf16[3072, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
        view_124: "bf16[256, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [256, 12, 128, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "bf16[256, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_29: "bf16[256, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:205 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_125: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [256, 128, -1]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        view_126: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [32768, 768]);  view_125 = None
        permute_63: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg93_1, view_126, permute_63);  arg93_1 = view_126 = permute_63 = None
        view_127: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [256, 128, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        add_46: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_127, convert_element_type_151);  view_127 = convert_element_type_151 = None
        convert_element_type_170: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.float32);  add_46 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_170, [2], correction = 0, keepdim = True)
        getitem_22: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_170, getitem_23);  convert_element_type_170 = getitem_23 = None
        add_47: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_49: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_50: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, arg94_1);  mul_49 = arg94_1 = None
        add_48: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, arg95_1);  mul_50 = arg95_1 = None
        convert_element_type_171: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        view_128: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_171, [32768, 768])
        permute_64: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg97_1, view_128, permute_64);  arg97_1 = view_128 = permute_64 = None
        view_129: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [256, 128, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_175: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_51: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.5)
        mul_52: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.7071067811865476);  convert_element_type_175 = None
        erf_5: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_49: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_53: "f32[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, add_49);  mul_51 = add_49 = None
        convert_element_type_176: "bf16[256, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        view_130: "bf16[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_176, [32768, 3072]);  convert_element_type_176 = None
        permute_65: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg99_1, view_130, permute_65);  arg99_1 = view_130 = permute_65 = None
        view_131: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [256, 128, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_50: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_131, convert_element_type_171);  view_131 = convert_element_type_171 = None
        convert_element_type_180: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.float32);  add_50 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_180, [2], correction = 0, keepdim = True)
        getitem_24: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_18: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_180, getitem_25);  convert_element_type_180 = getitem_25 = None
        add_51: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_54: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_55: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, arg100_1);  mul_54 = arg100_1 = None
        add_52: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, arg101_1);  mul_55 = arg101_1 = None
        convert_element_type_181: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:514 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        view_132: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_181, [32768, 768]);  convert_element_type_181 = None
        permute_66: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg103_1, view_132, permute_66);  arg103_1 = view_132 = permute_66 = None
        view_133: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [256, 128, 768]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_185: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        mul_56: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_185, 0.5)
        mul_57: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_185, 0.7071067811865476);  convert_element_type_185 = None
        erf_6: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
        add_53: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_58: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, add_53);  mul_56 = add_53 = None
        convert_element_type_186: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:516 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        convert_element_type_187: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_186, torch.float32);  convert_element_type_186 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_187, [2], correction = 0, keepdim = True)
        getitem_26: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[256, 128, 1][128, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:521 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        view_137: "i64[32768][1]cuda:0" = torch.ops.aten.reshape.default(arg107_1, [-1]);  arg107_1 = None
        ne_1: "b8[32768][1]cuda:0" = torch.ops.aten.ne.Scalar(view_137, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        full_default_14: "bf16[6][1]cuda:0" = torch.ops.aten.full.default([6], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "bf16[30528][1]cuda:0" = torch.ops.aten.cat.default([arg106_1, full_default_14]);  arg106_1 = full_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:516 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        sub_19: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_187, getitem_27);  convert_element_type_187 = getitem_27 = None
        add_54: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_59: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = rsqrt_13 = None
        mul_60: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, arg104_1);  mul_59 = arg104_1 = None
        add_55: "f32[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, arg105_1);  mul_60 = arg105_1 = None
        convert_element_type_188: "bf16[256, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        view_134: "bf16[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_188, [32768, 768]);  convert_element_type_188 = None
        permute_67: "bf16[768, 30522][1, 768]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "bf16[768, 30528][30528, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_67, [0, 6, 0, 0]);  permute_67 = None
        addmm_default: "bf16[32768, 30528][30528, 1]cuda:0" = torch.ops.aten.addmm.default(cat_default, view_134, constant_pad_nd_default);  cat_default = view_134 = constant_pad_nd_default = None
        slice_tensor: "bf16[32768, 30522][30528, 1]cuda:0" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -6);  addmm_default = None
        view_135: "bf16[256, 128, 30522][3907584, 30528, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [256, 128, 30522]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:521 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        view_136: "bf16[32768, 30522][30528, 1]cuda:0" = torch.ops.aten.reshape.default(view_135, [-1, 30522])
        convert_element_type_192: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_136, torch.float32);  view_136 = None
        amax_6: "f32[32768, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_192, [1], True)
        sub_20: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_192, amax_6);  convert_element_type_192 = amax_6 = None
        exp_6: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.exp.default(sub_20)
        sum_7: "f32[32768, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
        log: "f32[32768, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_21: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_20, log);  sub_20 = log = None
        convert_element_type_193: "bf16[32768, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_21, torch.bfloat16);  sub_21 = None
        ne: "b8[32768][1]cuda:0" = torch.ops.aten.ne.Scalar(view_137, -100)
        full_default_12: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[32768][1]cuda:0" = torch.ops.aten.where.self(ne, view_137, full_default_12);  ne = full_default_12 = None
        unsqueeze_3: "i64[32768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather: "bf16[32768, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_193, 1, unsqueeze_3);  convert_element_type_193 = unsqueeze_3 = None
        squeeze: "bf16[32768][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[32768][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_13: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "bf16[32768][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_13);  ne_1 = neg = full_default_13 = None
        sum_9: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_13);  where_13 = None
        ne_2: "b8[32768][1]cuda:0" = torch.ops.aten.ne.Scalar(view_137, -100);  view_137 = None
        sum_8: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_194: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_8, torch.bfloat16);  sum_8 = None
        div_6: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_9, convert_element_type_194);  sum_9 = convert_element_type_194 = None
        return (div_6, view_135)
