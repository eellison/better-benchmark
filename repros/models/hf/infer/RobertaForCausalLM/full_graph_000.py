class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[32, 512][512, 1]cuda:0", arg1_1: "i64[32, 512][512, 1]cuda:0", arg2_1: "i64[1, 512][512, 1]cuda:0", arg3_1: "bf16[50265, 768][768, 1]cuda:0", arg4_1: "bf16[2, 768][768, 1]cuda:0", arg5_1: "bf16[512, 768][768, 1]cuda:0", arg6_1: "bf16[768][1]cuda:0", arg7_1: "bf16[768][1]cuda:0", arg8_1: "bf16[768, 768][768, 1]cuda:0", arg9_1: "bf16[768][1]cuda:0", arg10_1: "bf16[768, 768][768, 1]cuda:0", arg11_1: "bf16[768][1]cuda:0", arg12_1: "bf16[768, 768][768, 1]cuda:0", arg13_1: "bf16[768][1]cuda:0", arg14_1: "bf16[768, 768][768, 1]cuda:0", arg15_1: "bf16[768][1]cuda:0", arg16_1: "bf16[768][1]cuda:0", arg17_1: "bf16[768][1]cuda:0", arg18_1: "bf16[3072, 768][768, 1]cuda:0", arg19_1: "bf16[3072][1]cuda:0", arg20_1: "bf16[768, 3072][3072, 1]cuda:0", arg21_1: "bf16[768][1]cuda:0", arg22_1: "bf16[768][1]cuda:0", arg23_1: "bf16[768][1]cuda:0", arg24_1: "bf16[768, 768][768, 1]cuda:0", arg25_1: "bf16[768][1]cuda:0", arg26_1: "bf16[768, 768][768, 1]cuda:0", arg27_1: "bf16[768][1]cuda:0", arg28_1: "bf16[768, 768][768, 1]cuda:0", arg29_1: "bf16[768][1]cuda:0", arg30_1: "bf16[768, 768][768, 1]cuda:0", arg31_1: "bf16[768][1]cuda:0", arg32_1: "bf16[768][1]cuda:0", arg33_1: "bf16[768][1]cuda:0", arg34_1: "bf16[3072, 768][768, 1]cuda:0", arg35_1: "bf16[3072][1]cuda:0", arg36_1: "bf16[768, 3072][3072, 1]cuda:0", arg37_1: "bf16[768][1]cuda:0", arg38_1: "bf16[768][1]cuda:0", arg39_1: "bf16[768][1]cuda:0", arg40_1: "bf16[768, 768][768, 1]cuda:0", arg41_1: "bf16[768][1]cuda:0", arg42_1: "bf16[768, 768][768, 1]cuda:0", arg43_1: "bf16[768][1]cuda:0", arg44_1: "bf16[768, 768][768, 1]cuda:0", arg45_1: "bf16[768][1]cuda:0", arg46_1: "bf16[768, 768][768, 1]cuda:0", arg47_1: "bf16[768][1]cuda:0", arg48_1: "bf16[768][1]cuda:0", arg49_1: "bf16[768][1]cuda:0", arg50_1: "bf16[3072, 768][768, 1]cuda:0", arg51_1: "bf16[3072][1]cuda:0", arg52_1: "bf16[768, 3072][3072, 1]cuda:0", arg53_1: "bf16[768][1]cuda:0", arg54_1: "bf16[768][1]cuda:0", arg55_1: "bf16[768][1]cuda:0", arg56_1: "bf16[768, 768][768, 1]cuda:0", arg57_1: "bf16[768][1]cuda:0", arg58_1: "bf16[768, 768][768, 1]cuda:0", arg59_1: "bf16[768][1]cuda:0", arg60_1: "bf16[768, 768][768, 1]cuda:0", arg61_1: "bf16[768][1]cuda:0", arg62_1: "bf16[768, 768][768, 1]cuda:0", arg63_1: "bf16[768][1]cuda:0", arg64_1: "bf16[768][1]cuda:0", arg65_1: "bf16[768][1]cuda:0", arg66_1: "bf16[3072, 768][768, 1]cuda:0", arg67_1: "bf16[3072][1]cuda:0", arg68_1: "bf16[768, 3072][3072, 1]cuda:0", arg69_1: "bf16[768][1]cuda:0", arg70_1: "bf16[768][1]cuda:0", arg71_1: "bf16[768][1]cuda:0", arg72_1: "bf16[768, 768][768, 1]cuda:0", arg73_1: "bf16[768][1]cuda:0", arg74_1: "bf16[768, 768][768, 1]cuda:0", arg75_1: "bf16[768][1]cuda:0", arg76_1: "bf16[768, 768][768, 1]cuda:0", arg77_1: "bf16[768][1]cuda:0", arg78_1: "bf16[768, 768][768, 1]cuda:0", arg79_1: "bf16[768][1]cuda:0", arg80_1: "bf16[768][1]cuda:0", arg81_1: "bf16[768][1]cuda:0", arg82_1: "bf16[3072, 768][768, 1]cuda:0", arg83_1: "bf16[3072][1]cuda:0", arg84_1: "bf16[768, 3072][3072, 1]cuda:0", arg85_1: "bf16[768][1]cuda:0", arg86_1: "bf16[768][1]cuda:0", arg87_1: "bf16[768][1]cuda:0", arg88_1: "bf16[768, 768][768, 1]cuda:0", arg89_1: "bf16[768][1]cuda:0", arg90_1: "bf16[768, 768][768, 1]cuda:0", arg91_1: "bf16[768][1]cuda:0", arg92_1: "bf16[768, 768][768, 1]cuda:0", arg93_1: "bf16[768][1]cuda:0", arg94_1: "bf16[768, 768][768, 1]cuda:0", arg95_1: "bf16[768][1]cuda:0", arg96_1: "bf16[768][1]cuda:0", arg97_1: "bf16[768][1]cuda:0", arg98_1: "bf16[3072, 768][768, 1]cuda:0", arg99_1: "bf16[3072][1]cuda:0", arg100_1: "bf16[768, 3072][3072, 1]cuda:0", arg101_1: "bf16[768][1]cuda:0", arg102_1: "bf16[768][1]cuda:0", arg103_1: "bf16[768][1]cuda:0", arg104_1: "bf16[768, 768][768, 1]cuda:0", arg105_1: "bf16[768][1]cuda:0", arg106_1: "bf16[768, 768][768, 1]cuda:0", arg107_1: "bf16[768][1]cuda:0", arg108_1: "bf16[768, 768][768, 1]cuda:0", arg109_1: "bf16[768][1]cuda:0", arg110_1: "bf16[768, 768][768, 1]cuda:0", arg111_1: "bf16[768][1]cuda:0", arg112_1: "bf16[768][1]cuda:0", arg113_1: "bf16[768][1]cuda:0", arg114_1: "bf16[3072, 768][768, 1]cuda:0", arg115_1: "bf16[3072][1]cuda:0", arg116_1: "bf16[768, 3072][3072, 1]cuda:0", arg117_1: "bf16[768][1]cuda:0", arg118_1: "bf16[768][1]cuda:0", arg119_1: "bf16[768][1]cuda:0", arg120_1: "bf16[768, 768][768, 1]cuda:0", arg121_1: "bf16[768][1]cuda:0", arg122_1: "bf16[768, 768][768, 1]cuda:0", arg123_1: "bf16[768][1]cuda:0", arg124_1: "bf16[768, 768][768, 1]cuda:0", arg125_1: "bf16[768][1]cuda:0", arg126_1: "bf16[768, 768][768, 1]cuda:0", arg127_1: "bf16[768][1]cuda:0", arg128_1: "bf16[768][1]cuda:0", arg129_1: "bf16[768][1]cuda:0", arg130_1: "bf16[3072, 768][768, 1]cuda:0", arg131_1: "bf16[3072][1]cuda:0", arg132_1: "bf16[768, 3072][3072, 1]cuda:0", arg133_1: "bf16[768][1]cuda:0", arg134_1: "bf16[768][1]cuda:0", arg135_1: "bf16[768][1]cuda:0", arg136_1: "bf16[768, 768][768, 1]cuda:0", arg137_1: "bf16[768][1]cuda:0", arg138_1: "bf16[768, 768][768, 1]cuda:0", arg139_1: "bf16[768][1]cuda:0", arg140_1: "bf16[768, 768][768, 1]cuda:0", arg141_1: "bf16[768][1]cuda:0", arg142_1: "bf16[768, 768][768, 1]cuda:0", arg143_1: "bf16[768][1]cuda:0", arg144_1: "bf16[768][1]cuda:0", arg145_1: "bf16[768][1]cuda:0", arg146_1: "bf16[3072, 768][768, 1]cuda:0", arg147_1: "bf16[3072][1]cuda:0", arg148_1: "bf16[768, 3072][3072, 1]cuda:0", arg149_1: "bf16[768][1]cuda:0", arg150_1: "bf16[768][1]cuda:0", arg151_1: "bf16[768][1]cuda:0", arg152_1: "bf16[768, 768][768, 1]cuda:0", arg153_1: "bf16[768][1]cuda:0", arg154_1: "bf16[768, 768][768, 1]cuda:0", arg155_1: "bf16[768][1]cuda:0", arg156_1: "bf16[768, 768][768, 1]cuda:0", arg157_1: "bf16[768][1]cuda:0", arg158_1: "bf16[768, 768][768, 1]cuda:0", arg159_1: "bf16[768][1]cuda:0", arg160_1: "bf16[768][1]cuda:0", arg161_1: "bf16[768][1]cuda:0", arg162_1: "bf16[3072, 768][768, 1]cuda:0", arg163_1: "bf16[3072][1]cuda:0", arg164_1: "bf16[768, 3072][3072, 1]cuda:0", arg165_1: "bf16[768][1]cuda:0", arg166_1: "bf16[768][1]cuda:0", arg167_1: "bf16[768][1]cuda:0", arg168_1: "bf16[768, 768][768, 1]cuda:0", arg169_1: "bf16[768][1]cuda:0", arg170_1: "bf16[768, 768][768, 1]cuda:0", arg171_1: "bf16[768][1]cuda:0", arg172_1: "bf16[768, 768][768, 1]cuda:0", arg173_1: "bf16[768][1]cuda:0", arg174_1: "bf16[768, 768][768, 1]cuda:0", arg175_1: "bf16[768][1]cuda:0", arg176_1: "bf16[768][1]cuda:0", arg177_1: "bf16[768][1]cuda:0", arg178_1: "bf16[3072, 768][768, 1]cuda:0", arg179_1: "bf16[3072][1]cuda:0", arg180_1: "bf16[768, 3072][3072, 1]cuda:0", arg181_1: "bf16[768][1]cuda:0", arg182_1: "bf16[768][1]cuda:0", arg183_1: "bf16[768][1]cuda:0", arg184_1: "bf16[768, 768][768, 1]cuda:0", arg185_1: "bf16[768][1]cuda:0", arg186_1: "bf16[768, 768][768, 1]cuda:0", arg187_1: "bf16[768][1]cuda:0", arg188_1: "bf16[768, 768][768, 1]cuda:0", arg189_1: "bf16[768][1]cuda:0", arg190_1: "bf16[768, 768][768, 1]cuda:0", arg191_1: "bf16[768][1]cuda:0", arg192_1: "bf16[768][1]cuda:0", arg193_1: "bf16[768][1]cuda:0", arg194_1: "bf16[3072, 768][768, 1]cuda:0", arg195_1: "bf16[3072][1]cuda:0", arg196_1: "bf16[768, 3072][3072, 1]cuda:0", arg197_1: "bf16[768][1]cuda:0", arg198_1: "bf16[768][1]cuda:0", arg199_1: "bf16[768][1]cuda:0", arg200_1: "bf16[768, 768][768, 1]cuda:0", arg201_1: "bf16[768][1]cuda:0", arg202_1: "bf16[768][1]cuda:0", arg203_1: "bf16[768][1]cuda:0", arg204_1: "bf16[50265][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:116 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg3_1, arg1_1, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:109 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand: "i64[32, 512][0, 1]cuda:0" = torch.ops.aten.expand.default(arg2_1, [32, -1]);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:157 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg1_1, 0);  arg1_1 = None
        convert_element_type: "i32[32, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:158 in create_position_ids_from_input_ids, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        cumsum: "i64[32, 512][512, 1]cuda:0" = torch.ops.aten.cumsum.default(convert_element_type, 1)
        convert_element_type_1: "i32[32, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        add: "i32[32, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
        mul: "i32[32, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add, convert_element_type);  add = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:159 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_2: "i64[32, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul, torch.int64);  mul = None
        add_1: "i64[32, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2, 0);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:110 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather: "i64[32, 512][512, 1]cuda:0" = torch.ops.aten.gather.default(expand, 1, add_1);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:111 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[32, 512][512, 1]cuda:0" = torch.ops.aten.expand.default(gather, [32, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:117 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg4_1, expand_1);  arg4_1 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:118 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add_2: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:120 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg5_1, add_1, 0);  arg5_1 = add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:121 in forward, code: embeddings = embeddings + position_embeddings
        add_3: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_2, embedding_2);  add_2 = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:123 in forward, code: embeddings = self.LayerNorm(embeddings)
        convert_element_type_3: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_3, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_6: "i64[512][1]cuda:0" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_6, 0);  add_6 = None
        unsqueeze_1: "i64[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 512, 1][512, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 512, 1][512, 512, 1, 1]cuda:0" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_2: "b8[32, 1, 512, 512][0, 512, 1, 0]cuda:0" = torch.ops.aten.expand.default(ge, [32, -1, 512, 512]);  ge = expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor = None
        scalar_tensor_1: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_1 = None
        full_default: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:123 in forward, code: embeddings = self.LayerNorm(embeddings)
        sub: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_3, getitem_1);  convert_element_type_3 = getitem_1 = None
        add_4: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        add_5: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, arg7_1);  mul_2 = arg7_1 = None
        convert_element_type_4: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_4, [16384, 768])
        permute: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [32, 512, 768]);  addmm = None
        view_2: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [32, 512, -1, 64]);  view_1 = None
        permute_1: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_3: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        expand_3: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_3, [32, 12, 512, 64]);  mul_3 = None
        clone_1: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_9: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [384, 512, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_3: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_4, [16384, 768])
        permute_2: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_1: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg11_1, view_3, permute_2);  arg11_1 = view_3 = permute_2 = None
        view_4: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 512, 768]);  addmm_1 = None
        view_5: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [32, 512, -1, 64]);  view_4 = None
        permute_3: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_6: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_4: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        expand_4: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_4, [32, 12, 64, 512]);  mul_4 = None
        clone_2: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_10: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [384, 64, 512]);  clone_2 = None
        bmm: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [32, 12, 512, 512]);  bmm = None
        eq: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_11, -inf)
        logical_not: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_1: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_16: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        amax: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_16, [-1], True)
        sub_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_16, amax);  convert_element_type_16 = amax = None
        exp: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_17: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where_1: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_1, full_default_1, convert_element_type_17);  logical_not_1 = full_default_1 = convert_element_type_17 = None
        expand_5: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_1, [32, 12, 512, 512]);  where_1 = None
        view_12: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_5, [384, 512, 512]);  expand_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_6: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_4, [16384, 768])
        permute_4: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_2: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg13_1, view_6, permute_4);  arg13_1 = view_6 = permute_4 = None
        view_7: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 768]);  addmm_2 = None
        view_8: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_7, [32, 512, -1, 64]);  view_7 = None
        permute_5: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_6: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [32, 12, 512, 64]);  permute_5 = None
        clone_3: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_13: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [384, 512, 64]);  clone_3 = None
        bmm_1: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
        view_14: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [32, 12, 512, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_4: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [32, 512, -1]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [16384, 768]);  view_15 = None
        permute_8: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_3: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg15_1, view_16, permute_8);  arg15_1 = view_16 = permute_8 = None
        view_17: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [32, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_9: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_17, convert_element_type_4);  view_17 = convert_element_type_4 = None
        convert_element_type_23: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_23, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_23, getitem_3);  convert_element_type_23 = getitem_3 = None
        add_10: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_5: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_6: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, arg16_1);  mul_5 = arg16_1 = None
        add_11: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, arg17_1);  mul_6 = arg17_1 = None
        convert_element_type_24: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_24, [16384, 768])
        permute_9: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_4: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg19_1, view_18, permute_9);  arg19_1 = view_18 = permute_9 = None
        view_19: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_28: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_7: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_28, 0.5)
        mul_8: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_28, 0.7071067811865476);  convert_element_type_28 = None
        erf: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_12: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, add_12);  mul_7 = add_12 = None
        convert_element_type_29: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_29, [16384, 3072]);  convert_element_type_29 = None
        permute_10: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_5: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg21_1, view_20, permute_10);  arg21_1 = view_20 = permute_10 = None
        view_21: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_13: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_21, convert_element_type_24);  view_21 = convert_element_type_24 = None
        convert_element_type_33: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_33, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_2: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_2 = None
        scalar_tensor_3: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_3 = None
        full_default_2: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_3: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_5);  convert_element_type_33 = getitem_5 = None
        add_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_10: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_11: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, arg22_1);  mul_10 = arg22_1 = None
        add_15: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, arg23_1);  mul_11 = arg23_1 = None
        convert_element_type_34: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_22: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_34, [16384, 768])
        permute_11: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_6: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg25_1, view_22, permute_11);  arg25_1 = view_22 = permute_11 = None
        view_23: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [32, 512, 768]);  addmm_6 = None
        view_24: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [32, 512, -1, 64]);  view_23 = None
        permute_12: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_12: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
        expand_7: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_12, [32, 12, 512, 64]);  mul_12 = None
        clone_7: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_31: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [384, 512, 64]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_25: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_34, [16384, 768])
        permute_13: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_7: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg27_1, view_25, permute_13);  arg27_1 = view_25 = permute_13 = None
        view_26: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [32, 512, 768]);  addmm_7 = None
        view_27: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [32, 512, -1, 64]);  view_26 = None
        permute_14: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_17: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        mul_13: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_17, 0.3535533905932738);  permute_17 = None
        expand_8: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_13, [32, 12, 64, 512]);  mul_13 = None
        clone_8: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_32: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [384, 64, 512]);  clone_8 = None
        bmm_2: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_31, view_32);  view_31 = view_32 = None
        view_33: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [32, 12, 512, 512]);  bmm_2 = None
        eq_1: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_33, -inf)
        logical_not_2: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_default_3: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_46: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        amax_1: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_46, [-1], True)
        sub_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_46, amax_1);  convert_element_type_46 = amax_1 = None
        exp_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_47: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        where_3: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_3, full_default_3, convert_element_type_47);  logical_not_3 = full_default_3 = convert_element_type_47 = None
        expand_9: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_3, [32, 12, 512, 512]);  where_3 = None
        view_34: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_9, [384, 512, 512]);  expand_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_28: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_34, [16384, 768])
        permute_15: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_8: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg29_1, view_28, permute_15);  arg29_1 = view_28 = permute_15 = None
        view_29: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [32, 512, 768]);  addmm_8 = None
        view_30: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [32, 512, -1, 64]);  view_29 = None
        permute_16: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_10: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_16, [32, 12, 512, 64]);  permute_16 = None
        clone_9: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_35: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [384, 512, 64]);  clone_9 = None
        bmm_3: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
        view_36: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [32, 12, 512, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_10: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_37: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [32, 512, -1]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [16384, 768]);  view_37 = None
        permute_19: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_9: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg31_1, view_38, permute_19);  arg31_1 = view_38 = permute_19 = None
        view_39: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_17: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_39, convert_element_type_34);  view_39 = convert_element_type_34 = None
        convert_element_type_53: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.float32);  add_17 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_53, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_53, getitem_7);  convert_element_type_53 = getitem_7 = None
        add_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_14: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_15: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, arg32_1);  mul_14 = arg32_1 = None
        add_19: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, arg33_1);  mul_15 = arg33_1 = None
        convert_element_type_54: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_54, [16384, 768])
        permute_20: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_10: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg35_1, view_40, permute_20);  arg35_1 = view_40 = permute_20 = None
        view_41: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_58: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_16: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.5)
        mul_17: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.7071067811865476);  convert_element_type_58 = None
        erf_1: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_20: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_18: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, add_20);  mul_16 = add_20 = None
        convert_element_type_59: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_59, [16384, 3072]);  convert_element_type_59 = None
        permute_21: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_11: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg37_1, view_42, permute_21);  arg37_1 = view_42 = permute_21 = None
        view_43: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [32, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_21: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_43, convert_element_type_54);  view_43 = convert_element_type_54 = None
        convert_element_type_63: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.float32);  add_21 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_63, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_4: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_4 = None
        scalar_tensor_5: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_5 = None
        full_default_4: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_6: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_63, getitem_9);  convert_element_type_63 = getitem_9 = None
        add_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_19: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_20: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, arg38_1);  mul_19 = arg38_1 = None
        add_23: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, arg39_1);  mul_20 = arg39_1 = None
        convert_element_type_64: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_44: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_64, [16384, 768])
        permute_22: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_12: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg41_1, view_44, permute_22);  arg41_1 = view_44 = permute_22 = None
        view_45: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [32, 512, 768]);  addmm_12 = None
        view_46: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [32, 512, -1, 64]);  view_45 = None
        permute_23: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_21: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_23, 0.3535533905932738);  permute_23 = None
        expand_11: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_21, [32, 12, 512, 64]);  mul_21 = None
        clone_13: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_53: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [384, 512, 64]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_47: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_64, [16384, 768])
        permute_24: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_13: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg43_1, view_47, permute_24);  arg43_1 = view_47 = permute_24 = None
        view_48: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [32, 512, 768]);  addmm_13 = None
        view_49: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_48, [32, 512, -1, 64]);  view_48 = None
        permute_25: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_28: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        mul_22: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_28, 0.3535533905932738);  permute_28 = None
        expand_12: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_22, [32, 12, 64, 512]);  mul_22 = None
        clone_14: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_54: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [384, 64, 512]);  clone_14 = None
        bmm_4: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_53, view_54);  view_53 = view_54 = None
        view_55: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [32, 12, 512, 512]);  bmm_4 = None
        eq_2: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_55, -inf)
        logical_not_4: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_default_5: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_76: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None
        amax_2: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_76, [-1], True)
        sub_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_76, amax_2);  convert_element_type_76 = amax_2 = None
        exp_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_77: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        where_5: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_5, full_default_5, convert_element_type_77);  logical_not_5 = full_default_5 = convert_element_type_77 = None
        expand_13: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_5, [32, 12, 512, 512]);  where_5 = None
        view_56: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_13, [384, 512, 512]);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_50: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_64, [16384, 768])
        permute_26: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_14: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg45_1, view_50, permute_26);  arg45_1 = view_50 = permute_26 = None
        view_51: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [32, 512, 768]);  addmm_14 = None
        view_52: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_51, [32, 512, -1, 64]);  view_51 = None
        permute_27: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_14: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_27, [32, 12, 512, 64]);  permute_27 = None
        clone_15: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_57: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [384, 512, 64]);  clone_15 = None
        bmm_5: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_56, view_57);  view_56 = view_57 = None
        view_58: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [32, 12, 512, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_16: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_59: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [32, 512, -1]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [16384, 768]);  view_59 = None
        permute_30: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_15: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg47_1, view_60, permute_30);  arg47_1 = view_60 = permute_30 = None
        view_61: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [32, 512, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_25: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_61, convert_element_type_64);  view_61 = convert_element_type_64 = None
        convert_element_type_83: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_83, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_83, getitem_11);  convert_element_type_83 = getitem_11 = None
        add_26: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_23: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_24: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, arg48_1);  mul_23 = arg48_1 = None
        add_27: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_24, arg49_1);  mul_24 = arg49_1 = None
        convert_element_type_84: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_84, [16384, 768])
        permute_31: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_16: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg51_1, view_62, permute_31);  arg51_1 = view_62 = permute_31 = None
        view_63: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_88: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_25: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.5)
        mul_26: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.7071067811865476);  convert_element_type_88 = None
        erf_2: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_28: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_27: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, add_28);  mul_25 = add_28 = None
        convert_element_type_89: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_89, [16384, 3072]);  convert_element_type_89 = None
        permute_32: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_17: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg53_1, view_64, permute_32);  arg53_1 = view_64 = permute_32 = None
        view_65: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [32, 512, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_29: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_65, convert_element_type_84);  view_65 = convert_element_type_84 = None
        convert_element_type_93: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.float32);  add_29 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_93, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_6: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_6 = None
        scalar_tensor_7: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_7 = None
        full_default_6: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_9: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_93, getitem_13);  convert_element_type_93 = getitem_13 = None
        add_30: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_28: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_29: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, arg54_1);  mul_28 = arg54_1 = None
        add_31: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, arg55_1);  mul_29 = arg55_1 = None
        convert_element_type_94: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_66: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_94, [16384, 768])
        permute_33: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_18: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg57_1, view_66, permute_33);  arg57_1 = view_66 = permute_33 = None
        view_67: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [32, 512, 768]);  addmm_18 = None
        view_68: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [32, 512, -1, 64]);  view_67 = None
        permute_34: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_30: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_34, 0.3535533905932738);  permute_34 = None
        expand_15: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_30, [32, 12, 512, 64]);  mul_30 = None
        clone_19: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_75: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [384, 512, 64]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_69: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_94, [16384, 768])
        permute_35: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_19: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg59_1, view_69, permute_35);  arg59_1 = view_69 = permute_35 = None
        view_70: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 768]);  addmm_19 = None
        view_71: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_70, [32, 512, -1, 64]);  view_70 = None
        permute_36: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_39: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        mul_31: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_39, 0.3535533905932738);  permute_39 = None
        expand_16: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_31, [32, 12, 64, 512]);  mul_31 = None
        clone_20: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_76: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [384, 64, 512]);  clone_20 = None
        bmm_6: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_75, view_76);  view_75 = view_76 = None
        view_77: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [32, 12, 512, 512]);  bmm_6 = None
        eq_3: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_77, -inf)
        logical_not_6: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_default_7: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_106: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        amax_3: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_106, [-1], True)
        sub_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_106, amax_3);  convert_element_type_106 = amax_3 = None
        exp_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_107: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        where_7: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_7, full_default_7, convert_element_type_107);  logical_not_7 = full_default_7 = convert_element_type_107 = None
        expand_17: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_7, [32, 12, 512, 512]);  where_7 = None
        view_78: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_17, [384, 512, 512]);  expand_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_72: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_94, [16384, 768])
        permute_37: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_20: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg61_1, view_72, permute_37);  arg61_1 = view_72 = permute_37 = None
        view_73: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [32, 512, 768]);  addmm_20 = None
        view_74: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [32, 512, -1, 64]);  view_73 = None
        permute_38: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_18: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_38, [32, 12, 512, 64]);  permute_38 = None
        clone_21: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_79: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [384, 512, 64]);  clone_21 = None
        bmm_7: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
        view_80: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [32, 12, 512, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_22: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_81: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [32, 512, -1]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_82: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [16384, 768]);  view_81 = None
        permute_41: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_21: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg63_1, view_82, permute_41);  arg63_1 = view_82 = permute_41 = None
        view_83: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [32, 512, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_33: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_83, convert_element_type_94);  view_83 = convert_element_type_94 = None
        convert_element_type_113: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_113, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_113, getitem_15);  convert_element_type_113 = getitem_15 = None
        add_34: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_32: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_33: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, arg64_1);  mul_32 = arg64_1 = None
        add_35: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, arg65_1);  mul_33 = arg65_1 = None
        convert_element_type_114: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_114, [16384, 768])
        permute_42: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_22: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg67_1, view_84, permute_42);  arg67_1 = view_84 = permute_42 = None
        view_85: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_118: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_34: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_118, 0.5)
        mul_35: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_118, 0.7071067811865476);  convert_element_type_118 = None
        erf_3: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
        add_36: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_36: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, add_36);  mul_34 = add_36 = None
        convert_element_type_119: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_119, [16384, 3072]);  convert_element_type_119 = None
        permute_43: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_23: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg69_1, view_86, permute_43);  arg69_1 = view_86 = permute_43 = None
        view_87: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_37: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_87, convert_element_type_114);  view_87 = convert_element_type_114 = None
        convert_element_type_123: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_123, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_8: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_8 = None
        scalar_tensor_9: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_9 = None
        full_default_8: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_123, getitem_17);  convert_element_type_123 = getitem_17 = None
        add_38: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_37: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_38: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, arg70_1);  mul_37 = arg70_1 = None
        add_39: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_38, arg71_1);  mul_38 = arg71_1 = None
        convert_element_type_124: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_88: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_124, [16384, 768])
        permute_44: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_24: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg73_1, view_88, permute_44);  arg73_1 = view_88 = permute_44 = None
        view_89: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [32, 512, 768]);  addmm_24 = None
        view_90: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [32, 512, -1, 64]);  view_89 = None
        permute_45: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_39: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_45, 0.3535533905932738);  permute_45 = None
        expand_19: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_39, [32, 12, 512, 64]);  mul_39 = None
        clone_25: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_97: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [384, 512, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_91: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_124, [16384, 768])
        permute_46: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_25: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg75_1, view_91, permute_46);  arg75_1 = view_91 = permute_46 = None
        view_92: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [32, 512, 768]);  addmm_25 = None
        view_93: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_92, [32, 512, -1, 64]);  view_92 = None
        permute_47: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_50: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        mul_40: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_50, 0.3535533905932738);  permute_50 = None
        expand_20: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_40, [32, 12, 64, 512]);  mul_40 = None
        clone_26: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_98: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [384, 64, 512]);  clone_26 = None
        bmm_8: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
        view_99: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [32, 12, 512, 512]);  bmm_8 = None
        eq_4: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_99, -inf)
        logical_not_8: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_default_9: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_136: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        amax_4: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_136, [-1], True)
        sub_13: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_136, amax_4);  convert_element_type_136 = amax_4 = None
        exp_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_137: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        where_9: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_9, full_default_9, convert_element_type_137);  logical_not_9 = full_default_9 = convert_element_type_137 = None
        expand_21: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_9, [32, 12, 512, 512]);  where_9 = None
        view_100: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_21, [384, 512, 512]);  expand_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_94: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_124, [16384, 768])
        permute_48: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_26: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg77_1, view_94, permute_48);  arg77_1 = view_94 = permute_48 = None
        view_95: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [32, 512, 768]);  addmm_26 = None
        view_96: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_95, [32, 512, -1, 64]);  view_95 = None
        permute_49: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_22: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_49, [32, 12, 512, 64]);  permute_49 = None
        clone_27: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_101: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [384, 512, 64]);  clone_27 = None
        bmm_9: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
        view_102: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [32, 12, 512, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_28: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_103: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [32, 512, -1]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [16384, 768]);  view_103 = None
        permute_52: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_27: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg79_1, view_104, permute_52);  arg79_1 = view_104 = permute_52 = None
        view_105: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [32, 512, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_41: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_105, convert_element_type_124);  view_105 = convert_element_type_124 = None
        convert_element_type_143: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_143, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_143, getitem_19);  convert_element_type_143 = getitem_19 = None
        add_42: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_41: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_42: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, arg80_1);  mul_41 = arg80_1 = None
        add_43: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_42, arg81_1);  mul_42 = arg81_1 = None
        convert_element_type_144: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_144, [16384, 768])
        permute_53: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_28: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg83_1, view_106, permute_53);  arg83_1 = view_106 = permute_53 = None
        view_107: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 512, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_148: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_43: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.5)
        mul_44: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.7071067811865476);  convert_element_type_148 = None
        erf_4: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
        add_44: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_45: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, add_44);  mul_43 = add_44 = None
        convert_element_type_149: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_149, [16384, 3072]);  convert_element_type_149 = None
        permute_54: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_29: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg85_1, view_108, permute_54);  arg85_1 = view_108 = permute_54 = None
        view_109: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [32, 512, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_45: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_109, convert_element_type_144);  view_109 = convert_element_type_144 = None
        convert_element_type_153: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.float32);  add_45 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_153, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_10: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_10 = None
        scalar_tensor_11: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_11 = None
        full_default_10: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_15: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_153, getitem_21);  convert_element_type_153 = getitem_21 = None
        add_46: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_46: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_47: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, arg86_1);  mul_46 = arg86_1 = None
        add_47: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_47, arg87_1);  mul_47 = arg87_1 = None
        convert_element_type_154: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_110: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_154, [16384, 768])
        permute_55: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_30: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg89_1, view_110, permute_55);  arg89_1 = view_110 = permute_55 = None
        view_111: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [32, 512, 768]);  addmm_30 = None
        view_112: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [32, 512, -1, 64]);  view_111 = None
        permute_56: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_48: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_56, 0.3535533905932738);  permute_56 = None
        expand_23: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_48, [32, 12, 512, 64]);  mul_48 = None
        clone_31: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_119: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [384, 512, 64]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_113: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_154, [16384, 768])
        permute_57: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_31: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg91_1, view_113, permute_57);  arg91_1 = view_113 = permute_57 = None
        view_114: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [32, 512, 768]);  addmm_31 = None
        view_115: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [32, 512, -1, 64]);  view_114 = None
        permute_58: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_61: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        mul_49: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_61, 0.3535533905932738);  permute_61 = None
        expand_24: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_49, [32, 12, 64, 512]);  mul_49 = None
        clone_32: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_120: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [384, 64, 512]);  clone_32 = None
        bmm_10: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
        view_121: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [32, 12, 512, 512]);  bmm_10 = None
        eq_5: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_121, -inf)
        logical_not_10: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_default_11: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_166: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_121, torch.float32);  view_121 = None
        amax_5: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_166, [-1], True)
        sub_16: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_166, amax_5);  convert_element_type_166 = amax_5 = None
        exp_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_167: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        where_11: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_11, full_default_11, convert_element_type_167);  logical_not_11 = full_default_11 = convert_element_type_167 = None
        expand_25: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_11, [32, 12, 512, 512]);  where_11 = None
        view_122: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_25, [384, 512, 512]);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_116: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_154, [16384, 768])
        permute_59: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_32: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg93_1, view_116, permute_59);  arg93_1 = view_116 = permute_59 = None
        view_117: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [32, 512, 768]);  addmm_32 = None
        view_118: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_117, [32, 512, -1, 64]);  view_117 = None
        permute_60: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_26: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_60, [32, 12, 512, 64]);  permute_60 = None
        clone_33: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_123: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [384, 512, 64]);  clone_33 = None
        bmm_11: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
        view_124: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [32, 12, 512, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_34: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_125: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [32, 512, -1]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [16384, 768]);  view_125 = None
        permute_63: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_33: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg95_1, view_126, permute_63);  arg95_1 = view_126 = permute_63 = None
        view_127: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [32, 512, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_49: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_127, convert_element_type_154);  view_127 = convert_element_type_154 = None
        convert_element_type_173: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.float32);  add_49 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_173, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_173, getitem_23);  convert_element_type_173 = getitem_23 = None
        add_50: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_50: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_51: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, arg96_1);  mul_50 = arg96_1 = None
        add_51: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, arg97_1);  mul_51 = arg97_1 = None
        convert_element_type_174: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_174, [16384, 768])
        permute_64: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_34: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg99_1, view_128, permute_64);  arg99_1 = view_128 = permute_64 = None
        view_129: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 512, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_178: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_52: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_178, 0.5)
        mul_53: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_178, 0.7071067811865476);  convert_element_type_178 = None
        erf_5: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
        add_52: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_54: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, add_52);  mul_52 = add_52 = None
        convert_element_type_179: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_179, [16384, 3072]);  convert_element_type_179 = None
        permute_65: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_35: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg101_1, view_130, permute_65);  arg101_1 = view_130 = permute_65 = None
        view_131: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [32, 512, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_53: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_131, convert_element_type_174);  view_131 = convert_element_type_174 = None
        convert_element_type_183: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.float32);  add_53 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_183, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_12: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_12 = None
        scalar_tensor_13: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_13 = None
        full_default_12: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_18: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_183, getitem_25);  convert_element_type_183 = getitem_25 = None
        add_54: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_55: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_56: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, arg102_1);  mul_55 = arg102_1 = None
        add_55: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_56, arg103_1);  mul_56 = arg103_1 = None
        convert_element_type_184: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_132: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_184, [16384, 768])
        permute_66: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_36: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg105_1, view_132, permute_66);  arg105_1 = view_132 = permute_66 = None
        view_133: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [32, 512, 768]);  addmm_36 = None
        view_134: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [32, 512, -1, 64]);  view_133 = None
        permute_67: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_57: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_67, 0.3535533905932738);  permute_67 = None
        expand_27: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_57, [32, 12, 512, 64]);  mul_57 = None
        clone_37: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_141: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [384, 512, 64]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_135: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_184, [16384, 768])
        permute_68: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_37: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg107_1, view_135, permute_68);  arg107_1 = view_135 = permute_68 = None
        view_136: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [32, 512, 768]);  addmm_37 = None
        view_137: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_136, [32, 512, -1, 64]);  view_136 = None
        permute_69: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_72: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_69, [0, 1, 3, 2]);  permute_69 = None
        mul_58: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_72, 0.3535533905932738);  permute_72 = None
        expand_28: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_58, [32, 12, 64, 512]);  mul_58 = None
        clone_38: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_142: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [384, 64, 512]);  clone_38 = None
        bmm_12: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_141, view_142);  view_141 = view_142 = None
        view_143: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [32, 12, 512, 512]);  bmm_12 = None
        eq_6: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_143, -inf)
        logical_not_12: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        full_default_13: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_196: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None
        amax_6: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_196, [-1], True)
        sub_19: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_196, amax_6);  convert_element_type_196 = amax_6 = None
        exp_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_197: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        where_13: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_13, full_default_13, convert_element_type_197);  logical_not_13 = full_default_13 = convert_element_type_197 = None
        expand_29: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_13, [32, 12, 512, 512]);  where_13 = None
        view_144: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_29, [384, 512, 512]);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_138: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_184, [16384, 768])
        permute_70: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_38: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg109_1, view_138, permute_70);  arg109_1 = view_138 = permute_70 = None
        view_139: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [32, 512, 768]);  addmm_38 = None
        view_140: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_139, [32, 512, -1, 64]);  view_139 = None
        permute_71: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_30: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_71, [32, 12, 512, 64]);  permute_71 = None
        clone_39: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_145: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [384, 512, 64]);  clone_39 = None
        bmm_13: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_144, view_145);  view_144 = view_145 = None
        view_146: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [32, 12, 512, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_40: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_147: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [32, 512, -1]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [16384, 768]);  view_147 = None
        permute_74: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_39: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg111_1, view_148, permute_74);  arg111_1 = view_148 = permute_74 = None
        view_149: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [32, 512, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_149, convert_element_type_184);  view_149 = convert_element_type_184 = None
        convert_element_type_203: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_203, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_203, getitem_27);  convert_element_type_203 = getitem_27 = None
        add_58: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_59: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_60: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, arg112_1);  mul_59 = arg112_1 = None
        add_59: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, arg113_1);  mul_60 = arg113_1 = None
        convert_element_type_204: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_204, [16384, 768])
        permute_75: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_40: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg115_1, view_150, permute_75);  arg115_1 = view_150 = permute_75 = None
        view_151: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_208: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_61: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_208, 0.5)
        mul_62: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_208, 0.7071067811865476);  convert_element_type_208 = None
        erf_6: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_60: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_63: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, add_60);  mul_61 = add_60 = None
        convert_element_type_209: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_209, [16384, 3072]);  convert_element_type_209 = None
        permute_76: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_41: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg117_1, view_152, permute_76);  arg117_1 = view_152 = permute_76 = None
        view_153: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [32, 512, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_61: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_153, convert_element_type_204);  view_153 = convert_element_type_204 = None
        convert_element_type_213: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.float32);  add_61 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_213, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_14: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_14 = None
        scalar_tensor_15: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_15 = None
        full_default_14: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_21: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_213, getitem_29);  convert_element_type_213 = getitem_29 = None
        add_62: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_64: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_65: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, arg118_1);  mul_64 = arg118_1 = None
        add_63: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, arg119_1);  mul_65 = arg119_1 = None
        convert_element_type_214: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_154: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_214, [16384, 768])
        permute_77: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_42: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg121_1, view_154, permute_77);  arg121_1 = view_154 = permute_77 = None
        view_155: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [32, 512, 768]);  addmm_42 = None
        view_156: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [32, 512, -1, 64]);  view_155 = None
        permute_78: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_66: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_78, 0.3535533905932738);  permute_78 = None
        expand_31: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_66, [32, 12, 512, 64]);  mul_66 = None
        clone_43: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_163: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [384, 512, 64]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_157: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_214, [16384, 768])
        permute_79: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_43: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg123_1, view_157, permute_79);  arg123_1 = view_157 = permute_79 = None
        view_158: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [32, 512, 768]);  addmm_43 = None
        view_159: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_158, [32, 512, -1, 64]);  view_158 = None
        permute_80: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_83: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        mul_67: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_83, 0.3535533905932738);  permute_83 = None
        expand_32: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_67, [32, 12, 64, 512]);  mul_67 = None
        clone_44: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_164: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [384, 64, 512]);  clone_44 = None
        bmm_14: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_163, view_164);  view_163 = view_164 = None
        view_165: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [32, 12, 512, 512]);  bmm_14 = None
        eq_7: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_165, -inf)
        logical_not_14: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        full_default_15: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_226: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_165, torch.float32);  view_165 = None
        amax_7: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_226, [-1], True)
        sub_22: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_226, amax_7);  convert_element_type_226 = amax_7 = None
        exp_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_227: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        where_15: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_15, full_default_15, convert_element_type_227);  logical_not_15 = full_default_15 = convert_element_type_227 = None
        expand_33: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_15, [32, 12, 512, 512]);  where_15 = None
        view_166: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_33, [384, 512, 512]);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_160: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_214, [16384, 768])
        permute_81: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_44: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg125_1, view_160, permute_81);  arg125_1 = view_160 = permute_81 = None
        view_161: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [32, 512, 768]);  addmm_44 = None
        view_162: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [32, 512, -1, 64]);  view_161 = None
        permute_82: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_34: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_82, [32, 12, 512, 64]);  permute_82 = None
        clone_45: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_167: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [384, 512, 64]);  clone_45 = None
        bmm_15: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
        view_168: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [32, 12, 512, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_46: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [32, 512, -1]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [16384, 768]);  view_169 = None
        permute_85: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_45: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg127_1, view_170, permute_85);  arg127_1 = view_170 = permute_85 = None
        view_171: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [32, 512, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_65: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_171, convert_element_type_214);  view_171 = convert_element_type_214 = None
        convert_element_type_233: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.float32);  add_65 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_233, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_233, getitem_31);  convert_element_type_233 = getitem_31 = None
        add_66: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_68: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_69: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, arg128_1);  mul_68 = arg128_1 = None
        add_67: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, arg129_1);  mul_69 = arg129_1 = None
        convert_element_type_234: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_234, [16384, 768])
        permute_86: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_46: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg131_1, view_172, permute_86);  arg131_1 = view_172 = permute_86 = None
        view_173: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [32, 512, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_238: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_70: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_238, 0.5)
        mul_71: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_238, 0.7071067811865476);  convert_element_type_238 = None
        erf_7: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_68: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_72: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, add_68);  mul_70 = add_68 = None
        convert_element_type_239: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_239, [16384, 3072]);  convert_element_type_239 = None
        permute_87: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_47: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg133_1, view_174, permute_87);  arg133_1 = view_174 = permute_87 = None
        view_175: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [32, 512, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_69: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_175, convert_element_type_234);  view_175 = convert_element_type_234 = None
        convert_element_type_243: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.float32);  add_69 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_243, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_16: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_16 = None
        scalar_tensor_17: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_17 = None
        full_default_16: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_24: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_243, getitem_33);  convert_element_type_243 = getitem_33 = None
        add_70: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_73: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_74: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, arg134_1);  mul_73 = arg134_1 = None
        add_71: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, arg135_1);  mul_74 = arg135_1 = None
        convert_element_type_244: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_176: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_244, [16384, 768])
        permute_88: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_48: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg137_1, view_176, permute_88);  arg137_1 = view_176 = permute_88 = None
        view_177: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [32, 512, 768]);  addmm_48 = None
        view_178: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_177, [32, 512, -1, 64]);  view_177 = None
        permute_89: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_75: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_89, 0.3535533905932738);  permute_89 = None
        expand_35: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_75, [32, 12, 512, 64]);  mul_75 = None
        clone_49: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_185: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [384, 512, 64]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_179: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_244, [16384, 768])
        permute_90: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_49: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg139_1, view_179, permute_90);  arg139_1 = view_179 = permute_90 = None
        view_180: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [32, 512, 768]);  addmm_49 = None
        view_181: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_180, [32, 512, -1, 64]);  view_180 = None
        permute_91: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_94: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        mul_76: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_94, 0.3535533905932738);  permute_94 = None
        expand_36: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_76, [32, 12, 64, 512]);  mul_76 = None
        clone_50: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_186: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [384, 64, 512]);  clone_50 = None
        bmm_16: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_185, view_186);  view_185 = view_186 = None
        view_187: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [32, 12, 512, 512]);  bmm_16 = None
        eq_8: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_187, -inf)
        logical_not_16: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        full_default_17: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_256: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_187, torch.float32);  view_187 = None
        amax_8: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_256, [-1], True)
        sub_25: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_256, amax_8);  convert_element_type_256 = amax_8 = None
        exp_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_257: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        where_17: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_17, full_default_17, convert_element_type_257);  logical_not_17 = full_default_17 = convert_element_type_257 = None
        expand_37: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_17, [32, 12, 512, 512]);  where_17 = None
        view_188: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_37, [384, 512, 512]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_182: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_244, [16384, 768])
        permute_92: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_50: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg141_1, view_182, permute_92);  arg141_1 = view_182 = permute_92 = None
        view_183: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [32, 512, 768]);  addmm_50 = None
        view_184: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_183, [32, 512, -1, 64]);  view_183 = None
        permute_93: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_38: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_93, [32, 12, 512, 64]);  permute_93 = None
        clone_51: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_189: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [384, 512, 64]);  clone_51 = None
        bmm_17: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_188, view_189);  view_188 = view_189 = None
        view_190: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [32, 12, 512, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_52: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_191: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [32, 512, -1]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [16384, 768]);  view_191 = None
        permute_96: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_51: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg143_1, view_192, permute_96);  arg143_1 = view_192 = permute_96 = None
        view_193: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [32, 512, 768]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_73: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_193, convert_element_type_244);  view_193 = convert_element_type_244 = None
        convert_element_type_263: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.float32);  add_73 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_263, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_263, getitem_35);  convert_element_type_263 = getitem_35 = None
        add_74: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_77: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_78: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, arg144_1);  mul_77 = arg144_1 = None
        add_75: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, arg145_1);  mul_78 = arg145_1 = None
        convert_element_type_264: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_264, [16384, 768])
        permute_97: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_52: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg147_1, view_194, permute_97);  arg147_1 = view_194 = permute_97 = None
        view_195: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [32, 512, 3072]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_268: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_79: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_268, 0.5)
        mul_80: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_268, 0.7071067811865476);  convert_element_type_268 = None
        erf_8: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_80);  mul_80 = None
        add_76: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_81: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, add_76);  mul_79 = add_76 = None
        convert_element_type_269: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_269, [16384, 3072]);  convert_element_type_269 = None
        permute_98: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_53: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg149_1, view_196, permute_98);  arg149_1 = view_196 = permute_98 = None
        view_197: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [32, 512, 768]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_77: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_197, convert_element_type_264);  view_197 = convert_element_type_264 = None
        convert_element_type_273: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.float32);  add_77 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_273, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_18: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_18 = None
        scalar_tensor_19: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_19 = None
        full_default_18: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_27: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_273, getitem_37);  convert_element_type_273 = getitem_37 = None
        add_78: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_82: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_83: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, arg150_1);  mul_82 = arg150_1 = None
        add_79: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, arg151_1);  mul_83 = arg151_1 = None
        convert_element_type_274: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_198: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_274, [16384, 768])
        permute_99: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_54: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg153_1, view_198, permute_99);  arg153_1 = view_198 = permute_99 = None
        view_199: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [32, 512, 768]);  addmm_54 = None
        view_200: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [32, 512, -1, 64]);  view_199 = None
        permute_100: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_84: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_100, 0.3535533905932738);  permute_100 = None
        expand_39: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_84, [32, 12, 512, 64]);  mul_84 = None
        clone_55: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_207: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_55, [384, 512, 64]);  clone_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_201: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_274, [16384, 768])
        permute_101: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_55: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg155_1, view_201, permute_101);  arg155_1 = view_201 = permute_101 = None
        view_202: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [32, 512, 768]);  addmm_55 = None
        view_203: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_202, [32, 512, -1, 64]);  view_202 = None
        permute_102: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_105: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_102, [0, 1, 3, 2]);  permute_102 = None
        mul_85: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_105, 0.3535533905932738);  permute_105 = None
        expand_40: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_85, [32, 12, 64, 512]);  mul_85 = None
        clone_56: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_208: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_56, [384, 64, 512]);  clone_56 = None
        bmm_18: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_207, view_208);  view_207 = view_208 = None
        view_209: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [32, 12, 512, 512]);  bmm_18 = None
        eq_9: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_209, -inf)
        logical_not_18: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        full_default_19: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_286: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        amax_9: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_286, [-1], True)
        sub_28: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_286, amax_9);  convert_element_type_286 = amax_9 = None
        exp_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_287: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        where_19: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_19, full_default_19, convert_element_type_287);  logical_not_19 = full_default_19 = convert_element_type_287 = None
        expand_41: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_19, [32, 12, 512, 512]);  where_19 = None
        view_210: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_41, [384, 512, 512]);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_204: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_274, [16384, 768])
        permute_103: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_56: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg157_1, view_204, permute_103);  arg157_1 = view_204 = permute_103 = None
        view_205: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [32, 512, 768]);  addmm_56 = None
        view_206: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_205, [32, 512, -1, 64]);  view_205 = None
        permute_104: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_42: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_104, [32, 12, 512, 64]);  permute_104 = None
        clone_57: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_211: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [384, 512, 64]);  clone_57 = None
        bmm_19: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_210, view_211);  view_210 = view_211 = None
        view_212: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [32, 12, 512, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_58: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_213: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [32, 512, -1]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_213, [16384, 768]);  view_213 = None
        permute_107: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_57: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg159_1, view_214, permute_107);  arg159_1 = view_214 = permute_107 = None
        view_215: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [32, 512, 768]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_81: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_215, convert_element_type_274);  view_215 = convert_element_type_274 = None
        convert_element_type_293: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_293, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_293, getitem_39);  convert_element_type_293 = getitem_39 = None
        add_82: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_86: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_87: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, arg160_1);  mul_86 = arg160_1 = None
        add_83: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_87, arg161_1);  mul_87 = arg161_1 = None
        convert_element_type_294: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_294, [16384, 768])
        permute_108: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_58: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg163_1, view_216, permute_108);  arg163_1 = view_216 = permute_108 = None
        view_217: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 3072]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_298: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_88: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_298, 0.5)
        mul_89: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_298, 0.7071067811865476);  convert_element_type_298 = None
        erf_9: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
        add_84: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_90: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, add_84);  mul_88 = add_84 = None
        convert_element_type_299: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_299, [16384, 3072]);  convert_element_type_299 = None
        permute_109: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_59: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg165_1, view_218, permute_109);  arg165_1 = view_218 = permute_109 = None
        view_219: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [32, 512, 768]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_85: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_219, convert_element_type_294);  view_219 = convert_element_type_294 = None
        convert_element_type_303: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.float32);  add_85 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_303, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_20: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_20 = None
        scalar_tensor_21: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_21 = None
        full_default_20: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_30: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_303, getitem_41);  convert_element_type_303 = getitem_41 = None
        add_86: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_91: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_92: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, arg166_1);  mul_91 = arg166_1 = None
        add_87: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, arg167_1);  mul_92 = arg167_1 = None
        convert_element_type_304: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_220: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_304, [16384, 768])
        permute_110: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_60: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg169_1, view_220, permute_110);  arg169_1 = view_220 = permute_110 = None
        view_221: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [32, 512, 768]);  addmm_60 = None
        view_222: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [32, 512, -1, 64]);  view_221 = None
        permute_111: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_93: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_111, 0.3535533905932738);  permute_111 = None
        expand_43: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_93, [32, 12, 512, 64]);  mul_93 = None
        clone_61: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_229: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [384, 512, 64]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_223: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_304, [16384, 768])
        permute_112: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_61: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg171_1, view_223, permute_112);  arg171_1 = view_223 = permute_112 = None
        view_224: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [32, 512, 768]);  addmm_61 = None
        view_225: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_224, [32, 512, -1, 64]);  view_224 = None
        permute_113: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_116: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_113, [0, 1, 3, 2]);  permute_113 = None
        mul_94: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_116, 0.3535533905932738);  permute_116 = None
        expand_44: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_94, [32, 12, 64, 512]);  mul_94 = None
        clone_62: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_230: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_62, [384, 64, 512]);  clone_62 = None
        bmm_20: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_229, view_230);  view_229 = view_230 = None
        view_231: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [32, 12, 512, 512]);  bmm_20 = None
        eq_10: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_231, -inf)
        logical_not_20: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        full_default_21: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_316: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_231, torch.float32);  view_231 = None
        amax_10: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_316, [-1], True)
        sub_31: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_316, amax_10);  convert_element_type_316 = amax_10 = None
        exp_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_317: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        where_21: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_21, full_default_21, convert_element_type_317);  logical_not_21 = full_default_21 = convert_element_type_317 = None
        expand_45: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_21, [32, 12, 512, 512]);  where_21 = None
        view_232: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_45, [384, 512, 512]);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_226: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_304, [16384, 768])
        permute_114: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_62: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg173_1, view_226, permute_114);  arg173_1 = view_226 = permute_114 = None
        view_227: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [32, 512, 768]);  addmm_62 = None
        view_228: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_227, [32, 512, -1, 64]);  view_227 = None
        permute_115: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_46: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_115, [32, 12, 512, 64]);  permute_115 = None
        clone_63: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_233: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [384, 512, 64]);  clone_63 = None
        bmm_21: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_232, view_233);  view_232 = view_233 = None
        view_234: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [32, 12, 512, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_64: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [32, 512, -1]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [16384, 768]);  view_235 = None
        permute_118: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        addmm_63: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg175_1, view_236, permute_118);  arg175_1 = view_236 = permute_118 = None
        view_237: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [32, 512, 768]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_89: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_237, convert_element_type_304);  view_237 = convert_element_type_304 = None
        convert_element_type_323: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.float32);  add_89 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_323, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_323, getitem_43);  convert_element_type_323 = getitem_43 = None
        add_90: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_95: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_96: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, arg176_1);  mul_95 = arg176_1 = None
        add_91: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_96, arg177_1);  mul_96 = arg177_1 = None
        convert_element_type_324: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_324, [16384, 768])
        permute_119: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_64: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg179_1, view_238, permute_119);  arg179_1 = view_238 = permute_119 = None
        view_239: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [32, 512, 3072]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_328: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_97: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_328, 0.5)
        mul_98: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_328, 0.7071067811865476);  convert_element_type_328 = None
        erf_10: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
        add_92: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_99: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, add_92);  mul_97 = add_92 = None
        convert_element_type_329: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_329, [16384, 3072]);  convert_element_type_329 = None
        permute_120: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_65: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg181_1, view_240, permute_120);  arg181_1 = view_240 = permute_120 = None
        view_241: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [32, 512, 768]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_93: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_241, convert_element_type_324);  view_241 = convert_element_type_324 = None
        convert_element_type_333: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.float32);  add_93 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_333, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_22: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0));  scalar_tensor_22 = None
        scalar_tensor_23: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor_23 = None
        full_default_22: "bf16[32, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 512, 512], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_33: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_333, getitem_45);  convert_element_type_333 = getitem_45 = None
        add_94: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_100: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_101: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, arg182_1);  mul_100 = arg182_1 = None
        add_95: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, arg183_1);  mul_101 = arg183_1 = None
        convert_element_type_334: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16);  add_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_242: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_334, [16384, 768])
        permute_121: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_66: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg185_1, view_242, permute_121);  arg185_1 = view_242 = permute_121 = None
        view_243: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [32, 512, 768]);  addmm_66 = None
        view_244: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_243, [32, 512, -1, 64]);  view_243 = None
        permute_122: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_102: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.mul.Scalar(permute_122, 0.3535533905932738);  permute_122 = None
        expand_47: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(mul_102, [32, 12, 512, 64]);  mul_102 = None
        clone_67: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_251: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [384, 512, 64]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_245: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_334, [16384, 768])
        permute_123: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_67: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg187_1, view_245, permute_123);  arg187_1 = view_245 = permute_123 = None
        view_246: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [32, 512, 768]);  addmm_67 = None
        view_247: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_246, [32, 512, -1, 64]);  view_246 = None
        permute_124: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_127: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
        mul_103: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.mul.Scalar(permute_127, 0.3535533905932738);  permute_127 = None
        expand_48: "bf16[32, 12, 64, 512][393216, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(mul_103, [32, 12, 64, 512]);  mul_103 = None
        clone_68: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_252: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [384, 64, 512]);  clone_68 = None
        bmm_22: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_251, view_252);  view_251 = view_252 = None
        view_253: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [32, 12, 512, 512]);  bmm_22 = None
        eq_11: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Scalar(view_253, -inf)
        logical_not_22: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        full_default_23: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_346: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_253, torch.float32);  view_253 = None
        amax_11: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_346, [-1], True)
        sub_34: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_346, amax_11);  convert_element_type_346 = amax_11 = None
        exp_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_347: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        where_23: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_23, full_default_23, convert_element_type_347);  logical_not_23 = full_default_23 = convert_element_type_347 = None
        expand_49: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.expand.default(where_23, [32, 12, 512, 512]);  where_23 = None
        view_254: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_49, [384, 512, 512]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_248: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_334, [16384, 768])
        permute_125: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_68: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg189_1, view_248, permute_125);  arg189_1 = view_248 = permute_125 = None
        view_249: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [32, 512, 768]);  addmm_68 = None
        view_250: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_249, [32, 512, -1, 64]);  view_249 = None
        permute_126: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_50: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_126, [32, 12, 512, 64]);  permute_126 = None
        clone_69: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_255: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [384, 512, 64]);  clone_69 = None
        bmm_23: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_254, view_255);  view_254 = view_255 = None
        view_256: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [32, 12, 512, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_70: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:253 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_257: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [32, 512, -1]);  clone_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        view_258: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [16384, 768]);  view_257 = None
        permute_129: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_69: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg191_1, view_258, permute_129);  arg191_1 = view_258 = permute_129 = None
        view_259: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [32, 512, 768]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_97: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_259, convert_element_type_334);  view_259 = convert_element_type_334 = None
        convert_element_type_353: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.float32);  add_97 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_353, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_353, getitem_47);  convert_element_type_353 = getitem_47 = None
        add_98: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_104: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_105: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, arg192_1);  mul_104 = arg192_1 = None
        add_99: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_105, arg193_1);  mul_105 = arg193_1 = None
        convert_element_type_354: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_354, [16384, 768])
        permute_130: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_70: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg195_1, view_260, permute_130);  arg195_1 = view_260 = permute_130 = None
        view_261: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [32, 512, 3072]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_358: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_106: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_358, 0.5)
        mul_107: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_358, 0.7071067811865476);  convert_element_type_358 = None
        erf_11: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
        add_100: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_108: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, add_100);  mul_106 = add_100 = None
        convert_element_type_359: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_108, torch.bfloat16);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_359, [16384, 3072]);  convert_element_type_359 = None
        permute_131: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        addmm_71: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg197_1, view_262, permute_131);  arg197_1 = view_262 = permute_131 = None
        view_263: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [32, 512, 768]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_101: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_263, convert_element_type_354);  view_263 = convert_element_type_354 = None
        convert_element_type_363: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.float32);  add_101 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_363, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        sub_36: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_363, getitem_49);  convert_element_type_363 = getitem_49 = None
        add_102: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_109: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_110: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_109, arg198_1);  mul_109 = arg198_1 = None
        add_103: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_110, arg199_1);  mul_110 = arg199_1 = None
        convert_element_type_364: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:893 in forward, code: x = self.dense(features)
        view_264: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_364, [16384, 768]);  convert_element_type_364 = None
        permute_132: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_72: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg201_1, view_264, permute_132);  arg201_1 = view_264 = permute_132 = None
        view_265: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [32, 512, 768]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_368: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        mul_111: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_368, 0.5)
        mul_112: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_368, 0.7071067811865476);  convert_element_type_368 = None
        erf_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_104: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_113: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, add_104);  mul_111 = add_104 = None
        convert_element_type_369: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:895 in forward, code: x = self.layer_norm(x)
        convert_element_type_370: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_369, torch.float32);  convert_element_type_369 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_370, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[32, 513][513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(arg0_1, [0, 1], -100.0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[32, 512][513, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_73: "i64[32, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_269: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [-1]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        ne_2: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_269, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:898 in forward, code: x = self.decoder(x)
        full_default_26: "bf16[7][1]cuda:0" = torch.ops.aten.full.default([7], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "bf16[50272][1]cuda:0" = torch.ops.aten.cat.default([arg204_1, full_default_26]);  arg204_1 = full_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:895 in forward, code: x = self.layer_norm(x)
        sub_37: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_370, getitem_51);  convert_element_type_370 = getitem_51 = None
        add_105: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        mul_114: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = rsqrt_25 = None
        mul_115: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, arg202_1);  mul_114 = arg202_1 = None
        add_106: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, arg203_1);  mul_115 = arg203_1 = None
        convert_element_type_371: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16);  add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:898 in forward, code: x = self.decoder(x)
        view_266: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_371, [16384, 768]);  convert_element_type_371 = None
        permute_133: "bf16[768, 50265][1, 768]cuda:0" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        constant_pad_nd_default: "bf16[768, 50272][50272, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_133, [0, 7, 0, 0]);  permute_133 = None
        addmm_default: "bf16[16384, 50272][50272, 1]cuda:0" = torch.ops.aten.addmm.default(cat_default, view_266, constant_pad_nd_default);  cat_default = view_266 = constant_pad_nd_default = None
        slice_tensor: "bf16[16384, 50265][50272, 1]cuda:0" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -7);  addmm_default = None
        view_267: "bf16[32, 512, 50265][25739264, 50272, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [32, 512, 50265]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_375: "f32[32, 512, 50265][25735680, 50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_267, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_268: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_375, [-1, 50265]);  convert_element_type_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_12: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(view_268, [1], True)
        sub_38: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = amax_12 = None
        exp_12: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.exp.default(sub_38)
        sum_13: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
        ne_1: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_269, -100)
        full_default_24: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[16384][1]cuda:0" = torch.ops.aten.where.self(ne_1, view_269, full_default_24);  ne_1 = full_default_24 = None
        unsqueeze_3: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather_1: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_3);  sub_39 = unsqueeze_3 = None
        squeeze: "f32[16384][1]cuda:0" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg: "f32[16384][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_25: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[16384][1]cuda:0" = torch.ops.aten.where.self(ne_2, neg, full_default_25);  ne_2 = neg = full_default_25 = None
        sum_15: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_25);  where_25 = None
        ne_3: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_269, -100);  view_269 = None
        sum_14: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
        convert_element_type_376: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        div_12: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_376);  sum_15 = convert_element_type_376 = None
        return (div_12, view_267)
