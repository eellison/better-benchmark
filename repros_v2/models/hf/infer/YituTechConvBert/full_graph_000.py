class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[32, 512][512, 1]cuda:0", arg1_1: "i64[1, 512][512, 1]cuda:0", arg2_1: "bf16[30522, 768][768, 1]cuda:0", arg3_1: "i64[1, 512][512, 1]cuda:0", arg4_1: "bf16[512, 768][768, 1]cuda:0", arg5_1: "bf16[2, 768][768, 1]cuda:0", arg6_1: "bf16[768][1]cuda:0", arg7_1: "bf16[768][1]cuda:0", arg8_1: "bf16[384, 768][768, 1]cuda:0", arg9_1: "bf16[384][1]cuda:0", arg10_1: "bf16[384, 768][768, 1]cuda:0", arg11_1: "bf16[384][1]cuda:0", arg12_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg13_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg14_1: "bf16[384, 1][1, 1]cuda:0", arg15_1: "bf16[384, 768][768, 1]cuda:0", arg16_1: "bf16[384][1]cuda:0", arg17_1: "bf16[54, 384][384, 1]cuda:0", arg18_1: "bf16[54][1]cuda:0", arg19_1: "bf16[384, 768][768, 1]cuda:0", arg20_1: "bf16[384][1]cuda:0", arg21_1: "bf16[768, 768][768, 1]cuda:0", arg22_1: "bf16[768][1]cuda:0", arg23_1: "bf16[768][1]cuda:0", arg24_1: "bf16[768][1]cuda:0", arg25_1: "bf16[3072, 768][768, 1]cuda:0", arg26_1: "bf16[3072][1]cuda:0", arg27_1: "bf16[768, 3072][3072, 1]cuda:0", arg28_1: "bf16[768][1]cuda:0", arg29_1: "bf16[768][1]cuda:0", arg30_1: "bf16[768][1]cuda:0", arg31_1: "bf16[384, 768][768, 1]cuda:0", arg32_1: "bf16[384][1]cuda:0", arg33_1: "bf16[384, 768][768, 1]cuda:0", arg34_1: "bf16[384][1]cuda:0", arg35_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg36_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg37_1: "bf16[384, 1][1, 1]cuda:0", arg38_1: "bf16[384, 768][768, 1]cuda:0", arg39_1: "bf16[384][1]cuda:0", arg40_1: "bf16[54, 384][384, 1]cuda:0", arg41_1: "bf16[54][1]cuda:0", arg42_1: "bf16[384, 768][768, 1]cuda:0", arg43_1: "bf16[384][1]cuda:0", arg44_1: "bf16[768, 768][768, 1]cuda:0", arg45_1: "bf16[768][1]cuda:0", arg46_1: "bf16[768][1]cuda:0", arg47_1: "bf16[768][1]cuda:0", arg48_1: "bf16[3072, 768][768, 1]cuda:0", arg49_1: "bf16[3072][1]cuda:0", arg50_1: "bf16[768, 3072][3072, 1]cuda:0", arg51_1: "bf16[768][1]cuda:0", arg52_1: "bf16[768][1]cuda:0", arg53_1: "bf16[768][1]cuda:0", arg54_1: "bf16[384, 768][768, 1]cuda:0", arg55_1: "bf16[384][1]cuda:0", arg56_1: "bf16[384, 768][768, 1]cuda:0", arg57_1: "bf16[384][1]cuda:0", arg58_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg59_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg60_1: "bf16[384, 1][1, 1]cuda:0", arg61_1: "bf16[384, 768][768, 1]cuda:0", arg62_1: "bf16[384][1]cuda:0", arg63_1: "bf16[54, 384][384, 1]cuda:0", arg64_1: "bf16[54][1]cuda:0", arg65_1: "bf16[384, 768][768, 1]cuda:0", arg66_1: "bf16[384][1]cuda:0", arg67_1: "bf16[768, 768][768, 1]cuda:0", arg68_1: "bf16[768][1]cuda:0", arg69_1: "bf16[768][1]cuda:0", arg70_1: "bf16[768][1]cuda:0", arg71_1: "bf16[3072, 768][768, 1]cuda:0", arg72_1: "bf16[3072][1]cuda:0", arg73_1: "bf16[768, 3072][3072, 1]cuda:0", arg74_1: "bf16[768][1]cuda:0", arg75_1: "bf16[768][1]cuda:0", arg76_1: "bf16[768][1]cuda:0", arg77_1: "bf16[384, 768][768, 1]cuda:0", arg78_1: "bf16[384][1]cuda:0", arg79_1: "bf16[384, 768][768, 1]cuda:0", arg80_1: "bf16[384][1]cuda:0", arg81_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg82_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg83_1: "bf16[384, 1][1, 1]cuda:0", arg84_1: "bf16[384, 768][768, 1]cuda:0", arg85_1: "bf16[384][1]cuda:0", arg86_1: "bf16[54, 384][384, 1]cuda:0", arg87_1: "bf16[54][1]cuda:0", arg88_1: "bf16[384, 768][768, 1]cuda:0", arg89_1: "bf16[384][1]cuda:0", arg90_1: "bf16[768, 768][768, 1]cuda:0", arg91_1: "bf16[768][1]cuda:0", arg92_1: "bf16[768][1]cuda:0", arg93_1: "bf16[768][1]cuda:0", arg94_1: "bf16[3072, 768][768, 1]cuda:0", arg95_1: "bf16[3072][1]cuda:0", arg96_1: "bf16[768, 3072][3072, 1]cuda:0", arg97_1: "bf16[768][1]cuda:0", arg98_1: "bf16[768][1]cuda:0", arg99_1: "bf16[768][1]cuda:0", arg100_1: "bf16[384, 768][768, 1]cuda:0", arg101_1: "bf16[384][1]cuda:0", arg102_1: "bf16[384, 768][768, 1]cuda:0", arg103_1: "bf16[384][1]cuda:0", arg104_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg105_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg106_1: "bf16[384, 1][1, 1]cuda:0", arg107_1: "bf16[384, 768][768, 1]cuda:0", arg108_1: "bf16[384][1]cuda:0", arg109_1: "bf16[54, 384][384, 1]cuda:0", arg110_1: "bf16[54][1]cuda:0", arg111_1: "bf16[384, 768][768, 1]cuda:0", arg112_1: "bf16[384][1]cuda:0", arg113_1: "bf16[768, 768][768, 1]cuda:0", arg114_1: "bf16[768][1]cuda:0", arg115_1: "bf16[768][1]cuda:0", arg116_1: "bf16[768][1]cuda:0", arg117_1: "bf16[3072, 768][768, 1]cuda:0", arg118_1: "bf16[3072][1]cuda:0", arg119_1: "bf16[768, 3072][3072, 1]cuda:0", arg120_1: "bf16[768][1]cuda:0", arg121_1: "bf16[768][1]cuda:0", arg122_1: "bf16[768][1]cuda:0", arg123_1: "bf16[384, 768][768, 1]cuda:0", arg124_1: "bf16[384][1]cuda:0", arg125_1: "bf16[384, 768][768, 1]cuda:0", arg126_1: "bf16[384][1]cuda:0", arg127_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg128_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg129_1: "bf16[384, 1][1, 1]cuda:0", arg130_1: "bf16[384, 768][768, 1]cuda:0", arg131_1: "bf16[384][1]cuda:0", arg132_1: "bf16[54, 384][384, 1]cuda:0", arg133_1: "bf16[54][1]cuda:0", arg134_1: "bf16[384, 768][768, 1]cuda:0", arg135_1: "bf16[384][1]cuda:0", arg136_1: "bf16[768, 768][768, 1]cuda:0", arg137_1: "bf16[768][1]cuda:0", arg138_1: "bf16[768][1]cuda:0", arg139_1: "bf16[768][1]cuda:0", arg140_1: "bf16[3072, 768][768, 1]cuda:0", arg141_1: "bf16[3072][1]cuda:0", arg142_1: "bf16[768, 3072][3072, 1]cuda:0", arg143_1: "bf16[768][1]cuda:0", arg144_1: "bf16[768][1]cuda:0", arg145_1: "bf16[768][1]cuda:0", arg146_1: "bf16[384, 768][768, 1]cuda:0", arg147_1: "bf16[384][1]cuda:0", arg148_1: "bf16[384, 768][768, 1]cuda:0", arg149_1: "bf16[384][1]cuda:0", arg150_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg151_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg152_1: "bf16[384, 1][1, 1]cuda:0", arg153_1: "bf16[384, 768][768, 1]cuda:0", arg154_1: "bf16[384][1]cuda:0", arg155_1: "bf16[54, 384][384, 1]cuda:0", arg156_1: "bf16[54][1]cuda:0", arg157_1: "bf16[384, 768][768, 1]cuda:0", arg158_1: "bf16[384][1]cuda:0", arg159_1: "bf16[768, 768][768, 1]cuda:0", arg160_1: "bf16[768][1]cuda:0", arg161_1: "bf16[768][1]cuda:0", arg162_1: "bf16[768][1]cuda:0", arg163_1: "bf16[3072, 768][768, 1]cuda:0", arg164_1: "bf16[3072][1]cuda:0", arg165_1: "bf16[768, 3072][3072, 1]cuda:0", arg166_1: "bf16[768][1]cuda:0", arg167_1: "bf16[768][1]cuda:0", arg168_1: "bf16[768][1]cuda:0", arg169_1: "bf16[384, 768][768, 1]cuda:0", arg170_1: "bf16[384][1]cuda:0", arg171_1: "bf16[384, 768][768, 1]cuda:0", arg172_1: "bf16[384][1]cuda:0", arg173_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg174_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg175_1: "bf16[384, 1][1, 1]cuda:0", arg176_1: "bf16[384, 768][768, 1]cuda:0", arg177_1: "bf16[384][1]cuda:0", arg178_1: "bf16[54, 384][384, 1]cuda:0", arg179_1: "bf16[54][1]cuda:0", arg180_1: "bf16[384, 768][768, 1]cuda:0", arg181_1: "bf16[384][1]cuda:0", arg182_1: "bf16[768, 768][768, 1]cuda:0", arg183_1: "bf16[768][1]cuda:0", arg184_1: "bf16[768][1]cuda:0", arg185_1: "bf16[768][1]cuda:0", arg186_1: "bf16[3072, 768][768, 1]cuda:0", arg187_1: "bf16[3072][1]cuda:0", arg188_1: "bf16[768, 3072][3072, 1]cuda:0", arg189_1: "bf16[768][1]cuda:0", arg190_1: "bf16[768][1]cuda:0", arg191_1: "bf16[768][1]cuda:0", arg192_1: "bf16[384, 768][768, 1]cuda:0", arg193_1: "bf16[384][1]cuda:0", arg194_1: "bf16[384, 768][768, 1]cuda:0", arg195_1: "bf16[384][1]cuda:0", arg196_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg197_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg198_1: "bf16[384, 1][1, 1]cuda:0", arg199_1: "bf16[384, 768][768, 1]cuda:0", arg200_1: "bf16[384][1]cuda:0", arg201_1: "bf16[54, 384][384, 1]cuda:0", arg202_1: "bf16[54][1]cuda:0", arg203_1: "bf16[384, 768][768, 1]cuda:0", arg204_1: "bf16[384][1]cuda:0", arg205_1: "bf16[768, 768][768, 1]cuda:0", arg206_1: "bf16[768][1]cuda:0", arg207_1: "bf16[768][1]cuda:0", arg208_1: "bf16[768][1]cuda:0", arg209_1: "bf16[3072, 768][768, 1]cuda:0", arg210_1: "bf16[3072][1]cuda:0", arg211_1: "bf16[768, 3072][3072, 1]cuda:0", arg212_1: "bf16[768][1]cuda:0", arg213_1: "bf16[768][1]cuda:0", arg214_1: "bf16[768][1]cuda:0", arg215_1: "bf16[384, 768][768, 1]cuda:0", arg216_1: "bf16[384][1]cuda:0", arg217_1: "bf16[384, 768][768, 1]cuda:0", arg218_1: "bf16[384][1]cuda:0", arg219_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg220_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg221_1: "bf16[384, 1][1, 1]cuda:0", arg222_1: "bf16[384, 768][768, 1]cuda:0", arg223_1: "bf16[384][1]cuda:0", arg224_1: "bf16[54, 384][384, 1]cuda:0", arg225_1: "bf16[54][1]cuda:0", arg226_1: "bf16[384, 768][768, 1]cuda:0", arg227_1: "bf16[384][1]cuda:0", arg228_1: "bf16[768, 768][768, 1]cuda:0", arg229_1: "bf16[768][1]cuda:0", arg230_1: "bf16[768][1]cuda:0", arg231_1: "bf16[768][1]cuda:0", arg232_1: "bf16[3072, 768][768, 1]cuda:0", arg233_1: "bf16[3072][1]cuda:0", arg234_1: "bf16[768, 3072][3072, 1]cuda:0", arg235_1: "bf16[768][1]cuda:0", arg236_1: "bf16[768][1]cuda:0", arg237_1: "bf16[768][1]cuda:0", arg238_1: "bf16[384, 768][768, 1]cuda:0", arg239_1: "bf16[384][1]cuda:0", arg240_1: "bf16[384, 768][768, 1]cuda:0", arg241_1: "bf16[384][1]cuda:0", arg242_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg243_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg244_1: "bf16[384, 1][1, 1]cuda:0", arg245_1: "bf16[384, 768][768, 1]cuda:0", arg246_1: "bf16[384][1]cuda:0", arg247_1: "bf16[54, 384][384, 1]cuda:0", arg248_1: "bf16[54][1]cuda:0", arg249_1: "bf16[384, 768][768, 1]cuda:0", arg250_1: "bf16[384][1]cuda:0", arg251_1: "bf16[768, 768][768, 1]cuda:0", arg252_1: "bf16[768][1]cuda:0", arg253_1: "bf16[768][1]cuda:0", arg254_1: "bf16[768][1]cuda:0", arg255_1: "bf16[3072, 768][768, 1]cuda:0", arg256_1: "bf16[3072][1]cuda:0", arg257_1: "bf16[768, 3072][3072, 1]cuda:0", arg258_1: "bf16[768][1]cuda:0", arg259_1: "bf16[768][1]cuda:0", arg260_1: "bf16[768][1]cuda:0", arg261_1: "bf16[384, 768][768, 1]cuda:0", arg262_1: "bf16[384][1]cuda:0", arg263_1: "bf16[384, 768][768, 1]cuda:0", arg264_1: "bf16[384][1]cuda:0", arg265_1: "bf16[768, 1, 9][9, 9, 1]cuda:0", arg266_1: "bf16[384, 768, 1][768, 1, 1]cuda:0", arg267_1: "bf16[384, 1][1, 1]cuda:0", arg268_1: "bf16[384, 768][768, 1]cuda:0", arg269_1: "bf16[384][1]cuda:0", arg270_1: "bf16[54, 384][384, 1]cuda:0", arg271_1: "bf16[54][1]cuda:0", arg272_1: "bf16[384, 768][768, 1]cuda:0", arg273_1: "bf16[384][1]cuda:0", arg274_1: "bf16[768, 768][768, 1]cuda:0", arg275_1: "bf16[768][1]cuda:0", arg276_1: "bf16[768][1]cuda:0", arg277_1: "bf16[768][1]cuda:0", arg278_1: "bf16[3072, 768][768, 1]cuda:0", arg279_1: "bf16[3072][1]cuda:0", arg280_1: "bf16[768, 3072][3072, 1]cuda:0", arg281_1: "bf16[768][1]cuda:0", arg282_1: "bf16[768][1]cuda:0", arg283_1: "bf16[768][1]cuda:0", arg284_1: "bf16[768, 768][768, 1]cuda:0", arg285_1: "bf16[768][1]cuda:0", arg286_1: "bf16[768][1]cuda:0", arg287_1: "bf16[768][1]cuda:0", arg288_1: "bf16[30522][1]cuda:0", arg289_1: "i64[32, 512][512, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:624 in forward, code: attention_mask = torch.ones(input_shape, device=device)
        full: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.full.default([32, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:998 in get_extended_attention_mask, code: extended_attention_mask = attention_mask[:, None, None, :]
        unsqueeze: "f32[32, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1: "f32[32, 1, 1, 512][512, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:1009 in get_extended_attention_mask, code: extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        convert_element_type: "bf16[32, 1, 1, 512][512, 512, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.bfloat16);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:1010 in get_extended_attention_mask, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        sub: "bf16[32, 1, 1, 512][512, 512, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, convert_element_type);  convert_element_type = sub = None
        full_default: "bf16[32, 1, 1, 512][512, 512, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 1, 512], -0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:99 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:100 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_1: "bf16[1, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg4_1, arg3_1);  arg4_1 = arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:103 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:628 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: "i64[32, 512][0, 1]cuda:0" = torch.ops.aten.expand.default(arg1_1, [32, 512]);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:101 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_2: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.embedding.default(arg5_1, expand);  arg5_1 = expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:103 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add_1: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:104 in forward, code: embeddings = self.LayerNorm(embeddings)
        convert_element_type_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_2: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_5: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = unsqueeze_5 = None
        iota_3: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_6: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = unsqueeze_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:104 in forward, code: embeddings = self.LayerNorm(embeddings)
        sub_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1, getitem_1);  convert_element_type_1 = getitem_1 = None
        add_2: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_1: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_2: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        add_3: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, arg7_1);  mul_2 = arg7_1 = None
        convert_element_type_2: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_4: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [16384, 768])
        permute_4: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_2: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg16_1, view_4, permute_4);  arg16_1 = view_4 = permute_4 = None
        view_5: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 384]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_6: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_33: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [16384, 768])
        permute: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [32, 512, 384]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_7: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [32, 512, -1, 64]);  view_1 = None

        # No stacktrace found for following nodes
        permute_default_34: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_2: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [16384, 768])
        permute_1: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_1: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg11_1, view_2, permute_1);  arg11_1 = view_2 = permute_1 = None
        view_3: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 512, 384]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_8: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [32, 512, -1, 64]);  view_3 = None

        # No stacktrace found for following nodes
        permute_default_35: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        _scaled_dot_product_cudnn_attention_default_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_33, permute_default_34, permute_default_35, None, False, scale = 0.125);  permute_default_33 = permute_default_34 = permute_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_63: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_11[0];  _scaled_dot_product_cudnn_attention_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_15: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None
        clone_8: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_12: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [16384, 768])
        permute_10: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_3: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg20_1, view_12, permute_10);  arg20_1 = view_12 = permute_10 = None
        view_13: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [32, 512, 384]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_11: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        clone_2: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        unsqueeze_2: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_2, -1);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_2, [0, 0, 4, 4], 0.0);  unsqueeze_2 = None
        iota: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_4: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        add_6: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
        unsqueeze_7: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_6, -1);  add_6 = None
        unsqueeze_8: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, -1);  unsqueeze_7 = None
        full_default_1: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd = unsqueeze_8 = full_default_1 = None
        permute_12: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
        view_15: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_12, [32, 3456, 512]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_13: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        view_16: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_13, [32, 512, 384, 9]);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_3: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_16, memory_format = torch.contiguous_format);  view_16 = None
        view_17: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [98304, 64, 9]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_1: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_17, [98304, 64, 9]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_2: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_2, arg12_1, None, [1], [4], [1], False, [0], 768);  permute_2 = arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_1: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution, arg13_1, None, [1], [0], [1], False, [0], 1);  convolution = arg13_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_4: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_1, arg14_1);  convolution_1 = arg14_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_8: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_4, [0, 2, 1]);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_3: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_8, view_5);  permute_8 = view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_1: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_3, memory_format = torch.contiguous_format);  mul_3 = None
        view_9: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [16384, 384]);  clone_1 = None
        permute_9: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        mm: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_9, permute_9);  view_9 = permute_9 = None
        view_10: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [32, 512, 54]);  mm = None
        add_5: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_10, arg18_1);  view_10 = arg18_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_11: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_5, [-1, 9, 1]);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_14: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        amax: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_14, [1], True)
        sub_2: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_14, amax);  convert_element_type_14 = amax = None
        exp: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_15: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_2: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_15, [98304, 9, 1]);  convert_element_type_15 = None
        bmm: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_1, expand_2);  expand_1 = expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_21: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [-1, 384]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_28: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_21, [32, -1, 6, 64]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_8, view_28], 2);  clone_8 = view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_29: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat, [32, 512, 768]);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_30: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [16384, 768]);  view_29 = None
        permute_16: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_4: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg22_1, view_30, permute_16);  arg22_1 = view_30 = permute_16 = None
        view_31: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 768]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_9: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_31, convert_element_type_2);  view_31 = convert_element_type_2 = None
        convert_element_type_30: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_30, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_4: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_30, getitem_3);  convert_element_type_30 = getitem_3 = None
        add_10: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_4: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
        mul_5: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, arg23_1);  mul_4 = arg23_1 = None
        add_11: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, arg24_1);  mul_5 = arg24_1 = None
        convert_element_type_31: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_32: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [16384, 768])
        permute_17: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_5: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg26_1, view_32, permute_17);  arg26_1 = view_32 = permute_17 = None
        view_33: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 3072]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_35: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_6: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.5)
        mul_7: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.7071067811865476);  convert_element_type_35 = None
        erf: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_12: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, add_12);  mul_6 = add_12 = None
        convert_element_type_36: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_34: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_36, [16384, 3072]);  convert_element_type_36 = None
        permute_18: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_6: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg28_1, view_34, permute_18);  arg28_1 = view_34 = permute_18 = None
        view_35: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [32, 512, 768]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_13: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_35, convert_element_type_31);  view_35 = convert_element_type_31 = None
        convert_element_type_40: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_6: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_12: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_6, 0);  iota_6 = unsqueeze_12 = None
        iota_7: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_13: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_5: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_5);  convert_element_type_40 = getitem_5 = None
        add_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_9: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = rsqrt_2 = None
        mul_10: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, arg29_1);  mul_9 = arg29_1 = None
        add_15: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, arg30_1);  mul_10 = arg30_1 = None
        convert_element_type_41: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_40: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_41, [16384, 768])
        permute_23: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_9: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg39_1, view_40, permute_23);  arg39_1 = view_40 = permute_23 = None
        view_41: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 384]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_42: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_41, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_30: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view_36: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_41, [16384, 768])
        permute_19: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_7: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg32_1, view_36, permute_19);  arg32_1 = view_36 = permute_19 = None
        view_37: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [32, 512, 384]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_43: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [32, 512, -1, 64]);  view_37 = None

        # No stacktrace found for following nodes
        permute_default_31: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_38: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_41, [16384, 768])
        permute_20: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_8: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg34_1, view_38, permute_20);  arg34_1 = view_38 = permute_20 = None
        view_39: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [32, 512, 384]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_44: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_39, [32, 512, -1, 64]);  view_39 = None

        # No stacktrace found for following nodes
        permute_default_32: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        _scaled_dot_product_cudnn_attention_default_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_30, permute_default_31, permute_default_32, None, False, scale = 0.125);  permute_default_30 = permute_default_31 = permute_default_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_62: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_10[0];  _scaled_dot_product_cudnn_attention_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_34: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3]);  getitem_62 = None
        clone_18: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_48: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_41, [16384, 768])
        permute_29: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_10: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg43_1, view_48, permute_29);  arg43_1 = view_48 = permute_29 = None
        view_49: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 384]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_30: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
        clone_12: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        unsqueeze_9: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_12, -1);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_1: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_9, [0, 0, 4, 4], 0.0);  unsqueeze_9 = None
        iota_4: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_10: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
        iota_5: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_11: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
        add_18: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_10, unsqueeze_11);  unsqueeze_10 = unsqueeze_11 = None
        unsqueeze_14: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_18, -1);  add_18 = None
        unsqueeze_15: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        full_default_2: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_1: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_1, [None, None, unsqueeze_15, full_default_2]);  constant_pad_nd_1 = unsqueeze_15 = full_default_2 = None
        permute_31: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
        view_51: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_31, [32, 3456, 512]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_32: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
        view_52: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_32, [32, 512, 384, 9]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_13: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_52, memory_format = torch.contiguous_format);  view_52 = None
        view_53: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [98304, 64, 9]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_7: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_53, [98304, 64, 9]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_21: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_41, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_2: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_21, arg35_1, None, [1], [4], [1], False, [0], 768);  permute_21 = arg35_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_3: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_2, arg36_1, None, [1], [0], [1], False, [0], 1);  convolution_2 = arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_16: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_3, arg37_1);  convolution_3 = arg37_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_27: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_16, [0, 2, 1]);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_11: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_27, view_41);  permute_27 = view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_11: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_11, memory_format = torch.contiguous_format);  mul_11 = None
        view_45: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [16384, 384]);  clone_11 = None
        permute_28: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_1: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_45, permute_28);  view_45 = permute_28 = None
        view_46: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [32, 512, 54]);  mm_1 = None
        add_17: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_46, arg41_1);  view_46 = arg41_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_47: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_17, [-1, 9, 1]);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_53: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None
        amax_2: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_53, [1], True)
        sub_6: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_53, amax_2);  convert_element_type_53 = amax_2 = None
        exp_2: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_3: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True)
        div_3: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_54: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_8: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_54, [98304, 9, 1]);  convert_element_type_54 = None
        bmm_3: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_7, expand_8);  expand_7 = expand_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_57: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [-1, 384]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_64: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_57, [32, -1, 6, 64]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_1: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_18, view_64], 2);  clone_18 = view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_65: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [32, 512, 768]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_66: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_65, [16384, 768]);  view_65 = None
        permute_35: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_11: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg45_1, view_66, permute_35);  arg45_1 = view_66 = permute_35 = None
        view_67: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [32, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_21: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_67, convert_element_type_41);  view_67 = convert_element_type_41 = None
        convert_element_type_69: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.float32);  add_21 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_69, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_8: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_69, getitem_7);  convert_element_type_69 = getitem_7 = None
        add_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = rsqrt_3 = None
        mul_13: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, arg46_1);  mul_12 = arg46_1 = None
        add_23: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, arg47_1);  mul_13 = arg47_1 = None
        convert_element_type_70: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_68: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_70, [16384, 768])
        permute_36: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_12: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg49_1, view_68, permute_36);  arg49_1 = view_68 = permute_36 = None
        view_69: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [32, 512, 3072]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_74: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None
        mul_14: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_74, 0.5)
        mul_15: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_74, 0.7071067811865476);  convert_element_type_74 = None
        erf_1: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_24: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, add_24);  mul_14 = add_24 = None
        convert_element_type_75: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_70: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_75, [16384, 3072]);  convert_element_type_75 = None
        permute_37: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_13: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg51_1, view_70, permute_37);  arg51_1 = view_70 = permute_37 = None
        view_71: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [32, 512, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_25: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_71, convert_element_type_70);  view_71 = convert_element_type_70 = None
        convert_element_type_79: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_79, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_10: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_19: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_10, 0);  iota_10 = unsqueeze_19 = None
        iota_11: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_20: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_9: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_79, getitem_9);  convert_element_type_79 = getitem_9 = None
        add_26: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_17: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = rsqrt_4 = None
        mul_18: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, arg52_1);  mul_17 = arg52_1 = None
        add_27: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, arg53_1);  mul_18 = arg53_1 = None
        convert_element_type_80: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_76: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_80, [16384, 768])
        permute_42: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_16: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg62_1, view_76, permute_42);  arg62_1 = view_76 = permute_42 = None
        view_77: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 384]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_78: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_77, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_27: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view_72: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_80, [16384, 768])
        permute_38: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_14: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg55_1, view_72, permute_38);  arg55_1 = view_72 = permute_38 = None
        view_73: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [32, 512, 384]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_79: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [32, 512, -1, 64]);  view_73 = None

        # No stacktrace found for following nodes
        permute_default_28: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_74: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_80, [16384, 768])
        permute_39: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_15: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg57_1, view_74, permute_39);  arg57_1 = view_74 = permute_39 = None
        view_75: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [32, 512, 384]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_80: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_75, [32, 512, -1, 64]);  view_75 = None

        # No stacktrace found for following nodes
        permute_default_29: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        _scaled_dot_product_cudnn_attention_default_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_27, permute_default_28, permute_default_29, None, False, scale = 0.125);  permute_default_27 = permute_default_28 = permute_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_61: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_9[0];  _scaled_dot_product_cudnn_attention_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_53: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_61, [0, 2, 1, 3]);  getitem_61 = None
        clone_28: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_84: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_80, [16384, 768])
        permute_48: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_17: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg66_1, view_84, permute_48);  arg66_1 = view_84 = permute_48 = None
        view_85: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [32, 512, 384]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_49: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
        clone_22: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        unsqueeze_16: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_22, -1);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_2: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_16, [0, 0, 4, 4], 0.0);  unsqueeze_16 = None
        iota_8: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_17: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_8, 0);  iota_8 = None
        iota_9: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_18: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
        add_30: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_17, unsqueeze_18);  unsqueeze_17 = unsqueeze_18 = None
        unsqueeze_21: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_30, -1);  add_30 = None
        unsqueeze_22: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_21, -1);  unsqueeze_21 = None
        full_default_3: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_2: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_2, [None, None, unsqueeze_22, full_default_3]);  constant_pad_nd_2 = unsqueeze_22 = full_default_3 = None
        permute_50: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_2, [0, 1, 2, 4, 3, 5]);  index_2 = None
        view_87: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_50, [32, 3456, 512]);  permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_51: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        view_88: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_51, [32, 512, 384, 9]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_23: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_88, memory_format = torch.contiguous_format);  view_88 = None
        view_89: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [98304, 64, 9]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_13: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_89, [98304, 64, 9]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_40: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_80, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_4: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_40, arg58_1, None, [1], [4], [1], False, [0], 768);  permute_40 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_5: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_4, arg59_1, None, [1], [0], [1], False, [0], 1);  convolution_4 = arg59_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_28: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_5, arg60_1);  convolution_5 = arg60_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_46: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_28, [0, 2, 1]);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_19: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_46, view_77);  permute_46 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_21: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_19, memory_format = torch.contiguous_format);  mul_19 = None
        view_81: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [16384, 384]);  clone_21 = None
        permute_47: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_2: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_81, permute_47);  view_81 = permute_47 = None
        view_82: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [32, 512, 54]);  mm_2 = None
        add_29: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_82, arg64_1);  view_82 = arg64_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_83: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_29, [-1, 9, 1]);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_92: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_83, torch.float32);  view_83 = None
        amax_4: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_92, [1], True)
        sub_10: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_92, amax_4);  convert_element_type_92 = amax_4 = None
        exp_4: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_5: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [1], True)
        div_6: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_93: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_14: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_93, [98304, 9, 1]);  convert_element_type_93 = None
        bmm_6: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_13, expand_14);  expand_13 = expand_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_93: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [-1, 384]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_100: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_93, [32, -1, 6, 64]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_2: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_28, view_100], 2);  clone_28 = view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_101: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_2, [32, 512, 768]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_102: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_101, [16384, 768]);  view_101 = None
        permute_54: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_18: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg68_1, view_102, permute_54);  arg68_1 = view_102 = permute_54 = None
        view_103: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [32, 512, 768]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_33: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_103, convert_element_type_80);  view_103 = convert_element_type_80 = None
        convert_element_type_108: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_108, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_108, getitem_11);  convert_element_type_108 = getitem_11 = None
        add_34: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_20: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_5);  sub_12 = rsqrt_5 = None
        mul_21: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, arg69_1);  mul_20 = arg69_1 = None
        add_35: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, arg70_1);  mul_21 = arg70_1 = None
        convert_element_type_109: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_109, [16384, 768])
        permute_55: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_19: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg72_1, view_104, permute_55);  arg72_1 = view_104 = permute_55 = None
        view_105: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 3072]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_113: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_22: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.5)
        mul_23: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.7071067811865476);  convert_element_type_113 = None
        erf_2: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_36: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, add_36);  mul_22 = add_36 = None
        convert_element_type_114: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_114, [16384, 3072]);  convert_element_type_114 = None
        permute_56: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_20: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg74_1, view_106, permute_56);  arg74_1 = view_106 = permute_56 = None
        view_107: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [32, 512, 768]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_37: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_107, convert_element_type_109);  view_107 = convert_element_type_109 = None
        convert_element_type_118: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_118, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_14: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_26: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_14, 0);  iota_14 = unsqueeze_26 = None
        iota_15: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_27: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_13: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_118, getitem_13);  convert_element_type_118 = getitem_13 = None
        add_38: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_25: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_6);  sub_13 = rsqrt_6 = None
        mul_26: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, arg75_1);  mul_25 = arg75_1 = None
        add_39: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_26, arg76_1);  mul_26 = arg76_1 = None
        convert_element_type_119: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_112: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_119, [16384, 768])
        permute_61: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_23: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg85_1, view_112, permute_61);  arg85_1 = view_112 = permute_61 = None
        view_113: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 384]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_114: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_24: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view_108: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_119, [16384, 768])
        permute_57: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_21: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg78_1, view_108, permute_57);  arg78_1 = view_108 = permute_57 = None
        view_109: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [32, 512, 384]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_115: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_109, [32, 512, -1, 64]);  view_109 = None

        # No stacktrace found for following nodes
        permute_default_25: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_110: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_119, [16384, 768])
        permute_58: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_22: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg80_1, view_110, permute_58);  arg80_1 = view_110 = permute_58 = None
        view_111: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 384]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_116: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [32, 512, -1, 64]);  view_111 = None

        # No stacktrace found for following nodes
        permute_default_26: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        _scaled_dot_product_cudnn_attention_default_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_24, permute_default_25, permute_default_26, None, False, scale = 0.125);  permute_default_24 = permute_default_25 = permute_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_60: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_8[0];  _scaled_dot_product_cudnn_attention_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_72: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        clone_38: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_120: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_119, [16384, 768])
        permute_67: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_24: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg89_1, view_120, permute_67);  arg89_1 = view_120 = permute_67 = None
        view_121: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [32, 512, 384]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_68: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        clone_32: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        unsqueeze_23: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_32, -1);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_3: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_23, [0, 0, 4, 4], 0.0);  unsqueeze_23 = None
        iota_12: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_24: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_12, 0);  iota_12 = None
        iota_13: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_25: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
        add_42: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_24, unsqueeze_25);  unsqueeze_24 = unsqueeze_25 = None
        unsqueeze_28: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_42, -1);  add_42 = None
        unsqueeze_29: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        full_default_4: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_3: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_3, [None, None, unsqueeze_29, full_default_4]);  constant_pad_nd_3 = unsqueeze_29 = full_default_4 = None
        permute_69: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_3, [0, 1, 2, 4, 3, 5]);  index_3 = None
        view_123: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_69, [32, 3456, 512]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_70: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        view_124: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_70, [32, 512, 384, 9]);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_33: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_124, memory_format = torch.contiguous_format);  view_124 = None
        view_125: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [98304, 64, 9]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_19: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_125, [98304, 64, 9]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_59: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_119, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_6: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_59, arg81_1, None, [1], [4], [1], False, [0], 768);  permute_59 = arg81_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_7: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_6, arg82_1, None, [1], [0], [1], False, [0], 1);  convolution_6 = arg82_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_40: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_7, arg83_1);  convolution_7 = arg83_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_65: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_27: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_65, view_113);  permute_65 = view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_31: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_27, memory_format = torch.contiguous_format);  mul_27 = None
        view_117: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [16384, 384]);  clone_31 = None
        permute_66: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_3: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_117, permute_66);  view_117 = permute_66 = None
        view_118: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [32, 512, 54]);  mm_3 = None
        add_41: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_118, arg87_1);  view_118 = arg87_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_119: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_41, [-1, 9, 1]);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_131: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        amax_6: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_131, [1], True)
        sub_14: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_131, amax_6);  convert_element_type_131 = amax_6 = None
        exp_6: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_7: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True)
        div_9: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_132: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_20: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_132, [98304, 9, 1]);  convert_element_type_132 = None
        bmm_9: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_19, expand_20);  expand_19 = expand_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_129: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [-1, 384]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_136: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_129, [32, -1, 6, 64]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_3: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_38, view_136], 2);  clone_38 = view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_137: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [32, 512, 768]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_138: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [16384, 768]);  view_137 = None
        permute_73: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_25: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg91_1, view_138, permute_73);  arg91_1 = view_138 = permute_73 = None
        view_139: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [32, 512, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_45: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_139, convert_element_type_119);  view_139 = convert_element_type_119 = None
        convert_element_type_147: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.float32);  add_45 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_147, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_16: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_147, getitem_15);  convert_element_type_147 = getitem_15 = None
        add_46: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_28: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_7);  sub_16 = rsqrt_7 = None
        mul_29: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, arg92_1);  mul_28 = arg92_1 = None
        add_47: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, arg93_1);  mul_29 = arg93_1 = None
        convert_element_type_148: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_140: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_148, [16384, 768])
        permute_74: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_26: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg95_1, view_140, permute_74);  arg95_1 = view_140 = permute_74 = None
        view_141: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [32, 512, 3072]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_152: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_30: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_152, 0.5)
        mul_31: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_152, 0.7071067811865476);  convert_element_type_152 = None
        erf_3: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_48: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, add_48);  mul_30 = add_48 = None
        convert_element_type_153: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_142: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_153, [16384, 3072]);  convert_element_type_153 = None
        permute_75: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_27: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg97_1, view_142, permute_75);  arg97_1 = view_142 = permute_75 = None
        view_143: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [32, 512, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_49: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_143, convert_element_type_148);  view_143 = convert_element_type_148 = None
        convert_element_type_157: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.float32);  add_49 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_157, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_18: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_33: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_18, 0);  iota_18 = unsqueeze_33 = None
        iota_19: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_34: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = unsqueeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_17: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_157, getitem_17);  convert_element_type_157 = getitem_17 = None
        add_50: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_33: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_8);  sub_17 = rsqrt_8 = None
        mul_34: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, arg98_1);  mul_33 = arg98_1 = None
        add_51: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, arg99_1);  mul_34 = arg99_1 = None
        convert_element_type_158: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_148: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_158, [16384, 768])
        permute_80: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_30: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg108_1, view_148, permute_80);  arg108_1 = view_148 = permute_80 = None
        view_149: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [32, 512, 384]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_150: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_149, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_21: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view_144: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_158, [16384, 768])
        permute_76: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_28: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg101_1, view_144, permute_76);  arg101_1 = view_144 = permute_76 = None
        view_145: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 512, 384]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_151: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_145, [32, 512, -1, 64]);  view_145 = None

        # No stacktrace found for following nodes
        permute_default_22: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_146: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_158, [16384, 768])
        permute_77: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_29: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg103_1, view_146, permute_77);  arg103_1 = view_146 = permute_77 = None
        view_147: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [32, 512, 384]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_152: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [32, 512, -1, 64]);  view_147 = None

        # No stacktrace found for following nodes
        permute_default_23: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        _scaled_dot_product_cudnn_attention_default_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_21, permute_default_22, permute_default_23, None, False, scale = 0.125);  permute_default_21 = permute_default_22 = permute_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_59: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_7[0];  _scaled_dot_product_cudnn_attention_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_91: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3]);  getitem_59 = None
        clone_48: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_156: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_158, [16384, 768])
        permute_86: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_31: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg112_1, view_156, permute_86);  arg112_1 = view_156 = permute_86 = None
        view_157: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [32, 512, 384]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_87: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
        clone_42: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
        unsqueeze_30: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_42, -1);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_4: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_30, [0, 0, 4, 4], 0.0);  unsqueeze_30 = None
        iota_16: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_31: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_16, 0);  iota_16 = None
        iota_17: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_32: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
        add_54: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_31, unsqueeze_32);  unsqueeze_31 = unsqueeze_32 = None
        unsqueeze_35: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_54, -1);  add_54 = None
        unsqueeze_36: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_35, -1);  unsqueeze_35 = None
        full_default_5: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_4: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_4, [None, None, unsqueeze_36, full_default_5]);  constant_pad_nd_4 = unsqueeze_36 = full_default_5 = None
        permute_88: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_4, [0, 1, 2, 4, 3, 5]);  index_4 = None
        view_159: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_88, [32, 3456, 512]);  permute_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_89: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None
        view_160: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_89, [32, 512, 384, 9]);  permute_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_43: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_160, memory_format = torch.contiguous_format);  view_160 = None
        view_161: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [98304, 64, 9]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_25: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_161, [98304, 64, 9]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_78: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_158, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_8: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_78, arg104_1, None, [1], [4], [1], False, [0], 768);  permute_78 = arg104_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_9: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_8, arg105_1, None, [1], [0], [1], False, [0], 1);  convolution_8 = arg105_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_52: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_9, arg106_1);  convolution_9 = arg106_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_84: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_52, [0, 2, 1]);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_35: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_84, view_149);  permute_84 = view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_41: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_35, memory_format = torch.contiguous_format);  mul_35 = None
        view_153: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [16384, 384]);  clone_41 = None
        permute_85: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_4: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_153, permute_85);  view_153 = permute_85 = None
        view_154: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [32, 512, 54]);  mm_4 = None
        add_53: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_154, arg110_1);  view_154 = arg110_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_155: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_53, [-1, 9, 1]);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_170: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_155, torch.float32);  view_155 = None
        amax_8: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_170, [1], True)
        sub_18: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_170, amax_8);  convert_element_type_170 = amax_8 = None
        exp_8: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_9: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [1], True)
        div_12: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_171: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_26: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_171, [98304, 9, 1]);  convert_element_type_171 = None
        bmm_12: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_25, expand_26);  expand_25 = expand_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_165: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [-1, 384]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_172: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_165, [32, -1, 6, 64]);  view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_4: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_48, view_172], 2);  clone_48 = view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_173: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [32, 512, 768]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_173, [16384, 768]);  view_173 = None
        permute_92: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_32: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg114_1, view_174, permute_92);  arg114_1 = view_174 = permute_92 = None
        view_175: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [32, 512, 768]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_175, convert_element_type_158);  view_175 = convert_element_type_158 = None
        convert_element_type_186: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_186, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_20: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_186, getitem_19);  convert_element_type_186 = getitem_19 = None
        add_58: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_36: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_9);  sub_20 = rsqrt_9 = None
        mul_37: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, arg115_1);  mul_36 = arg115_1 = None
        add_59: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, arg116_1);  mul_37 = arg116_1 = None
        convert_element_type_187: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_176: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_187, [16384, 768])
        permute_93: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_33: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg118_1, view_176, permute_93);  arg118_1 = view_176 = permute_93 = None
        view_177: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [32, 512, 3072]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_191: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        mul_38: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_191, 0.5)
        mul_39: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_191, 0.7071067811865476);  convert_element_type_191 = None
        erf_4: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_60: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, add_60);  mul_38 = add_60 = None
        convert_element_type_192: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_178: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_192, [16384, 3072]);  convert_element_type_192 = None
        permute_94: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_34: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg120_1, view_178, permute_94);  arg120_1 = view_178 = permute_94 = None
        view_179: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 512, 768]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_61: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_179, convert_element_type_187);  view_179 = convert_element_type_187 = None
        convert_element_type_196: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.float32);  add_61 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_196, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_22: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_40: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_22, 0);  iota_22 = unsqueeze_40 = None
        iota_23: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_41: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_21: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_196, getitem_21);  convert_element_type_196 = getitem_21 = None
        add_62: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_41: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_10);  sub_21 = rsqrt_10 = None
        mul_42: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, arg121_1);  mul_41 = arg121_1 = None
        add_63: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_42, arg122_1);  mul_42 = arg122_1 = None
        convert_element_type_197: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_184: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_197, [16384, 768])
        permute_99: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_37: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg131_1, view_184, permute_99);  arg131_1 = view_184 = permute_99 = None
        view_185: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [32, 512, 384]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_186: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_185, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_18: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view_180: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_197, [16384, 768])
        permute_95: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_35: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg124_1, view_180, permute_95);  arg124_1 = view_180 = permute_95 = None
        view_181: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [32, 512, 384]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_187: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_181, [32, 512, -1, 64]);  view_181 = None

        # No stacktrace found for following nodes
        permute_default_19: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_182: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_197, [16384, 768])
        permute_96: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_36: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg126_1, view_182, permute_96);  arg126_1 = view_182 = permute_96 = None
        view_183: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [32, 512, 384]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_188: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_183, [32, 512, -1, 64]);  view_183 = None

        # No stacktrace found for following nodes
        permute_default_20: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        _scaled_dot_product_cudnn_attention_default_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_18, permute_default_19, permute_default_20, None, False, scale = 0.125);  permute_default_18 = permute_default_19 = permute_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_58: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_6[0];  _scaled_dot_product_cudnn_attention_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_110: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        clone_58: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_192: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_197, [16384, 768])
        permute_105: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_38: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg135_1, view_192, permute_105);  arg135_1 = view_192 = permute_105 = None
        view_193: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [32, 512, 384]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_106: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
        clone_52: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        unsqueeze_37: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_52, -1);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_5: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_37, [0, 0, 4, 4], 0.0);  unsqueeze_37 = None
        iota_20: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_38: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_20, 0);  iota_20 = None
        iota_21: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_39: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
        add_66: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_38, unsqueeze_39);  unsqueeze_38 = unsqueeze_39 = None
        unsqueeze_42: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_66, -1);  add_66 = None
        unsqueeze_43: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        full_default_6: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_5: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_5, [None, None, unsqueeze_43, full_default_6]);  constant_pad_nd_5 = unsqueeze_43 = full_default_6 = None
        permute_107: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_5, [0, 1, 2, 4, 3, 5]);  index_5 = None
        view_195: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_107, [32, 3456, 512]);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_108: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
        view_196: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_108, [32, 512, 384, 9]);  permute_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_53: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_196, memory_format = torch.contiguous_format);  view_196 = None
        view_197: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [98304, 64, 9]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_31: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_197, [98304, 64, 9]);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_97: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_197, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_10: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_97, arg127_1, None, [1], [4], [1], False, [0], 768);  permute_97 = arg127_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_11: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_10, arg128_1, None, [1], [0], [1], False, [0], 1);  convolution_10 = arg128_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_64: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_11, arg129_1);  convolution_11 = arg129_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_103: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_64, [0, 2, 1]);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_43: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_103, view_185);  permute_103 = view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_51: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_43, memory_format = torch.contiguous_format);  mul_43 = None
        view_189: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [16384, 384]);  clone_51 = None
        permute_104: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        mm_5: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_189, permute_104);  view_189 = permute_104 = None
        view_190: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [32, 512, 54]);  mm_5 = None
        add_65: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_190, arg133_1);  view_190 = arg133_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_191: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_65, [-1, 9, 1]);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_209: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_191, torch.float32);  view_191 = None
        amax_10: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_209, [1], True)
        sub_22: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_209, amax_10);  convert_element_type_209 = amax_10 = None
        exp_10: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_11: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [1], True)
        div_15: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_210: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_32: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_210, [98304, 9, 1]);  convert_element_type_210 = None
        bmm_15: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_31, expand_32);  expand_31 = expand_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_201: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [-1, 384]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_208: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_201, [32, -1, 6, 64]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_5: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_58, view_208], 2);  clone_58 = view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_209: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [32, 512, 768]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_210: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_209, [16384, 768]);  view_209 = None
        permute_111: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_39: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg137_1, view_210, permute_111);  arg137_1 = view_210 = permute_111 = None
        view_211: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [32, 512, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_69: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_211, convert_element_type_197);  view_211 = convert_element_type_197 = None
        convert_element_type_225: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.float32);  add_69 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_225, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_24: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_225, getitem_23);  convert_element_type_225 = getitem_23 = None
        add_70: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_44: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_11);  sub_24 = rsqrt_11 = None
        mul_45: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, arg138_1);  mul_44 = arg138_1 = None
        add_71: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_45, arg139_1);  mul_45 = arg139_1 = None
        convert_element_type_226: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_212: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_226, [16384, 768])
        permute_112: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_40: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg141_1, view_212, permute_112);  arg141_1 = view_212 = permute_112 = None
        view_213: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_230: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_213, torch.float32);  view_213 = None
        mul_46: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_230, 0.5)
        mul_47: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_230, 0.7071067811865476);  convert_element_type_230 = None
        erf_5: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_72: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, add_72);  mul_46 = add_72 = None
        convert_element_type_231: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_231, [16384, 3072]);  convert_element_type_231 = None
        permute_113: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_41: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg143_1, view_214, permute_113);  arg143_1 = view_214 = permute_113 = None
        view_215: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [32, 512, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_73: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_215, convert_element_type_226);  view_215 = convert_element_type_226 = None
        convert_element_type_235: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.float32);  add_73 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_235, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_26: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_47: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_26, 0);  iota_26 = unsqueeze_47 = None
        iota_27: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_48: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = unsqueeze_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_25: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_235, getitem_25);  convert_element_type_235 = getitem_25 = None
        add_74: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_49: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_12);  sub_25 = rsqrt_12 = None
        mul_50: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, arg144_1);  mul_49 = arg144_1 = None
        add_75: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, arg145_1);  mul_50 = arg145_1 = None
        convert_element_type_236: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_220: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_236, [16384, 768])
        permute_118: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_44: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg154_1, view_220, permute_118);  arg154_1 = view_220 = permute_118 = None
        view_221: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [32, 512, 384]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_222: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_15: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view_216: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_236, [16384, 768])
        permute_114: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_42: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg147_1, view_216, permute_114);  arg147_1 = view_216 = permute_114 = None
        view_217: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [32, 512, 384]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_223: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_217, [32, 512, -1, 64]);  view_217 = None

        # No stacktrace found for following nodes
        permute_default_16: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_218: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_236, [16384, 768])
        permute_115: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_43: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg149_1, view_218, permute_115);  arg149_1 = view_218 = permute_115 = None
        view_219: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [32, 512, 384]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_224: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_219, [32, 512, -1, 64]);  view_219 = None

        # No stacktrace found for following nodes
        permute_default_17: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        _scaled_dot_product_cudnn_attention_default_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_15, permute_default_16, permute_default_17, None, False, scale = 0.125);  permute_default_15 = permute_default_16 = permute_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_57: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_5[0];  _scaled_dot_product_cudnn_attention_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_129: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
        clone_68: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_228: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_236, [16384, 768])
        permute_124: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_45: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg158_1, view_228, permute_124);  arg158_1 = view_228 = permute_124 = None
        view_229: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [32, 512, 384]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_125: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        clone_62: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        unsqueeze_44: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_62, -1);  clone_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_6: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_44, [0, 0, 4, 4], 0.0);  unsqueeze_44 = None
        iota_24: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_45: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_24, 0);  iota_24 = None
        iota_25: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_46: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
        add_78: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_45, unsqueeze_46);  unsqueeze_45 = unsqueeze_46 = None
        unsqueeze_49: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_78, -1);  add_78 = None
        unsqueeze_50: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_49, -1);  unsqueeze_49 = None
        full_default_7: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_6: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_6, [None, None, unsqueeze_50, full_default_7]);  constant_pad_nd_6 = unsqueeze_50 = full_default_7 = None
        permute_126: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_6, [0, 1, 2, 4, 3, 5]);  index_6 = None
        view_231: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_126, [32, 3456, 512]);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_127: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        view_232: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_127, [32, 512, 384, 9]);  permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_63: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_232, memory_format = torch.contiguous_format);  view_232 = None
        view_233: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [98304, 64, 9]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_37: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_233, [98304, 64, 9]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_116: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_236, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_12: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_116, arg150_1, None, [1], [4], [1], False, [0], 768);  permute_116 = arg150_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_13: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_12, arg151_1, None, [1], [0], [1], False, [0], 1);  convolution_12 = arg151_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_76: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_13, arg152_1);  convolution_13 = arg152_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_122: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_76, [0, 2, 1]);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_51: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_122, view_221);  permute_122 = view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_61: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_51, memory_format = torch.contiguous_format);  mul_51 = None
        view_225: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [16384, 384]);  clone_61 = None
        permute_123: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        mm_6: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_225, permute_123);  view_225 = permute_123 = None
        view_226: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [32, 512, 54]);  mm_6 = None
        add_77: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_226, arg156_1);  view_226 = arg156_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_227: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_77, [-1, 9, 1]);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_248: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_227, torch.float32);  view_227 = None
        amax_12: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_248, [1], True)
        sub_26: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_248, amax_12);  convert_element_type_248 = amax_12 = None
        exp_12: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_13: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True)
        div_18: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_249: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_38: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_249, [98304, 9, 1]);  convert_element_type_249 = None
        bmm_18: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_37, expand_38);  expand_37 = expand_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_237: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [-1, 384]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_244: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_237, [32, -1, 6, 64]);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_6: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_68, view_244], 2);  clone_68 = view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_245: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [32, 512, 768]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_246: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_245, [16384, 768]);  view_245 = None
        permute_130: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_46: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg160_1, view_246, permute_130);  arg160_1 = view_246 = permute_130 = None
        view_247: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [32, 512, 768]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_81: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_247, convert_element_type_236);  view_247 = convert_element_type_236 = None
        convert_element_type_264: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_264, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_28: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_264, getitem_27);  convert_element_type_264 = getitem_27 = None
        add_82: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_52: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_13);  sub_28 = rsqrt_13 = None
        mul_53: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, arg161_1);  mul_52 = arg161_1 = None
        add_83: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, arg162_1);  mul_53 = arg162_1 = None
        convert_element_type_265: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_248: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_265, [16384, 768])
        permute_131: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_47: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg164_1, view_248, permute_131);  arg164_1 = view_248 = permute_131 = None
        view_249: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [32, 512, 3072]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_269: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        mul_54: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_269, 0.5)
        mul_55: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_269, 0.7071067811865476);  convert_element_type_269 = None
        erf_6: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_84: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, add_84);  mul_54 = add_84 = None
        convert_element_type_270: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_250: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_270, [16384, 3072]);  convert_element_type_270 = None
        permute_132: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_48: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg166_1, view_250, permute_132);  arg166_1 = view_250 = permute_132 = None
        view_251: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [32, 512, 768]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_85: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_251, convert_element_type_265);  view_251 = convert_element_type_265 = None
        convert_element_type_274: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.float32);  add_85 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_274, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_30: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_54: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_30, 0);  iota_30 = unsqueeze_54 = None
        iota_31: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_55: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_29: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_274, getitem_29);  convert_element_type_274 = getitem_29 = None
        add_86: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_57: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_14);  sub_29 = rsqrt_14 = None
        mul_58: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg167_1);  mul_57 = arg167_1 = None
        add_87: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg168_1);  mul_58 = arg168_1 = None
        convert_element_type_275: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_256: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_275, [16384, 768])
        permute_137: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_51: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg177_1, view_256, permute_137);  arg177_1 = view_256 = permute_137 = None
        view_257: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [32, 512, 384]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_258: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_12: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view_252: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_275, [16384, 768])
        permute_133: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_49: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg170_1, view_252, permute_133);  arg170_1 = view_252 = permute_133 = None
        view_253: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [32, 512, 384]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_259: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_253, [32, 512, -1, 64]);  view_253 = None

        # No stacktrace found for following nodes
        permute_default_13: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_254: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_275, [16384, 768])
        permute_134: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_50: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg172_1, view_254, permute_134);  arg172_1 = view_254 = permute_134 = None
        view_255: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [32, 512, 384]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_260: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_255, [32, 512, -1, 64]);  view_255 = None

        # No stacktrace found for following nodes
        permute_default_14: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None
        _scaled_dot_product_cudnn_attention_default_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_12, permute_default_13, permute_default_14, None, False, scale = 0.125);  permute_default_12 = permute_default_13 = permute_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_56: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_4[0];  _scaled_dot_product_cudnn_attention_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_148: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
        clone_78: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_264: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_275, [16384, 768])
        permute_143: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_52: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg181_1, view_264, permute_143);  arg181_1 = view_264 = permute_143 = None
        view_265: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [32, 512, 384]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_144: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
        clone_72: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        unsqueeze_51: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_72, -1);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_7: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_51, [0, 0, 4, 4], 0.0);  unsqueeze_51 = None
        iota_28: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_52: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_28, 0);  iota_28 = None
        iota_29: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_53: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
        add_90: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_52, unsqueeze_53);  unsqueeze_52 = unsqueeze_53 = None
        unsqueeze_56: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_90, -1);  add_90 = None
        unsqueeze_57: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        full_default_8: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_7: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_7, [None, None, unsqueeze_57, full_default_8]);  constant_pad_nd_7 = unsqueeze_57 = full_default_8 = None
        permute_145: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_7, [0, 1, 2, 4, 3, 5]);  index_7 = None
        view_267: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_145, [32, 3456, 512]);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_146: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
        view_268: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_146, [32, 512, 384, 9]);  permute_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_73: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_268, memory_format = torch.contiguous_format);  view_268 = None
        view_269: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [98304, 64, 9]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_43: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_269, [98304, 64, 9]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_135: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_275, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_14: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_135, arg173_1, None, [1], [4], [1], False, [0], 768);  permute_135 = arg173_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_15: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_14, arg174_1, None, [1], [0], [1], False, [0], 1);  convolution_14 = arg174_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_88: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_15, arg175_1);  convolution_15 = arg175_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_141: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_88, [0, 2, 1]);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_59: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_141, view_257);  permute_141 = view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_71: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_59, memory_format = torch.contiguous_format);  mul_59 = None
        view_261: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [16384, 384]);  clone_71 = None
        permute_142: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        mm_7: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_261, permute_142);  view_261 = permute_142 = None
        view_262: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [32, 512, 54]);  mm_7 = None
        add_89: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_262, arg179_1);  view_262 = arg179_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_263: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_89, [-1, 9, 1]);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_287: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        amax_14: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_287, [1], True)
        sub_30: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_287, amax_14);  convert_element_type_287 = amax_14 = None
        exp_14: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_15: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [1], True)
        div_21: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_288: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_44: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_288, [98304, 9, 1]);  convert_element_type_288 = None
        bmm_21: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_43, expand_44);  expand_43 = expand_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_273: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [-1, 384]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_280: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_273, [32, -1, 6, 64]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_7: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_78, view_280], 2);  clone_78 = view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_281: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [32, 512, 768]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_282: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_281, [16384, 768]);  view_281 = None
        permute_149: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_53: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg183_1, view_282, permute_149);  arg183_1 = view_282 = permute_149 = None
        view_283: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [32, 512, 768]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_93: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_283, convert_element_type_275);  view_283 = convert_element_type_275 = None
        convert_element_type_303: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.float32);  add_93 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_303, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_32: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_303, getitem_31);  convert_element_type_303 = getitem_31 = None
        add_94: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_60: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_15);  sub_32 = rsqrt_15 = None
        mul_61: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, arg184_1);  mul_60 = arg184_1 = None
        add_95: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, arg185_1);  mul_61 = arg185_1 = None
        convert_element_type_304: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16);  add_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_284: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_304, [16384, 768])
        permute_150: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_54: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg187_1, view_284, permute_150);  arg187_1 = view_284 = permute_150 = None
        view_285: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [32, 512, 3072]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_308: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_285, torch.float32);  view_285 = None
        mul_62: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_308, 0.5)
        mul_63: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_308, 0.7071067811865476);  convert_element_type_308 = None
        erf_7: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_96: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, add_96);  mul_62 = add_96 = None
        convert_element_type_309: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_286: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_309, [16384, 3072]);  convert_element_type_309 = None
        permute_151: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_55: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg189_1, view_286, permute_151);  arg189_1 = view_286 = permute_151 = None
        view_287: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [32, 512, 768]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_97: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_287, convert_element_type_304);  view_287 = convert_element_type_304 = None
        convert_element_type_313: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.float32);  add_97 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_313, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_34: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_61: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_34, 0);  iota_34 = unsqueeze_61 = None
        iota_35: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_62: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = unsqueeze_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_33: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_313, getitem_33);  convert_element_type_313 = getitem_33 = None
        add_98: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_65: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_16);  sub_33 = rsqrt_16 = None
        mul_66: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, arg190_1);  mul_65 = arg190_1 = None
        add_99: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_66, arg191_1);  mul_66 = arg191_1 = None
        convert_element_type_314: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_292: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_314, [16384, 768])
        permute_156: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_58: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg200_1, view_292, permute_156);  arg200_1 = view_292 = permute_156 = None
        view_293: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 384]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_294: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_293, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_9: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view_288: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_314, [16384, 768])
        permute_152: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_56: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg193_1, view_288, permute_152);  arg193_1 = view_288 = permute_152 = None
        view_289: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [32, 512, 384]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_295: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_289, [32, 512, -1, 64]);  view_289 = None

        # No stacktrace found for following nodes
        permute_default_10: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_290: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_314, [16384, 768])
        permute_153: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_57: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg195_1, view_290, permute_153);  arg195_1 = view_290 = permute_153 = None
        view_291: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [32, 512, 384]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_296: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_291, [32, 512, -1, 64]);  view_291 = None

        # No stacktrace found for following nodes
        permute_default_11: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        _scaled_dot_product_cudnn_attention_default_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_9, permute_default_10, permute_default_11, None, False, scale = 0.125);  permute_default_9 = permute_default_10 = permute_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_55: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_3[0];  _scaled_dot_product_cudnn_attention_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_167: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
        clone_88: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_300: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_314, [16384, 768])
        permute_162: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_59: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg204_1, view_300, permute_162);  arg204_1 = view_300 = permute_162 = None
        view_301: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [32, 512, 384]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_163: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_301, [0, 2, 1]);  view_301 = None
        clone_82: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
        unsqueeze_58: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_82, -1);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_8: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_58, [0, 0, 4, 4], 0.0);  unsqueeze_58 = None
        iota_32: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_59: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_32, 0);  iota_32 = None
        iota_33: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_60: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
        add_102: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_59, unsqueeze_60);  unsqueeze_59 = unsqueeze_60 = None
        unsqueeze_63: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_102, -1);  add_102 = None
        unsqueeze_64: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_63, -1);  unsqueeze_63 = None
        full_default_9: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_8: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_8, [None, None, unsqueeze_64, full_default_9]);  constant_pad_nd_8 = unsqueeze_64 = full_default_9 = None
        permute_164: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_8, [0, 1, 2, 4, 3, 5]);  index_8 = None
        view_303: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_164, [32, 3456, 512]);  permute_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_165: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
        view_304: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_165, [32, 512, 384, 9]);  permute_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_83: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_304, memory_format = torch.contiguous_format);  view_304 = None
        view_305: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [98304, 64, 9]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_49: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_305, [98304, 64, 9]);  view_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_154: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_314, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_16: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_154, arg196_1, None, [1], [4], [1], False, [0], 768);  permute_154 = arg196_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_17: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_16, arg197_1, None, [1], [0], [1], False, [0], 1);  convolution_16 = arg197_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_100: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_17, arg198_1);  convolution_17 = arg198_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_160: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_100, [0, 2, 1]);  add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_67: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_160, view_293);  permute_160 = view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_81: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_67, memory_format = torch.contiguous_format);  mul_67 = None
        view_297: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [16384, 384]);  clone_81 = None
        permute_161: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        mm_8: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_297, permute_161);  view_297 = permute_161 = None
        view_298: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [32, 512, 54]);  mm_8 = None
        add_101: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_298, arg202_1);  view_298 = arg202_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_299: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_101, [-1, 9, 1]);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_326: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        amax_16: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_326, [1], True)
        sub_34: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_326, amax_16);  convert_element_type_326 = amax_16 = None
        exp_16: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_17: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [1], True)
        div_24: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_327: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_50: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_327, [98304, 9, 1]);  convert_element_type_327 = None
        bmm_24: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_49, expand_50);  expand_49 = expand_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_309: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [-1, 384]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_316: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_309, [32, -1, 6, 64]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_8: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_88, view_316], 2);  clone_88 = view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_317: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_8, [32, 512, 768]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_318: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_317, [16384, 768]);  view_317 = None
        permute_168: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_60: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg206_1, view_318, permute_168);  arg206_1 = view_318 = permute_168 = None
        view_319: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [32, 512, 768]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_105: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_319, convert_element_type_314);  view_319 = convert_element_type_314 = None
        convert_element_type_342: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_342, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_36: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_342, getitem_35);  convert_element_type_342 = getitem_35 = None
        add_106: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_68: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_17);  sub_36 = rsqrt_17 = None
        mul_69: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, arg207_1);  mul_68 = arg207_1 = None
        add_107: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, arg208_1);  mul_69 = arg208_1 = None
        convert_element_type_343: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16);  add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_320: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_343, [16384, 768])
        permute_169: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_61: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg210_1, view_320, permute_169);  arg210_1 = view_320 = permute_169 = None
        view_321: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [32, 512, 3072]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_347: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_321, torch.float32);  view_321 = None
        mul_70: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.5)
        mul_71: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.7071067811865476);  convert_element_type_347 = None
        erf_8: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_108: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_72: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, add_108);  mul_70 = add_108 = None
        convert_element_type_348: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_322: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_348, [16384, 3072]);  convert_element_type_348 = None
        permute_170: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_62: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg212_1, view_322, permute_170);  arg212_1 = view_322 = permute_170 = None
        view_323: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [32, 512, 768]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_109: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_323, convert_element_type_343);  view_323 = convert_element_type_343 = None
        convert_element_type_352: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.float32);  add_109 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_352, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_38: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_68: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_38, 0);  iota_38 = unsqueeze_68 = None
        iota_39: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_69: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = unsqueeze_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_37: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_352, getitem_37);  convert_element_type_352 = getitem_37 = None
        add_110: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        mul_73: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_18);  sub_37 = rsqrt_18 = None
        mul_74: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, arg213_1);  mul_73 = arg213_1 = None
        add_111: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, arg214_1);  mul_74 = arg214_1 = None
        convert_element_type_353: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_328: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_353, [16384, 768])
        permute_175: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        addmm_65: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg223_1, view_328, permute_175);  arg223_1 = view_328 = permute_175 = None
        view_329: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [32, 512, 384]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_330: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_6: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view_324: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_353, [16384, 768])
        permute_171: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_63: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg216_1, view_324, permute_171);  arg216_1 = view_324 = permute_171 = None
        view_325: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [32, 512, 384]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_331: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_325, [32, 512, -1, 64]);  view_325 = None

        # No stacktrace found for following nodes
        permute_default_7: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_326: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_353, [16384, 768])
        permute_172: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_64: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg218_1, view_326, permute_172);  arg218_1 = view_326 = permute_172 = None
        view_327: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [32, 512, 384]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_332: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_327, [32, 512, -1, 64]);  view_327 = None

        # No stacktrace found for following nodes
        permute_default_8: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        _scaled_dot_product_cudnn_attention_default_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_6, permute_default_7, permute_default_8, None, False, scale = 0.125);  permute_default_6 = permute_default_7 = permute_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_54: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_2[0];  _scaled_dot_product_cudnn_attention_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_186: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None
        clone_98: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_336: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_353, [16384, 768])
        permute_181: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_66: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg227_1, view_336, permute_181);  arg227_1 = view_336 = permute_181 = None
        view_337: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [32, 512, 384]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_182: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
        clone_92: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        unsqueeze_65: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_92, -1);  clone_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_9: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_65, [0, 0, 4, 4], 0.0);  unsqueeze_65 = None
        iota_36: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_66: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_36, 0);  iota_36 = None
        iota_37: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_67: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
        add_114: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_66, unsqueeze_67);  unsqueeze_66 = unsqueeze_67 = None
        unsqueeze_70: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_114, -1);  add_114 = None
        unsqueeze_71: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        full_default_10: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_9: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_9, [None, None, unsqueeze_71, full_default_10]);  constant_pad_nd_9 = unsqueeze_71 = full_default_10 = None
        permute_183: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_9, [0, 1, 2, 4, 3, 5]);  index_9 = None
        view_339: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_183, [32, 3456, 512]);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_184: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        view_340: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_184, [32, 512, 384, 9]);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_93: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_340, memory_format = torch.contiguous_format);  view_340 = None
        view_341: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [98304, 64, 9]);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_55: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_341, [98304, 64, 9]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_173: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_353, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_18: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_173, arg219_1, None, [1], [4], [1], False, [0], 768);  permute_173 = arg219_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_19: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_18, arg220_1, None, [1], [0], [1], False, [0], 1);  convolution_18 = arg220_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_112: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_19, arg221_1);  convolution_19 = arg221_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_179: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_112, [0, 2, 1]);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_75: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_179, view_329);  permute_179 = view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_91: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_75, memory_format = torch.contiguous_format);  mul_75 = None
        view_333: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [16384, 384]);  clone_91 = None
        permute_180: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        mm_9: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_333, permute_180);  view_333 = permute_180 = None
        view_334: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [32, 512, 54]);  mm_9 = None
        add_113: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_334, arg225_1);  view_334 = arg225_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_335: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_113, [-1, 9, 1]);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_365: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_335, torch.float32);  view_335 = None
        amax_18: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_365, [1], True)
        sub_38: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_365, amax_18);  convert_element_type_365 = amax_18 = None
        exp_18: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        sum_19: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True)
        div_27: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_366: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_56: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_366, [98304, 9, 1]);  convert_element_type_366 = None
        bmm_27: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_55, expand_56);  expand_55 = expand_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_345: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [-1, 384]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_352: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_345, [32, -1, 6, 64]);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_9: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_98, view_352], 2);  clone_98 = view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_353: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [32, 512, 768]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_354: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_353, [16384, 768]);  view_353 = None
        permute_187: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        addmm_67: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg229_1, view_354, permute_187);  arg229_1 = view_354 = permute_187 = None
        view_355: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [32, 512, 768]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_117: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_355, convert_element_type_353);  view_355 = convert_element_type_353 = None
        convert_element_type_381: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.float32);  add_117 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_381, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_40: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_381, getitem_39);  convert_element_type_381 = getitem_39 = None
        add_118: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_76: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_19);  sub_40 = rsqrt_19 = None
        mul_77: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, arg230_1);  mul_76 = arg230_1 = None
        add_119: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_77, arg231_1);  mul_77 = arg231_1 = None
        convert_element_type_382: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_356: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_382, [16384, 768])
        permute_188: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_68: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg233_1, view_356, permute_188);  arg233_1 = view_356 = permute_188 = None
        view_357: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [32, 512, 3072]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_386: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_357, torch.float32);  view_357 = None
        mul_78: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_386, 0.5)
        mul_79: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_386, 0.7071067811865476);  convert_element_type_386 = None
        erf_9: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_120: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_80: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, add_120);  mul_78 = add_120 = None
        convert_element_type_387: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_358: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_387, [16384, 3072]);  convert_element_type_387 = None
        permute_189: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        addmm_69: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg235_1, view_358, permute_189);  arg235_1 = view_358 = permute_189 = None
        view_359: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [32, 512, 768]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_121: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_359, convert_element_type_382);  view_359 = convert_element_type_382 = None
        convert_element_type_391: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.float32);  add_121 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_391, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_42: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_75: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_42, 0);  iota_42 = unsqueeze_75 = None
        iota_43: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_76: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = unsqueeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_41: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_391, getitem_41);  convert_element_type_391 = getitem_41 = None
        add_122: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_81: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_20);  sub_41 = rsqrt_20 = None
        mul_82: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, arg236_1);  mul_81 = arg236_1 = None
        add_123: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, arg237_1);  mul_82 = arg237_1 = None
        convert_element_type_392: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_123, torch.bfloat16);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_364: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_392, [16384, 768])
        permute_194: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_72: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg246_1, view_364, permute_194);  arg246_1 = view_364 = permute_194 = None
        view_365: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [32, 512, 384]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_366: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_365, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_3: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view_360: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_392, [16384, 768])
        permute_190: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        addmm_70: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg239_1, view_360, permute_190);  arg239_1 = view_360 = permute_190 = None
        view_361: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [32, 512, 384]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_367: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_361, [32, 512, -1, 64]);  view_361 = None

        # No stacktrace found for following nodes
        permute_default_4: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_362: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_392, [16384, 768])
        permute_191: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        addmm_71: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg241_1, view_362, permute_191);  arg241_1 = view_362 = permute_191 = None
        view_363: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [32, 512, 384]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_368: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_363, [32, 512, -1, 64]);  view_363 = None

        # No stacktrace found for following nodes
        permute_default_5: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
        _scaled_dot_product_cudnn_attention_default_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default_3, permute_default_4, permute_default_5, None, False, scale = 0.125);  permute_default_3 = permute_default_4 = permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_53: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default_1[0];  _scaled_dot_product_cudnn_attention_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_205: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        clone_108: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_372: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_392, [16384, 768])
        permute_200: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_73: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg250_1, view_372, permute_200);  arg250_1 = view_372 = permute_200 = None
        view_373: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [32, 512, 384]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_201: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_373, [0, 2, 1]);  view_373 = None
        clone_102: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        unsqueeze_72: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_102, -1);  clone_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_10: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_72, [0, 0, 4, 4], 0.0);  unsqueeze_72 = None
        iota_40: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_73: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_40, 0);  iota_40 = None
        iota_41: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_74: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
        add_126: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_73, unsqueeze_74);  unsqueeze_73 = unsqueeze_74 = None
        unsqueeze_77: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_126, -1);  add_126 = None
        unsqueeze_78: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_77, -1);  unsqueeze_77 = None
        full_default_11: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_10: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_10, [None, None, unsqueeze_78, full_default_11]);  constant_pad_nd_10 = unsqueeze_78 = full_default_11 = None
        permute_202: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_10, [0, 1, 2, 4, 3, 5]);  index_10 = None
        view_375: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_202, [32, 3456, 512]);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_203: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
        view_376: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_203, [32, 512, 384, 9]);  permute_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_103: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_376, memory_format = torch.contiguous_format);  view_376 = None
        view_377: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [98304, 64, 9]);  clone_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_61: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_377, [98304, 64, 9]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_192: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_392, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_20: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_192, arg242_1, None, [1], [4], [1], False, [0], 768);  permute_192 = arg242_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_21: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_20, arg243_1, None, [1], [0], [1], False, [0], 1);  convolution_20 = arg243_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_124: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_21, arg244_1);  convolution_21 = arg244_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_198: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_124, [0, 2, 1]);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_83: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_198, view_365);  permute_198 = view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_101: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_83, memory_format = torch.contiguous_format);  mul_83 = None
        view_369: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [16384, 384]);  clone_101 = None
        permute_199: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        mm_10: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_369, permute_199);  view_369 = permute_199 = None
        view_370: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [32, 512, 54]);  mm_10 = None
        add_125: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_370, arg248_1);  view_370 = arg248_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_371: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_125, [-1, 9, 1]);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_404: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_371, torch.float32);  view_371 = None
        amax_20: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_404, [1], True)
        sub_42: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_404, amax_20);  convert_element_type_404 = amax_20 = None
        exp_20: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_21: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [1], True)
        div_30: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_405: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_30, torch.bfloat16);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_62: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_405, [98304, 9, 1]);  convert_element_type_405 = None
        bmm_30: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_61, expand_62);  expand_61 = expand_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_381: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [-1, 384]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_388: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_381, [32, -1, 6, 64]);  view_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_10: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_108, view_388], 2);  clone_108 = view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_389: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [32, 512, 768]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_390: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_389, [16384, 768]);  view_389 = None
        permute_206: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_74: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg252_1, view_390, permute_206);  arg252_1 = view_390 = permute_206 = None
        view_391: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [32, 512, 768]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_129: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_391, convert_element_type_392);  view_391 = convert_element_type_392 = None
        convert_element_type_420: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.float32);  add_129 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_420, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_44: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_420, getitem_43);  convert_element_type_420 = getitem_43 = None
        add_130: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_84: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_21);  sub_44 = rsqrt_21 = None
        mul_85: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, arg253_1);  mul_84 = arg253_1 = None
        add_131: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, arg254_1);  mul_85 = arg254_1 = None
        convert_element_type_421: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_421, [16384, 768])
        permute_207: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_75: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg256_1, view_392, permute_207);  arg256_1 = view_392 = permute_207 = None
        view_393: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [32, 512, 3072]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_425: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        mul_86: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_425, 0.5)
        mul_87: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_425, 0.7071067811865476);  convert_element_type_425 = None
        erf_10: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_132: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_88: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, add_132);  mul_86 = add_132 = None
        convert_element_type_426: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_394: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_426, [16384, 3072]);  convert_element_type_426 = None
        permute_208: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_76: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg258_1, view_394, permute_208);  arg258_1 = view_394 = permute_208 = None
        view_395: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [32, 512, 768]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_133: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_395, convert_element_type_421);  view_395 = convert_element_type_421 = None
        convert_element_type_430: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.float32);  add_133 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_430, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_46: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_82: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_46, 0);  iota_46 = unsqueeze_82 = None
        iota_47: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_83: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = unsqueeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_45: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_430, getitem_45);  convert_element_type_430 = getitem_45 = None
        add_134: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_89: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_22);  sub_45 = rsqrt_22 = None
        mul_90: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, arg259_1);  mul_89 = arg259_1 = None
        add_135: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, arg260_1);  mul_90 = arg260_1 = None
        convert_element_type_431: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_400: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_431, [16384, 768])
        permute_213: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_79: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg269_1, view_400, permute_213);  arg269_1 = view_400 = permute_213 = None
        view_401: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [32, 512, 384]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_402: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_401, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        view_396: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_431, [16384, 768])
        permute_209: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_77: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg262_1, view_396, permute_209);  arg262_1 = view_396 = permute_209 = None
        view_397: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [32, 512, 384]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_403: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_397, [32, 512, -1, 64]);  view_397 = None

        # No stacktrace found for following nodes
        permute_default_1: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_398: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_431, [16384, 768])
        permute_210: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_78: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg264_1, view_398, permute_210);  arg264_1 = view_398 = permute_210 = None
        view_399: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [32, 512, 384]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        view_404: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_399, [32, 512, -1, 64]);  view_399 = None

        # No stacktrace found for following nodes
        permute_default_2: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        _scaled_dot_product_cudnn_attention_default = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_default, permute_default_1, permute_default_2, None, False, scale = 0.125);  permute_default = permute_default_1 = permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_52: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_default[0];  _scaled_dot_product_cudnn_attention_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_224: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None
        clone_118: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        view_408: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_431, [16384, 768])
        permute_219: "bf16[768, 384][1, 768]cuda:0" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        addmm_80: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg273_1, view_408, permute_219);  arg273_1 = view_408 = permute_219 = None
        view_409: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [32, 512, 384]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_220: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
        clone_112: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        unsqueeze_79: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_112, -1);  clone_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_11: "bf16[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(unsqueeze_79, [0, 0, 4, 4], 0.0);  unsqueeze_79 = None
        iota_44: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_80: "i64[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_44, 0);  iota_44 = None
        iota_45: "i64[9][1]cuda:0" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_81: "i64[9, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
        add_138: "i64[9, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_80, unsqueeze_81);  unsqueeze_80 = unsqueeze_81 = None
        unsqueeze_84: "i64[9, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_138, -1);  add_138 = None
        unsqueeze_85: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        full_default_12: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_11: "bf16[32, 384, 9, 512, 1, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(constant_pad_nd_11, [None, None, unsqueeze_85, full_default_12]);  constant_pad_nd_11 = unsqueeze_85 = full_default_12 = None
        permute_221: "bf16[32, 384, 9, 1, 512, 1][1769472, 4608, 512, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(index_11, [0, 1, 2, 4, 3, 5]);  index_11 = None
        view_411: "bf16[32, 3456, 512][1769472, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_221, [32, 3456, 512]);  permute_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_222: "bf16[32, 512, 3456][1769472, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
        view_412: "bf16[32, 512, 384, 9][1769472, 1, 4608, 512]cuda:0" = torch.ops.aten.reshape.default(permute_222, [32, 512, 384, 9]);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_113: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.clone.default(view_412, memory_format = torch.contiguous_format);  view_412 = None
        view_413: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [98304, 64, 9]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_67: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.expand.default(view_413, [98304, 64, 9]);  view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_211: "bf16[32, 768, 512][393216, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_431, [0, 2, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_22: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.aten.convolution.default(permute_211, arg265_1, None, [1], [4], [1], False, [0], 768);  permute_211 = arg265_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_23: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.convolution.default(convolution_22, arg266_1, None, [1], [0], [1], False, [0], 1);  convolution_22 = arg266_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_136: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_23, arg267_1);  convolution_23 = arg267_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_217: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(add_136, [0, 2, 1]);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_91: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.mul.Tensor(permute_217, view_401);  permute_217 = view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_111: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(mul_91, memory_format = torch.contiguous_format);  mul_91 = None
        view_405: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_111, [16384, 384]);  clone_111 = None
        permute_218: "bf16[384, 54][1, 384]cuda:0" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        mm_11: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.mm.default(view_405, permute_218);  view_405 = permute_218 = None
        view_406: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [32, 512, 54]);  mm_11 = None
        add_137: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_406, arg271_1);  view_406 = arg271_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_407: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_137, [-1, 9, 1]);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_443: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_407, torch.float32);  view_407 = None
        amax_22: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_443, [1], True)
        sub_46: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_443, amax_22);  convert_element_type_443 = amax_22 = None
        exp_22: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_23: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [1], True)
        div_33: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_444: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16);  div_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_68: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_444, [98304, 9, 1]);  convert_element_type_444 = None
        bmm_33: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.bmm.default(expand_67, expand_68);  expand_67 = expand_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        view_417: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [-1, 384]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_424: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_417, [32, -1, 6, 64]);  view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_11: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.cat.default([clone_118, view_424], 2);  clone_118 = view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_425: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(cat_11, [32, 512, 768]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_426: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_425, [16384, 768]);  view_425 = None
        permute_225: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_81: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg275_1, view_426, permute_225);  arg275_1 = view_426 = permute_225 = None
        view_427: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [32, 512, 768]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_141: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_427, convert_element_type_431);  view_427 = convert_element_type_431 = None
        convert_element_type_459: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_141, torch.float32);  add_141 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_459, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_48: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_459, getitem_47);  convert_element_type_459 = getitem_47 = None
        add_142: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        mul_92: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_23);  sub_48 = rsqrt_23 = None
        mul_93: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, arg276_1);  mul_92 = arg276_1 = None
        add_143: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, arg277_1);  mul_93 = arg277_1 = None
        convert_element_type_460: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_428: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_460, [16384, 768])
        permute_226: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_82: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg279_1, view_428, permute_226);  arg279_1 = view_428 = permute_226 = None
        view_429: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [32, 512, 3072]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_464: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_429, torch.float32);  view_429 = None
        mul_94: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_464, 0.5)
        mul_95: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_464, 0.7071067811865476);  convert_element_type_464 = None
        erf_11: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_144: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, add_144);  mul_94 = add_144 = None
        convert_element_type_465: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_430: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_465, [16384, 3072]);  convert_element_type_465 = None
        permute_227: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        addmm_83: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg281_1, view_430, permute_227);  arg281_1 = view_430 = permute_227 = None
        view_431: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [32, 512, 768]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_145: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_431, convert_element_type_460);  view_431 = convert_element_type_460 = None
        convert_element_type_469: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.float32);  add_145 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_469, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        sub_49: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_469, getitem_49);  convert_element_type_469 = getitem_49 = None
        add_146: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        mul_97: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_24);  sub_49 = rsqrt_24 = None
        mul_98: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, arg282_1);  mul_97 = arg282_1 = None
        add_147: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_98, arg283_1);  mul_98 = arg283_1 = None
        convert_element_type_470: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.bfloat16);  add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:662 in forward, code: hidden_states = self.dense(generator_hidden_states)
        view_432: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_470, [16384, 768]);  convert_element_type_470 = None
        permute_228: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_84: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg285_1, view_432, permute_228);  arg285_1 = view_432 = permute_228 = None
        view_433: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [32, 512, 768]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_474: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_433, torch.float32);  view_433 = None
        mul_99: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_474, 0.5)
        mul_100: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_474, 0.7071067811865476);  convert_element_type_474 = None
        erf_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
        add_148: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_101: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, add_148);  mul_99 = add_148 = None
        convert_element_type_475: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:664 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_476: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_475, torch.float32);  convert_element_type_475 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_476, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[32, 512, 1][512, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:724 in forward, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_437: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(arg289_1, [-1]);  arg289_1 = None
        ne_1: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_437, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:718 in forward, code: prediction_scores = self.generator_lm_head(prediction_scores)
        full_default_15: "bf16[6][1]cuda:0" = torch.ops.aten.full.default([6], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "bf16[30528][1]cuda:0" = torch.ops.aten.cat.default([arg288_1, full_default_15]);  arg288_1 = full_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:664 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_50: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_476, getitem_51);  convert_element_type_476 = getitem_51 = None
        add_149: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_102: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_25);  sub_50 = rsqrt_25 = None
        mul_103: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, arg286_1);  mul_102 = arg286_1 = None
        add_150: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_103, arg287_1);  mul_103 = arg287_1 = None
        convert_element_type_477: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_150, torch.bfloat16);  add_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:718 in forward, code: prediction_scores = self.generator_lm_head(prediction_scores)
        view_434: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_477, [16384, 768]);  convert_element_type_477 = None
        permute_229: "bf16[768, 30522][1, 768]cuda:0" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        constant_pad_nd_default: "bf16[768, 30528][30528, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_229, [0, 6, 0, 0]);  permute_229 = None
        addmm_default: "bf16[16384, 30528][30528, 1]cuda:0" = torch.ops.aten.addmm.default(cat_default, view_434, constant_pad_nd_default);  cat_default = view_434 = constant_pad_nd_default = None
        slice_tensor: "bf16[16384, 30522][30528, 1]cuda:0" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -6);  addmm_default = None
        view_435: "bf16[32, 512, 30522][15630336, 30528, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [32, 512, 30522]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:724 in forward, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_436: "bf16[16384, 30522][30528, 1]cuda:0" = torch.ops.aten.reshape.default(view_435, [-1, 30522])
        convert_element_type_481: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        amax_24: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_481, [1], True)
        sub_51: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_481, amax_24);  convert_element_type_481 = amax_24 = None
        exp_24: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.exp.default(sub_51)
        sum_25: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_52: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_51, log);  sub_51 = log = None
        convert_element_type_482: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_52, torch.bfloat16);  sub_52 = None
        ne: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_437, -100)
        full_default_13: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384][1]cuda:0" = torch.ops.aten.where.self(ne, view_437, full_default_13);  ne = full_default_13 = None
        unsqueeze_86: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "bf16[16384, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_482, 1, unsqueeze_86);  convert_element_type_482 = unsqueeze_86 = None
        squeeze: "bf16[16384][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[16384][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_14: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[16384][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_14);  ne_1 = neg = full_default_14 = None
        sum_27: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_437, -100);  view_437 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_483: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.bfloat16);  sum_26 = None
        div_36: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_483);  sum_27 = convert_element_type_483 = None
        return (div_36, view_435)
