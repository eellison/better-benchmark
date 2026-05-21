class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "i64[1, 512]", arg2_1: "i64[1, 512]", arg3_1: "f16[30522, 1024]", arg4_1: "f16[2, 1024]", arg5_1: "f16[512, 1024]", arg6_1: "f16[1024]", arg7_1: "f16[1024]", arg8_1: "f16[1024, 1024]", arg9_1: "f16[1024]", arg10_1: "f16[1024, 1024]", arg11_1: "f16[1024]", arg12_1: "f16[1024, 1024]", arg13_1: "f16[1024]", arg14_1: "f16[1024, 1024]", arg15_1: "f16[1024]", arg16_1: "f16[1024]", arg17_1: "f16[1024]", arg18_1: "f16[3072, 1024]", arg19_1: "f16[3072]", arg20_1: "f16[1024, 3072]", arg21_1: "f16[1024]", arg22_1: "f16[1024]", arg23_1: "f16[1024]", arg24_1: "f16[1024, 1024]", arg25_1: "f16[1024]", arg26_1: "f16[1024, 1024]", arg27_1: "f16[1024]", arg28_1: "f16[1024, 1024]", arg29_1: "f16[1024]", arg30_1: "f16[1024, 1024]", arg31_1: "f16[1024]", arg32_1: "f16[1024]", arg33_1: "f16[1024]", arg34_1: "f16[3072, 1024]", arg35_1: "f16[3072]", arg36_1: "f16[1024, 3072]", arg37_1: "f16[1024]", arg38_1: "f16[1024]", arg39_1: "f16[1024]", arg40_1: "f16[1024, 1024]", arg41_1: "f16[1024]", arg42_1: "f16[1024, 1024]", arg43_1: "f16[1024]", arg44_1: "f16[1024, 1024]", arg45_1: "f16[1024]", arg46_1: "f16[1024, 1024]", arg47_1: "f16[1024]", arg48_1: "f16[1024]", arg49_1: "f16[1024]", arg50_1: "f16[3072, 1024]", arg51_1: "f16[3072]", arg52_1: "f16[1024, 3072]", arg53_1: "f16[1024]", arg54_1: "f16[1024]", arg55_1: "f16[1024]", arg56_1: "f16[1024, 1024]", arg57_1: "f16[1024]", arg58_1: "f16[1024, 1024]", arg59_1: "f16[1024]", arg60_1: "f16[1024, 1024]", arg61_1: "f16[1024]", arg62_1: "f16[1024, 1024]", arg63_1: "f16[1024]", arg64_1: "f16[1024]", arg65_1: "f16[1024]", arg66_1: "f16[3072, 1024]", arg67_1: "f16[3072]", arg68_1: "f16[1024, 3072]", arg69_1: "f16[1024]", arg70_1: "f16[1024]", arg71_1: "f16[1024]", arg72_1: "f16[1024, 1024]", arg73_1: "f16[1024]", arg74_1: "f16[1024, 1024]", arg75_1: "f16[1024]", arg76_1: "f16[1024, 1024]", arg77_1: "f16[1024]", arg78_1: "f16[1024, 1024]", arg79_1: "f16[1024]", arg80_1: "f16[1024]", arg81_1: "f16[1024]", arg82_1: "f16[3072, 1024]", arg83_1: "f16[3072]", arg84_1: "f16[1024, 3072]", arg85_1: "f16[1024]", arg86_1: "f16[1024]", arg87_1: "f16[1024]", arg88_1: "f16[1024, 1024]", arg89_1: "f16[1024]", arg90_1: "f16[1024, 1024]", arg91_1: "f16[1024]", arg92_1: "f16[1024, 1024]", arg93_1: "f16[1024]", arg94_1: "f16[1024, 1024]", arg95_1: "f16[1024]", arg96_1: "f16[1024]", arg97_1: "f16[1024]", arg98_1: "f16[3072, 1024]", arg99_1: "f16[3072]", arg100_1: "f16[1024, 3072]", arg101_1: "f16[1024]", arg102_1: "f16[1024]", arg103_1: "f16[1024]", arg104_1: "f16[1024, 1024]", arg105_1: "f16[1024]", arg106_1: "f16[1024, 1024]", arg107_1: "f16[1024]", arg108_1: "f16[1024, 1024]", arg109_1: "f16[1024]", arg110_1: "f16[1024, 1024]", arg111_1: "f16[1024]", arg112_1: "f16[1024]", arg113_1: "f16[1024]", arg114_1: "f16[3072, 1024]", arg115_1: "f16[3072]", arg116_1: "f16[1024, 3072]", arg117_1: "f16[1024]", arg118_1: "f16[1024]", arg119_1: "f16[1024]", arg120_1: "f16[1024, 1024]", arg121_1: "f16[1024]", arg122_1: "f16[1024, 1024]", arg123_1: "f16[1024]", arg124_1: "f16[1024, 1024]", arg125_1: "f16[1024]", arg126_1: "f16[1024, 1024]", arg127_1: "f16[1024]", arg128_1: "f16[1024]", arg129_1: "f16[1024]", arg130_1: "f16[3072, 1024]", arg131_1: "f16[3072]", arg132_1: "f16[1024, 3072]", arg133_1: "f16[1024]", arg134_1: "f16[1024]", arg135_1: "f16[1024]", arg136_1: "f16[1024, 1024]", arg137_1: "f16[1024]", arg138_1: "f16[1024, 1024]", arg139_1: "f16[1024]", arg140_1: "f16[1024, 1024]", arg141_1: "f16[1024]", arg142_1: "f16[1024, 1024]", arg143_1: "f16[1024]", arg144_1: "f16[1024]", arg145_1: "f16[1024]", arg146_1: "f16[3072, 1024]", arg147_1: "f16[3072]", arg148_1: "f16[1024, 3072]", arg149_1: "f16[1024]", arg150_1: "f16[1024]", arg151_1: "f16[1024]", arg152_1: "f16[1024, 1024]", arg153_1: "f16[1024]", arg154_1: "f16[1024, 1024]", arg155_1: "f16[1024]", arg156_1: "f16[1024, 1024]", arg157_1: "f16[1024]", arg158_1: "f16[1024, 1024]", arg159_1: "f16[1024]", arg160_1: "f16[1024]", arg161_1: "f16[1024]", arg162_1: "f16[3072, 1024]", arg163_1: "f16[3072]", arg164_1: "f16[1024, 3072]", arg165_1: "f16[1024]", arg166_1: "f16[1024]", arg167_1: "f16[1024]", arg168_1: "f16[1024, 1024]", arg169_1: "f16[1024]", arg170_1: "f16[1024, 1024]", arg171_1: "f16[1024]", arg172_1: "f16[1024, 1024]", arg173_1: "f16[1024]", arg174_1: "f16[1024, 1024]", arg175_1: "f16[1024]", arg176_1: "f16[1024]", arg177_1: "f16[1024]", arg178_1: "f16[3072, 1024]", arg179_1: "f16[3072]", arg180_1: "f16[1024, 3072]", arg181_1: "f16[1024]", arg182_1: "f16[1024]", arg183_1: "f16[1024]", arg184_1: "f16[1024, 1024]", arg185_1: "f16[1024]", arg186_1: "f16[1024, 1024]", arg187_1: "f16[1024]", arg188_1: "f16[1024, 1024]", arg189_1: "f16[1024]", arg190_1: "f16[1024, 1024]", arg191_1: "f16[1024]", arg192_1: "f16[1024]", arg193_1: "f16[1024]", arg194_1: "f16[3072, 1024]", arg195_1: "f16[3072]", arg196_1: "f16[1024, 3072]", arg197_1: "f16[1024]", arg198_1: "f16[1024]", arg199_1: "f16[1024]", arg200_1: "f16[1024, 1024]", arg201_1: "f16[1024]", arg202_1: "f16[1024, 1024]", arg203_1: "f16[1024]", arg204_1: "f16[1024, 1024]", arg205_1: "f16[1024]", arg206_1: "f16[1024, 1024]", arg207_1: "f16[1024]", arg208_1: "f16[1024]", arg209_1: "f16[1024]", arg210_1: "f16[3072, 1024]", arg211_1: "f16[3072]", arg212_1: "f16[1024, 3072]", arg213_1: "f16[1024]", arg214_1: "f16[1024]", arg215_1: "f16[1024]", arg216_1: "f16[1024, 1024]", arg217_1: "f16[1024]", arg218_1: "f16[1024, 1024]", arg219_1: "f16[1024]", arg220_1: "f16[1024, 1024]", arg221_1: "f16[1024]", arg222_1: "f16[1024, 1024]", arg223_1: "f16[1024]", arg224_1: "f16[1024]", arg225_1: "f16[1024]", arg226_1: "f16[3072, 1024]", arg227_1: "f16[3072]", arg228_1: "f16[1024, 3072]", arg229_1: "f16[1024]", arg230_1: "f16[1024]", arg231_1: "f16[1024]", arg232_1: "f16[1024, 1024]", arg233_1: "f16[1024]", arg234_1: "f16[1024, 1024]", arg235_1: "f16[1024]", arg236_1: "f16[1024, 1024]", arg237_1: "f16[1024]", arg238_1: "f16[1024, 1024]", arg239_1: "f16[1024]", arg240_1: "f16[1024]", arg241_1: "f16[1024]", arg242_1: "f16[3072, 1024]", arg243_1: "f16[3072]", arg244_1: "f16[1024, 3072]", arg245_1: "f16[1024]", arg246_1: "f16[1024]", arg247_1: "f16[1024]", arg248_1: "f16[1024, 1024]", arg249_1: "f16[1024]", arg250_1: "f16[1024, 1024]", arg251_1: "f16[1024]", arg252_1: "f16[1024, 1024]", arg253_1: "f16[1024]", arg254_1: "f16[1024, 1024]", arg255_1: "f16[1024]", arg256_1: "f16[1024]", arg257_1: "f16[1024]", arg258_1: "f16[3072, 1024]", arg259_1: "f16[3072]", arg260_1: "f16[1024, 3072]", arg261_1: "f16[1024]", arg262_1: "f16[1024]", arg263_1: "f16[1024]", arg264_1: "f16[1024, 1024]", arg265_1: "f16[1024]", arg266_1: "f16[1024, 1024]", arg267_1: "f16[1024]", arg268_1: "f16[1024, 1024]", arg269_1: "f16[1024]", arg270_1: "f16[1024, 1024]", arg271_1: "f16[1024]", arg272_1: "f16[1024]", arg273_1: "f16[1024]", arg274_1: "f16[3072, 1024]", arg275_1: "f16[3072]", arg276_1: "f16[1024, 3072]", arg277_1: "f16[1024]", arg278_1: "f16[1024]", arg279_1: "f16[1024]", arg280_1: "f16[1024, 1024]", arg281_1: "f16[1024]", arg282_1: "f16[1024, 1024]", arg283_1: "f16[1024]", arg284_1: "f16[1024, 1024]", arg285_1: "f16[1024]", arg286_1: "f16[1024, 1024]", arg287_1: "f16[1024]", arg288_1: "f16[1024]", arg289_1: "f16[1024]", arg290_1: "f16[3072, 1024]", arg291_1: "f16[3072]", arg292_1: "f16[1024, 3072]", arg293_1: "f16[1024]", arg294_1: "f16[1024]", arg295_1: "f16[1024]", arg296_1: "f16[1024, 1024]", arg297_1: "f16[1024]", arg298_1: "f16[1024, 1024]", arg299_1: "f16[1024]", arg300_1: "f16[1024, 1024]", arg301_1: "f16[1024]", arg302_1: "f16[1024, 1024]", arg303_1: "f16[1024]", arg304_1: "f16[1024]", arg305_1: "f16[1024]", arg306_1: "f16[3072, 1024]", arg307_1: "f16[3072]", arg308_1: "f16[1024, 3072]", arg309_1: "f16[1024]", arg310_1: "f16[1024]", arg311_1: "f16[1024]", arg312_1: "f16[1024, 1024]", arg313_1: "f16[1024]", arg314_1: "f16[1024, 1024]", arg315_1: "f16[1024]", arg316_1: "f16[1024, 1024]", arg317_1: "f16[1024]", arg318_1: "f16[1024, 1024]", arg319_1: "f16[1024]", arg320_1: "f16[1024]", arg321_1: "f16[1024]", arg322_1: "f16[3072, 1024]", arg323_1: "f16[3072]", arg324_1: "f16[1024, 3072]", arg325_1: "f16[1024]", arg326_1: "f16[1024]", arg327_1: "f16[1024]", arg328_1: "f16[1024, 1024]", arg329_1: "f16[1024]", arg330_1: "f16[1024, 1024]", arg331_1: "f16[1024]", arg332_1: "f16[1024, 1024]", arg333_1: "f16[1024]", arg334_1: "f16[1024, 1024]", arg335_1: "f16[1024]", arg336_1: "f16[1024]", arg337_1: "f16[1024]", arg338_1: "f16[3072, 1024]", arg339_1: "f16[3072]", arg340_1: "f16[1024, 3072]", arg341_1: "f16[1024]", arg342_1: "f16[1024]", arg343_1: "f16[1024]", arg344_1: "f16[1024, 1024]", arg345_1: "f16[1024]", arg346_1: "f16[1024, 1024]", arg347_1: "f16[1024]", arg348_1: "f16[1024, 1024]", arg349_1: "f16[1024]", arg350_1: "f16[1024, 1024]", arg351_1: "f16[1024]", arg352_1: "f16[1024]", arg353_1: "f16[1024]", arg354_1: "f16[3072, 1024]", arg355_1: "f16[3072]", arg356_1: "f16[1024, 3072]", arg357_1: "f16[1024]", arg358_1: "f16[1024]", arg359_1: "f16[1024]", arg360_1: "f16[1024, 1024]", arg361_1: "f16[1024]", arg362_1: "f16[1024, 1024]", arg363_1: "f16[1024]", arg364_1: "f16[1024, 1024]", arg365_1: "f16[1024]", arg366_1: "f16[1024, 1024]", arg367_1: "f16[1024]", arg368_1: "f16[1024]", arg369_1: "f16[1024]", arg370_1: "f16[3072, 1024]", arg371_1: "f16[3072]", arg372_1: "f16[1024, 3072]", arg373_1: "f16[1024]", arg374_1: "f16[1024]", arg375_1: "f16[1024]", arg376_1: "f16[1024, 1024]", arg377_1: "f16[1024]", arg378_1: "f16[1024, 1024]", arg379_1: "f16[1024]", arg380_1: "f16[1024, 1024]", arg381_1: "f16[1024]", arg382_1: "f16[1024, 1024]", arg383_1: "f16[1024]", arg384_1: "f16[1024]", arg385_1: "f16[1024]", arg386_1: "f16[3072, 1024]", arg387_1: "f16[3072]", arg388_1: "f16[1024, 3072]", arg389_1: "f16[1024]", arg390_1: "f16[1024]", arg391_1: "f16[1024]", arg392_1: "f16[1024, 1024]", arg393_1: "f16[1024]", arg394_1: "f16[1024]", arg395_1: "f16[1024]", arg396_1: "f16[30522]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f16[1, 512, 1024]" = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:96 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand: "i64[1, 512]" = torch.ops.aten.expand.default(arg2_1, [1, -1]);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:97 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather: "i64[1, 512]" = torch.ops.aten.gather.default(expand, 1, arg1_1);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:98 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[1, 512]" = torch.ops.aten.expand.default(gather, [1, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:104 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f16[1, 512, 1024]" = torch.ops.aten.embedding.default(arg4_1, expand_1);  arg4_1 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:105 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:107 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f16[1, 512, 1024]" = torch.ops.aten.embedding.default(arg5_1, arg1_1);  arg5_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:108 in forward, code: embeddings = embeddings + position_embeddings
        add_1: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        convert_element_type: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean[0]
        getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        convert_element_type_1: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_3, torch.float16);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_1, [512, 1024])
        permute: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm, [1, 512, 1024]);  addmm = None
        view_2: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1, [1, 512, -1, 64]);  view_1 = None
        permute_1: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_11: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        mul_2: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_11, 0.3535533905932738);  convert_element_type_11 = None
        expand_3: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_2, [1, 16, 512, 64]);  mul_2 = None
        view_9: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_3, [16, 512, 64]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_3: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_1, [512, 1024])
        permute_2: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_1: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg11_1, view_3, permute_2);  arg11_1 = view_3 = permute_2 = None
        view_4: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 1024]);  addmm_1 = None
        view_5: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_4, [1, 512, -1, 64]);  view_4 = None
        permute_3: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_12: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_3, torch.float32);  permute_3 = None
        permute_6: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_12, [0, 1, 3, 2]);  convert_element_type_12 = None
        mul_3: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        expand_4: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_3, [1, 16, 64, 512]);  mul_3 = None
        view_10: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_4, [16, 64, 512]);  expand_4 = None
        bmm: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm, [1, 16, 512, 512]);  bmm = None

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
        add_6: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = where = None
        eq: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_6, -inf)
        logical_not: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_2: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_6, [-1], True)
        sub_1: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
        exp: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        where_1: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = full_default_2 = div = None
        expand_5: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_1, [1, 16, 512, 512]);  where_1 = None
        view_12: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_5, [16, 512, 512]);  expand_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_6: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_1, [512, 1024])
        permute_4: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_2: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg13_1, view_6, permute_4);  arg13_1 = view_6 = permute_4 = None
        view_7: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_2, [1, 512, 1024]);  addmm_2 = None
        view_8: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_7, [1, 512, -1, 64]);  view_7 = None
        permute_5: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_13: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_5, torch.float32);  permute_5 = None
        expand_6: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_13, [1, 16, 512, 64]);  convert_element_type_13 = None
        view_13: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_6, [16, 512, 64]);  expand_6 = None
        bmm_1: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
        view_14: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 16, 512, 64]);  bmm_1 = None
        convert_element_type_15: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_14, torch.float16);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_15, [0, 2, 1, 3]);  convert_element_type_15 = None
        clone_1: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_1, [1, 512, -1]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_15, [512, 1024]);  view_15 = None
        permute_8: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_3: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg15_1, view_16, permute_8);  arg15_1 = view_16 = permute_8 = None
        view_17: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_7: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_17, convert_element_type_1);  view_17 = convert_element_type_1 = None
        convert_element_type_19: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_19, [2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_19, getitem_3);  convert_element_type_19 = getitem_3 = None
        add_8: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_4: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_4, arg16_1);  mul_4 = arg16_1 = None
        add_9: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_5, arg17_1);  mul_5 = arg17_1 = None
        convert_element_type_20: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_9, torch.float16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_20, [512, 1024])
        permute_9: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
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
        permute_10: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_5: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg21_1, view_20, permute_10);  arg21_1 = view_20 = permute_10 = None
        view_21: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_21, convert_element_type_20);  view_21 = convert_element_type_20 = None
        convert_element_type_29: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32);  add_11 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_29, [2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_29, getitem_5);  convert_element_type_29 = getitem_5 = None
        add_12: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_9: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_9, arg22_1);  mul_9 = arg22_1 = None
        add_13: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_10, arg23_1);  mul_10 = arg23_1 = None
        convert_element_type_30: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_13, torch.float16);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_22: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_30, [512, 1024])
        permute_11: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_6: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg25_1, view_22, permute_11);  arg25_1 = view_22 = permute_11 = None
        view_23: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_6, [1, 512, 1024]);  addmm_6 = None
        view_24: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_23, [1, 512, -1, 64]);  view_23 = None
        permute_12: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_40: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_12, torch.float32);  permute_12 = None
        mul_11: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_40, 0.3535533905932738);  convert_element_type_40 = None
        expand_7: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_11, [1, 16, 512, 64]);  mul_11 = None
        view_31: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_7, [16, 512, 64]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_25: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_30, [512, 1024])
        permute_13: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_7: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg27_1, view_25, permute_13);  arg27_1 = view_25 = permute_13 = None
        view_26: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 1024]);  addmm_7 = None
        view_27: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_26, [1, 512, -1, 64]);  view_26 = None
        permute_14: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_41: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_14, torch.float32);  permute_14 = None
        permute_17: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_41, [0, 1, 3, 2]);  convert_element_type_41 = None
        mul_12: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_17, 0.3535533905932738);  permute_17 = None
        expand_8: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_12, [1, 16, 64, 512]);  mul_12 = None
        view_32: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_8, [16, 64, 512]);  expand_8 = None
        bmm_2: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_31, view_32);  view_31 = view_32 = None
        view_33: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [1, 16, 512, 512]);  bmm_2 = None
        full_default_4: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        add_14: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_33, where_2);  view_33 = where_2 = None
        eq_1: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_14, -inf)
        logical_not_2: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_default_5: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_1: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_14, [-1], True)
        sub_4: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_14, amax_1);  add_14 = amax_1 = None
        exp_1: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        where_3: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full_default_5, div_1);  logical_not_3 = full_default_5 = div_1 = None
        expand_9: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_3, [1, 16, 512, 512]);  where_3 = None
        view_34: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_9, [16, 512, 512]);  expand_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_28: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_30, [512, 1024])
        permute_15: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_8: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg29_1, view_28, permute_15);  arg29_1 = view_28 = permute_15 = None
        view_29: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_8, [1, 512, 1024]);  addmm_8 = None
        view_30: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_29, [1, 512, -1, 64]);  view_29 = None
        permute_16: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_42: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_16, torch.float32);  permute_16 = None
        expand_10: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_42, [1, 16, 512, 64]);  convert_element_type_42 = None
        view_35: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_10, [16, 512, 64]);  expand_10 = None
        bmm_3: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
        view_36: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 16, 512, 64]);  bmm_3 = None
        convert_element_type_44: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_36, torch.float16);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_44, [0, 2, 1, 3]);  convert_element_type_44 = None
        clone_4: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_37: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_4, [1, 512, -1]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_37, [512, 1024]);  view_37 = None
        permute_19: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_9: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg31_1, view_38, permute_19);  arg31_1 = view_38 = permute_19 = None
        view_39: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_9, [1, 512, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_39, convert_element_type_30);  view_39 = convert_element_type_30 = None
        convert_element_type_48: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_15, torch.float32);  add_15 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_7);  convert_element_type_48 = getitem_7 = None
        add_16: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_13: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_14: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_13, arg32_1);  mul_13 = arg32_1 = None
        add_17: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_14, arg33_1);  mul_14 = arg33_1 = None
        convert_element_type_49: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_17, torch.float16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_49, [512, 1024])
        permute_20: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
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
        permute_21: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_11: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg37_1, view_42, permute_21);  arg37_1 = view_42 = permute_21 = None
        view_43: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_19: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_43, convert_element_type_49);  view_43 = convert_element_type_49 = None
        convert_element_type_58: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_19, torch.float32);  add_19 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_58, [2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_58, getitem_9);  convert_element_type_58 = getitem_9 = None
        add_20: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_18: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_19: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_18, arg38_1);  mul_18 = arg38_1 = None
        add_21: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_19, arg39_1);  mul_19 = arg39_1 = None
        convert_element_type_59: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_21, torch.float16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_44: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_59, [512, 1024])
        permute_22: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_12: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg41_1, view_44, permute_22);  arg41_1 = view_44 = permute_22 = None
        view_45: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_12, [1, 512, 1024]);  addmm_12 = None
        view_46: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_45, [1, 512, -1, 64]);  view_45 = None
        permute_23: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_69: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_23, torch.float32);  permute_23 = None
        mul_20: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_69, 0.3535533905932738);  convert_element_type_69 = None
        expand_11: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_20, [1, 16, 512, 64]);  mul_20 = None
        view_53: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_11, [16, 512, 64]);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_47: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_59, [512, 1024])
        permute_24: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_13: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg43_1, view_47, permute_24);  arg43_1 = view_47 = permute_24 = None
        view_48: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 1024]);  addmm_13 = None
        view_49: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_48, [1, 512, -1, 64]);  view_48 = None
        permute_25: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_70: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_25, torch.float32);  permute_25 = None
        permute_28: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_70, [0, 1, 3, 2]);  convert_element_type_70 = None
        mul_21: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_28, 0.3535533905932738);  permute_28 = None
        expand_12: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_21, [1, 16, 64, 512]);  mul_21 = None
        view_54: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_12, [16, 64, 512]);  expand_12 = None
        bmm_4: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_53, view_54);  view_53 = view_54 = None
        view_55: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [1, 16, 512, 512]);  bmm_4 = None
        full_default_7: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        add_22: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_55, where_4);  view_55 = where_4 = None
        eq_2: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_22, -inf)
        logical_not_4: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_default_8: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_2: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_22, [-1], True)
        sub_7: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_22, amax_2);  add_22 = amax_2 = None
        exp_2: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        where_5: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full_default_8, div_2);  logical_not_5 = full_default_8 = div_2 = None
        expand_13: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_5, [1, 16, 512, 512]);  where_5 = None
        view_56: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_13, [16, 512, 512]);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_50: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_59, [512, 1024])
        permute_26: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_14: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg45_1, view_50, permute_26);  arg45_1 = view_50 = permute_26 = None
        view_51: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_14, [1, 512, 1024]);  addmm_14 = None
        view_52: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_51, [1, 512, -1, 64]);  view_51 = None
        permute_27: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_71: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_27, torch.float32);  permute_27 = None
        expand_14: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_71, [1, 16, 512, 64]);  convert_element_type_71 = None
        view_57: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_14, [16, 512, 64]);  expand_14 = None
        bmm_5: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_56, view_57);  view_56 = view_57 = None
        view_58: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 16, 512, 64]);  bmm_5 = None
        convert_element_type_73: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_58, torch.float16);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_73, [0, 2, 1, 3]);  convert_element_type_73 = None
        clone_7: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_59: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_7, [1, 512, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_59, [512, 1024]);  view_59 = None
        permute_30: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_15: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg47_1, view_60, permute_30);  arg47_1 = view_60 = permute_30 = None
        view_61: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_15, [1, 512, 1024]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_61, convert_element_type_59);  view_61 = convert_element_type_59 = None
        convert_element_type_77: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_77, [2], correction = 0, keepdim = True)
        getitem_10: "f32[1, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_77, getitem_11);  convert_element_type_77 = getitem_11 = None
        add_24: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_22: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_23: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_22, arg48_1);  mul_22 = arg48_1 = None
        add_25: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_23, arg49_1);  mul_23 = arg49_1 = None
        convert_element_type_78: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_25, torch.float16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_78, [512, 1024])
        permute_31: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
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
        permute_32: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_17: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg53_1, view_64, permute_32);  arg53_1 = view_64 = permute_32 = None
        view_65: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_65, convert_element_type_78);  view_65 = convert_element_type_78 = None
        convert_element_type_87: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_87, [2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_9: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_87, getitem_13);  convert_element_type_87 = getitem_13 = None
        add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_27: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_28: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_27, arg54_1);  mul_27 = arg54_1 = None
        add_29: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_28, arg55_1);  mul_28 = arg55_1 = None
        convert_element_type_88: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_29, torch.float16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_66: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_88, [512, 1024])
        permute_33: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_18: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg57_1, view_66, permute_33);  arg57_1 = view_66 = permute_33 = None
        view_67: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_18, [1, 512, 1024]);  addmm_18 = None
        view_68: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_67, [1, 512, -1, 64]);  view_67 = None
        permute_34: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_98: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_34, torch.float32);  permute_34 = None
        mul_29: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_98, 0.3535533905932738);  convert_element_type_98 = None
        expand_15: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_29, [1, 16, 512, 64]);  mul_29 = None
        view_75: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_15, [16, 512, 64]);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_69: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_88, [512, 1024])
        permute_35: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_19: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg59_1, view_69, permute_35);  arg59_1 = view_69 = permute_35 = None
        view_70: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 1024]);  addmm_19 = None
        view_71: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_70, [1, 512, -1, 64]);  view_70 = None
        permute_36: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_99: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_36, torch.float32);  permute_36 = None
        permute_39: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_99, [0, 1, 3, 2]);  convert_element_type_99 = None
        mul_30: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_39, 0.3535533905932738);  permute_39 = None
        expand_16: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_30, [1, 16, 64, 512]);  mul_30 = None
        view_76: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_16, [16, 64, 512]);  expand_16 = None
        bmm_6: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_75, view_76);  view_75 = view_76 = None
        view_77: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [1, 16, 512, 512]);  bmm_6 = None
        full_default_10: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        add_30: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_77, where_6);  view_77 = where_6 = None
        eq_3: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_30, -inf)
        logical_not_6: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_default_11: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_3: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_30, [-1], True)
        sub_10: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_30, amax_3);  add_30 = amax_3 = None
        exp_3: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        where_7: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full_default_11, div_3);  logical_not_7 = full_default_11 = div_3 = None
        expand_17: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_7, [1, 16, 512, 512]);  where_7 = None
        view_78: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_17, [16, 512, 512]);  expand_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_72: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_88, [512, 1024])
        permute_37: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_20: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg61_1, view_72, permute_37);  arg61_1 = view_72 = permute_37 = None
        view_73: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_20, [1, 512, 1024]);  addmm_20 = None
        view_74: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_73, [1, 512, -1, 64]);  view_73 = None
        permute_38: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_100: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_38, torch.float32);  permute_38 = None
        expand_18: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_100, [1, 16, 512, 64]);  convert_element_type_100 = None
        view_79: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_18, [16, 512, 64]);  expand_18 = None
        bmm_7: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
        view_80: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 16, 512, 64]);  bmm_7 = None
        convert_element_type_102: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_80, torch.float16);  view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_102, [0, 2, 1, 3]);  convert_element_type_102 = None
        clone_10: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_81: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_10, [1, 512, -1]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_82: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_81, [512, 1024]);  view_81 = None
        permute_41: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_21: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg63_1, view_82, permute_41);  arg63_1 = view_82 = permute_41 = None
        view_83: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 1024]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_83, convert_element_type_88);  view_83 = convert_element_type_88 = None
        convert_element_type_106: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32);  add_31 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_106, [2], correction = 0, keepdim = True)
        getitem_14: "f32[1, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_106, getitem_15);  convert_element_type_106 = getitem_15 = None
        add_32: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_31: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_32: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_31, arg64_1);  mul_31 = arg64_1 = None
        add_33: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_32, arg65_1);  mul_32 = arg65_1 = None
        convert_element_type_107: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_33, torch.float16);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_107, [512, 1024])
        permute_42: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
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
        permute_43: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_23: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg69_1, view_86, permute_43);  arg69_1 = view_86 = permute_43 = None
        view_87: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_35: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_87, convert_element_type_107);  view_87 = convert_element_type_107 = None
        convert_element_type_116: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_116, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_12: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_116, getitem_17);  convert_element_type_116 = getitem_17 = None
        add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_36: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_37: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_36, arg70_1);  mul_36 = arg70_1 = None
        add_37: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_37, arg71_1);  mul_37 = arg71_1 = None
        convert_element_type_117: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_88: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_117, [512, 1024])
        permute_44: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_24: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg73_1, view_88, permute_44);  arg73_1 = view_88 = permute_44 = None
        view_89: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_24, [1, 512, 1024]);  addmm_24 = None
        view_90: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_89, [1, 512, -1, 64]);  view_89 = None
        permute_45: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_127: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_45, torch.float32);  permute_45 = None
        mul_38: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_127, 0.3535533905932738);  convert_element_type_127 = None
        expand_19: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_38, [1, 16, 512, 64]);  mul_38 = None
        view_97: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_19, [16, 512, 64]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_91: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_117, [512, 1024])
        permute_46: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_25: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg75_1, view_91, permute_46);  arg75_1 = view_91 = permute_46 = None
        view_92: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_25, [1, 512, 1024]);  addmm_25 = None
        view_93: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_92, [1, 512, -1, 64]);  view_92 = None
        permute_47: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_128: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_47, torch.float32);  permute_47 = None
        permute_50: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_128, [0, 1, 3, 2]);  convert_element_type_128 = None
        mul_39: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_50, 0.3535533905932738);  permute_50 = None
        expand_20: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_39, [1, 16, 64, 512]);  mul_39 = None
        view_98: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_20, [16, 64, 512]);  expand_20 = None
        bmm_8: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
        view_99: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [1, 16, 512, 512]);  bmm_8 = None
        full_default_13: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        add_38: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_99, where_8);  view_99 = where_8 = None
        eq_4: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_38, -inf)
        logical_not_8: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_default_14: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_4: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_38, [-1], True)
        sub_13: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_38, amax_4);  add_38 = amax_4 = None
        exp_4: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        where_9: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full_default_14, div_4);  logical_not_9 = full_default_14 = div_4 = None
        expand_21: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_9, [1, 16, 512, 512]);  where_9 = None
        view_100: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_21, [16, 512, 512]);  expand_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_94: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_117, [512, 1024])
        permute_48: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_26: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg77_1, view_94, permute_48);  arg77_1 = view_94 = permute_48 = None
        view_95: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_26, [1, 512, 1024]);  addmm_26 = None
        view_96: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_95, [1, 512, -1, 64]);  view_95 = None
        permute_49: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_129: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_49, torch.float32);  permute_49 = None
        expand_22: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_129, [1, 16, 512, 64]);  convert_element_type_129 = None
        view_101: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_22, [16, 512, 64]);  expand_22 = None
        bmm_9: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
        view_102: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 16, 512, 64]);  bmm_9 = None
        convert_element_type_131: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_102, torch.float16);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_131, [0, 2, 1, 3]);  convert_element_type_131 = None
        clone_13: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_103: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_13, [1, 512, -1]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_103, [512, 1024]);  view_103 = None
        permute_52: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_27: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg79_1, view_104, permute_52);  arg79_1 = view_104 = permute_52 = None
        view_105: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_27, [1, 512, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_105, convert_element_type_117);  view_105 = convert_element_type_117 = None
        convert_element_type_135: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_39, torch.float32);  add_39 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_135, [2], correction = 0, keepdim = True)
        getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_135, getitem_19);  convert_element_type_135 = getitem_19 = None
        add_40: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_40: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_41: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_40, arg80_1);  mul_40 = arg80_1 = None
        add_41: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_41, arg81_1);  mul_41 = arg81_1 = None
        convert_element_type_136: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_41, torch.float16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_136, [512, 1024])
        permute_53: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
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
        permute_54: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_29: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg85_1, view_108, permute_54);  arg85_1 = view_108 = permute_54 = None
        view_109: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_29, [1, 512, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_109, convert_element_type_136);  view_109 = convert_element_type_136 = None
        convert_element_type_145: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_43, torch.float32);  add_43 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_145, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_15: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_145, getitem_21);  convert_element_type_145 = getitem_21 = None
        add_44: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_45: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_46: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_45, arg86_1);  mul_45 = arg86_1 = None
        add_45: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_46, arg87_1);  mul_46 = arg87_1 = None
        convert_element_type_146: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_45, torch.float16);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_110: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_146, [512, 1024])
        permute_55: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_30: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg89_1, view_110, permute_55);  arg89_1 = view_110 = permute_55 = None
        view_111: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_30, [1, 512, 1024]);  addmm_30 = None
        view_112: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_111, [1, 512, -1, 64]);  view_111 = None
        permute_56: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_156: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_56, torch.float32);  permute_56 = None
        mul_47: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_156, 0.3535533905932738);  convert_element_type_156 = None
        expand_23: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_47, [1, 16, 512, 64]);  mul_47 = None
        view_119: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_23, [16, 512, 64]);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_113: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_146, [512, 1024])
        permute_57: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_31: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg91_1, view_113, permute_57);  arg91_1 = view_113 = permute_57 = None
        view_114: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_31, [1, 512, 1024]);  addmm_31 = None
        view_115: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_114, [1, 512, -1, 64]);  view_114 = None
        permute_58: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_157: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_58, torch.float32);  permute_58 = None
        permute_61: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_157, [0, 1, 3, 2]);  convert_element_type_157 = None
        mul_48: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_61, 0.3535533905932738);  permute_61 = None
        expand_24: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_48, [1, 16, 64, 512]);  mul_48 = None
        view_120: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_24, [16, 64, 512]);  expand_24 = None
        bmm_10: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
        view_121: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [1, 16, 512, 512]);  bmm_10 = None
        full_default_16: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        add_46: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_121, where_10);  view_121 = where_10 = None
        eq_5: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_46, -inf)
        logical_not_10: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_default_17: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_5: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_46, [-1], True)
        sub_16: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_46, amax_5);  add_46 = amax_5 = None
        exp_5: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        where_11: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full_default_17, div_5);  logical_not_11 = full_default_17 = div_5 = None
        expand_25: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_11, [1, 16, 512, 512]);  where_11 = None
        view_122: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_25, [16, 512, 512]);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_116: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_146, [512, 1024])
        permute_59: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_32: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg93_1, view_116, permute_59);  arg93_1 = view_116 = permute_59 = None
        view_117: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_32, [1, 512, 1024]);  addmm_32 = None
        view_118: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_117, [1, 512, -1, 64]);  view_117 = None
        permute_60: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_158: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_60, torch.float32);  permute_60 = None
        expand_26: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_158, [1, 16, 512, 64]);  convert_element_type_158 = None
        view_123: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_26, [16, 512, 64]);  expand_26 = None
        bmm_11: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
        view_124: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 16, 512, 64]);  bmm_11 = None
        convert_element_type_160: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_124, torch.float16);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_160, [0, 2, 1, 3]);  convert_element_type_160 = None
        clone_16: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_125: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_16, [1, 512, -1]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_125, [512, 1024]);  view_125 = None
        permute_63: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_33: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg95_1, view_126, permute_63);  arg95_1 = view_126 = permute_63 = None
        view_127: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_33, [1, 512, 1024]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_47: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_127, convert_element_type_146);  view_127 = convert_element_type_146 = None
        convert_element_type_164: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_47, torch.float32);  add_47 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_164, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_164, getitem_23);  convert_element_type_164 = getitem_23 = None
        add_48: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_49: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_50: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_49, arg96_1);  mul_49 = arg96_1 = None
        add_49: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_50, arg97_1);  mul_50 = arg97_1 = None
        convert_element_type_165: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_49, torch.float16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_165, [512, 1024])
        permute_64: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
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
        permute_65: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_35: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg101_1, view_130, permute_65);  arg101_1 = view_130 = permute_65 = None
        view_131: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_35, [1, 512, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_51: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_131, convert_element_type_165);  view_131 = convert_element_type_165 = None
        convert_element_type_174: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32);  add_51 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_174, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_18: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_174, getitem_25);  convert_element_type_174 = getitem_25 = None
        add_52: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_54: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_55: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_54, arg102_1);  mul_54 = arg102_1 = None
        add_53: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_55, arg103_1);  mul_55 = arg103_1 = None
        convert_element_type_175: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_53, torch.float16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_132: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_175, [512, 1024])
        permute_66: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_36: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg105_1, view_132, permute_66);  arg105_1 = view_132 = permute_66 = None
        view_133: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_36, [1, 512, 1024]);  addmm_36 = None
        view_134: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_133, [1, 512, -1, 64]);  view_133 = None
        permute_67: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_185: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_67, torch.float32);  permute_67 = None
        mul_56: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_185, 0.3535533905932738);  convert_element_type_185 = None
        expand_27: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_56, [1, 16, 512, 64]);  mul_56 = None
        view_141: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_27, [16, 512, 64]);  expand_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_135: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_175, [512, 1024])
        permute_68: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_37: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg107_1, view_135, permute_68);  arg107_1 = view_135 = permute_68 = None
        view_136: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_37, [1, 512, 1024]);  addmm_37 = None
        view_137: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_136, [1, 512, -1, 64]);  view_136 = None
        permute_69: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_186: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_69, torch.float32);  permute_69 = None
        permute_72: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_186, [0, 1, 3, 2]);  convert_element_type_186 = None
        mul_57: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_72, 0.3535533905932738);  permute_72 = None
        expand_28: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_57, [1, 16, 64, 512]);  mul_57 = None
        view_142: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_28, [16, 64, 512]);  expand_28 = None
        bmm_12: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_141, view_142);  view_141 = view_142 = None
        view_143: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [1, 16, 512, 512]);  bmm_12 = None
        full_default_19: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        add_54: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_143, where_12);  view_143 = where_12 = None
        eq_6: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_54, -inf)
        logical_not_12: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        full_default_20: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_6: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_54, [-1], True)
        sub_19: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_54, amax_6);  add_54 = amax_6 = None
        exp_6: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        where_13: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_13, full_default_20, div_6);  logical_not_13 = full_default_20 = div_6 = None
        expand_29: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_13, [1, 16, 512, 512]);  where_13 = None
        view_144: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_29, [16, 512, 512]);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_138: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_175, [512, 1024])
        permute_70: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_38: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg109_1, view_138, permute_70);  arg109_1 = view_138 = permute_70 = None
        view_139: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_38, [1, 512, 1024]);  addmm_38 = None
        view_140: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_139, [1, 512, -1, 64]);  view_139 = None
        permute_71: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_187: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_71, torch.float32);  permute_71 = None
        expand_30: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_187, [1, 16, 512, 64]);  convert_element_type_187 = None
        view_145: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_30, [16, 512, 64]);  expand_30 = None
        bmm_13: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_144, view_145);  view_144 = view_145 = None
        view_146: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 16, 512, 64]);  bmm_13 = None
        convert_element_type_189: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_146, torch.float16);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_189, [0, 2, 1, 3]);  convert_element_type_189 = None
        clone_19: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_147: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_19, [1, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_147, [512, 1024]);  view_147 = None
        permute_74: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_39: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg111_1, view_148, permute_74);  arg111_1 = view_148 = permute_74 = None
        view_149: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_39, [1, 512, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_55: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_149, convert_element_type_175);  view_149 = convert_element_type_175 = None
        convert_element_type_193: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_193, [2], correction = 0, keepdim = True)
        getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_193, getitem_27);  convert_element_type_193 = getitem_27 = None
        add_56: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_58: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_59: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_58, arg112_1);  mul_58 = arg112_1 = None
        add_57: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_59, arg113_1);  mul_59 = arg113_1 = None
        convert_element_type_194: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_57, torch.float16);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_194, [512, 1024])
        permute_75: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
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
        permute_76: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_41: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg117_1, view_152, permute_76);  arg117_1 = view_152 = permute_76 = None
        view_153: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_41, [1, 512, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_153, convert_element_type_194);  view_153 = convert_element_type_194 = None
        convert_element_type_203: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_203, [2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_21: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_203, getitem_29);  convert_element_type_203 = getitem_29 = None
        add_60: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_63: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_64: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_63, arg118_1);  mul_63 = arg118_1 = None
        add_61: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_64, arg119_1);  mul_64 = arg119_1 = None
        convert_element_type_204: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_61, torch.float16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_154: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_204, [512, 1024])
        permute_77: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_42: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg121_1, view_154, permute_77);  arg121_1 = view_154 = permute_77 = None
        view_155: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_42, [1, 512, 1024]);  addmm_42 = None
        view_156: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_155, [1, 512, -1, 64]);  view_155 = None
        permute_78: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_214: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_78, torch.float32);  permute_78 = None
        mul_65: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_214, 0.3535533905932738);  convert_element_type_214 = None
        expand_31: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_65, [1, 16, 512, 64]);  mul_65 = None
        view_163: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_31, [16, 512, 64]);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_157: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_204, [512, 1024])
        permute_79: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_43: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg123_1, view_157, permute_79);  arg123_1 = view_157 = permute_79 = None
        view_158: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_43, [1, 512, 1024]);  addmm_43 = None
        view_159: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_158, [1, 512, -1, 64]);  view_158 = None
        permute_80: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_215: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_80, torch.float32);  permute_80 = None
        permute_83: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_215, [0, 1, 3, 2]);  convert_element_type_215 = None
        mul_66: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_83, 0.3535533905932738);  permute_83 = None
        expand_32: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_66, [1, 16, 64, 512]);  mul_66 = None
        view_164: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_32, [16, 64, 512]);  expand_32 = None
        bmm_14: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_163, view_164);  view_163 = view_164 = None
        view_165: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_14, [1, 16, 512, 512]);  bmm_14 = None
        full_default_22: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        add_62: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_165, where_14);  view_165 = where_14 = None
        eq_7: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_62, -inf)
        logical_not_14: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        full_default_23: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_7: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_62, [-1], True)
        sub_22: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_62, amax_7);  add_62 = amax_7 = None
        exp_7: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        where_15: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_15, full_default_23, div_7);  logical_not_15 = full_default_23 = div_7 = None
        expand_33: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_15, [1, 16, 512, 512]);  where_15 = None
        view_166: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_33, [16, 512, 512]);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_160: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_204, [512, 1024])
        permute_81: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_44: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg125_1, view_160, permute_81);  arg125_1 = view_160 = permute_81 = None
        view_161: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_44, [1, 512, 1024]);  addmm_44 = None
        view_162: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_161, [1, 512, -1, 64]);  view_161 = None
        permute_82: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_216: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_82, torch.float32);  permute_82 = None
        expand_34: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_216, [1, 16, 512, 64]);  convert_element_type_216 = None
        view_167: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_34, [16, 512, 64]);  expand_34 = None
        bmm_15: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
        view_168: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 16, 512, 64]);  bmm_15 = None
        convert_element_type_218: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_168, torch.float16);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_218, [0, 2, 1, 3]);  convert_element_type_218 = None
        clone_22: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_22, [1, 512, -1]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_169, [512, 1024]);  view_169 = None
        permute_85: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_45: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg127_1, view_170, permute_85);  arg127_1 = view_170 = permute_85 = None
        view_171: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_45, [1, 512, 1024]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_171, convert_element_type_204);  view_171 = convert_element_type_204 = None
        convert_element_type_222: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_222, [2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_222, getitem_31);  convert_element_type_222 = getitem_31 = None
        add_64: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_67: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_68: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_67, arg128_1);  mul_67 = arg128_1 = None
        add_65: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_68, arg129_1);  mul_68 = arg129_1 = None
        convert_element_type_223: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_65, torch.float16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_223, [512, 1024])
        permute_86: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
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
        permute_87: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_47: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg133_1, view_174, permute_87);  arg133_1 = view_174 = permute_87 = None
        view_175: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_47, [1, 512, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_175, convert_element_type_223);  view_175 = convert_element_type_223 = None
        convert_element_type_232: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_67, torch.float32);  add_67 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_232, [2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_24: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_232, getitem_33);  convert_element_type_232 = getitem_33 = None
        add_68: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_72: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_73: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_72, arg134_1);  mul_72 = arg134_1 = None
        add_69: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_73, arg135_1);  mul_73 = arg135_1 = None
        convert_element_type_233: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_69, torch.float16);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_176: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_233, [512, 1024])
        permute_88: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_48: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg137_1, view_176, permute_88);  arg137_1 = view_176 = permute_88 = None
        view_177: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_48, [1, 512, 1024]);  addmm_48 = None
        view_178: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_177, [1, 512, -1, 64]);  view_177 = None
        permute_89: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_243: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_89, torch.float32);  permute_89 = None
        mul_74: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_243, 0.3535533905932738);  convert_element_type_243 = None
        expand_35: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_74, [1, 16, 512, 64]);  mul_74 = None
        view_185: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_35, [16, 512, 64]);  expand_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_179: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_233, [512, 1024])
        permute_90: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_49: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg139_1, view_179, permute_90);  arg139_1 = view_179 = permute_90 = None
        view_180: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_49, [1, 512, 1024]);  addmm_49 = None
        view_181: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_180, [1, 512, -1, 64]);  view_180 = None
        permute_91: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_244: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_91, torch.float32);  permute_91 = None
        permute_94: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_244, [0, 1, 3, 2]);  convert_element_type_244 = None
        mul_75: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_94, 0.3535533905932738);  permute_94 = None
        expand_36: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_75, [1, 16, 64, 512]);  mul_75 = None
        view_186: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_36, [16, 64, 512]);  expand_36 = None
        bmm_16: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_185, view_186);  view_185 = view_186 = None
        view_187: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [1, 16, 512, 512]);  bmm_16 = None
        full_default_25: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_25, full_default_24);  full_default_25 = full_default_24 = None
        add_70: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_187, where_16);  view_187 = where_16 = None
        eq_8: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_70, -inf)
        logical_not_16: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        full_default_26: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_8: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_70, [-1], True)
        sub_25: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_70, amax_8);  add_70 = amax_8 = None
        exp_8: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        where_17: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_17, full_default_26, div_8);  logical_not_17 = full_default_26 = div_8 = None
        expand_37: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_17, [1, 16, 512, 512]);  where_17 = None
        view_188: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_37, [16, 512, 512]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_182: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_233, [512, 1024])
        permute_92: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_50: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg141_1, view_182, permute_92);  arg141_1 = view_182 = permute_92 = None
        view_183: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_50, [1, 512, 1024]);  addmm_50 = None
        view_184: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_183, [1, 512, -1, 64]);  view_183 = None
        permute_93: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_245: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_93, torch.float32);  permute_93 = None
        expand_38: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_245, [1, 16, 512, 64]);  convert_element_type_245 = None
        view_189: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_38, [16, 512, 64]);  expand_38 = None
        bmm_17: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_188, view_189);  view_188 = view_189 = None
        view_190: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 16, 512, 64]);  bmm_17 = None
        convert_element_type_247: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_190, torch.float16);  view_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_247, [0, 2, 1, 3]);  convert_element_type_247 = None
        clone_25: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_191: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_25, [1, 512, -1]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_191, [512, 1024]);  view_191 = None
        permute_96: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_51: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg143_1, view_192, permute_96);  arg143_1 = view_192 = permute_96 = None
        view_193: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_51, [1, 512, 1024]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_193, convert_element_type_233);  view_193 = convert_element_type_233 = None
        convert_element_type_251: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_251, [2], correction = 0, keepdim = True)
        getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_251, getitem_35);  convert_element_type_251 = getitem_35 = None
        add_72: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_76: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_77: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_76, arg144_1);  mul_76 = arg144_1 = None
        add_73: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_77, arg145_1);  mul_77 = arg145_1 = None
        convert_element_type_252: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_73, torch.float16);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_252, [512, 1024])
        permute_97: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
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
        permute_98: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_53: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg149_1, view_196, permute_98);  arg149_1 = view_196 = permute_98 = None
        view_197: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_53, [1, 512, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_75: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_197, convert_element_type_252);  view_197 = convert_element_type_252 = None
        convert_element_type_261: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_75, torch.float32);  add_75 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_261, [2], correction = 0, keepdim = True)
        getitem_36: "f32[1, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_27: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_261, getitem_37);  convert_element_type_261 = getitem_37 = None
        add_76: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_81: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_82: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_81, arg150_1);  mul_81 = arg150_1 = None
        add_77: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_82, arg151_1);  mul_82 = arg151_1 = None
        convert_element_type_262: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_77, torch.float16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_198: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_262, [512, 1024])
        permute_99: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_54: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg153_1, view_198, permute_99);  arg153_1 = view_198 = permute_99 = None
        view_199: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_54, [1, 512, 1024]);  addmm_54 = None
        view_200: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_199, [1, 512, -1, 64]);  view_199 = None
        permute_100: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_272: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_100, torch.float32);  permute_100 = None
        mul_83: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_272, 0.3535533905932738);  convert_element_type_272 = None
        expand_39: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_83, [1, 16, 512, 64]);  mul_83 = None
        view_207: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_39, [16, 512, 64]);  expand_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_201: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_262, [512, 1024])
        permute_101: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_55: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg155_1, view_201, permute_101);  arg155_1 = view_201 = permute_101 = None
        view_202: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_55, [1, 512, 1024]);  addmm_55 = None
        view_203: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_202, [1, 512, -1, 64]);  view_202 = None
        permute_102: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_273: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_102, torch.float32);  permute_102 = None
        permute_105: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_273, [0, 1, 3, 2]);  convert_element_type_273 = None
        mul_84: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_105, 0.3535533905932738);  permute_105 = None
        expand_40: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_84, [1, 16, 64, 512]);  mul_84 = None
        view_208: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_40, [16, 64, 512]);  expand_40 = None
        bmm_18: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_207, view_208);  view_207 = view_208 = None
        view_209: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_18, [1, 16, 512, 512]);  bmm_18 = None
        full_default_28: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_28, full_default_27);  full_default_28 = full_default_27 = None
        add_78: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_209, where_18);  view_209 = where_18 = None
        eq_9: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_78, -inf)
        logical_not_18: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        full_default_29: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_9: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
        sub_28: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_78, amax_9);  add_78 = amax_9 = None
        exp_9: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        where_19: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_19, full_default_29, div_9);  logical_not_19 = full_default_29 = div_9 = None
        expand_41: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_19, [1, 16, 512, 512]);  where_19 = None
        view_210: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_41, [16, 512, 512]);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_204: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_262, [512, 1024])
        permute_103: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_56: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg157_1, view_204, permute_103);  arg157_1 = view_204 = permute_103 = None
        view_205: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_56, [1, 512, 1024]);  addmm_56 = None
        view_206: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_205, [1, 512, -1, 64]);  view_205 = None
        permute_104: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_274: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_104, torch.float32);  permute_104 = None
        expand_42: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_274, [1, 16, 512, 64]);  convert_element_type_274 = None
        view_211: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_42, [16, 512, 64]);  expand_42 = None
        bmm_19: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_210, view_211);  view_210 = view_211 = None
        view_212: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 16, 512, 64]);  bmm_19 = None
        convert_element_type_276: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_212, torch.float16);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_276, [0, 2, 1, 3]);  convert_element_type_276 = None
        clone_28: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_213: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_28, [1, 512, -1]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_213, [512, 1024]);  view_213 = None
        permute_107: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_57: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg159_1, view_214, permute_107);  arg159_1 = view_214 = permute_107 = None
        view_215: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_57, [1, 512, 1024]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_79: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_215, convert_element_type_262);  view_215 = convert_element_type_262 = None
        convert_element_type_280: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32);  add_79 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_280, [2], correction = 0, keepdim = True)
        getitem_38: "f32[1, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_280, getitem_39);  convert_element_type_280 = getitem_39 = None
        add_80: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_85: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_86: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_85, arg160_1);  mul_85 = arg160_1 = None
        add_81: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_86, arg161_1);  mul_86 = arg161_1 = None
        convert_element_type_281: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_81, torch.float16);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_281, [512, 1024])
        permute_108: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
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
        permute_109: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_59: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg165_1, view_218, permute_109);  arg165_1 = view_218 = permute_109 = None
        view_219: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_59, [1, 512, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_219, convert_element_type_281);  view_219 = convert_element_type_281 = None
        convert_element_type_290: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32);  add_83 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_290, [2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_30: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_290, getitem_41);  convert_element_type_290 = getitem_41 = None
        add_84: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_90: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_91: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_90, arg166_1);  mul_90 = arg166_1 = None
        add_85: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_91, arg167_1);  mul_91 = arg167_1 = None
        convert_element_type_291: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_85, torch.float16);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_220: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_291, [512, 1024])
        permute_110: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_60: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg169_1, view_220, permute_110);  arg169_1 = view_220 = permute_110 = None
        view_221: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_60, [1, 512, 1024]);  addmm_60 = None
        view_222: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_221, [1, 512, -1, 64]);  view_221 = None
        permute_111: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_301: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_111, torch.float32);  permute_111 = None
        mul_92: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_301, 0.3535533905932738);  convert_element_type_301 = None
        expand_43: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_92, [1, 16, 512, 64]);  mul_92 = None
        view_229: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_43, [16, 512, 64]);  expand_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_223: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_291, [512, 1024])
        permute_112: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_61: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg171_1, view_223, permute_112);  arg171_1 = view_223 = permute_112 = None
        view_224: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_61, [1, 512, 1024]);  addmm_61 = None
        view_225: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_224, [1, 512, -1, 64]);  view_224 = None
        permute_113: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_302: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_113, torch.float32);  permute_113 = None
        permute_116: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_302, [0, 1, 3, 2]);  convert_element_type_302 = None
        mul_93: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_116, 0.3535533905932738);  permute_116 = None
        expand_44: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_93, [1, 16, 64, 512]);  mul_93 = None
        view_230: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_44, [16, 64, 512]);  expand_44 = None
        bmm_20: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_229, view_230);  view_229 = view_230 = None
        view_231: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [1, 16, 512, 512]);  bmm_20 = None
        full_default_31: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_31, full_default_30);  full_default_31 = full_default_30 = None
        add_86: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_231, where_20);  view_231 = where_20 = None
        eq_10: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_86, -inf)
        logical_not_20: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        full_default_32: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_10: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_86, [-1], True)
        sub_31: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_86, amax_10);  add_86 = amax_10 = None
        exp_10: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        where_21: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_21, full_default_32, div_10);  logical_not_21 = full_default_32 = div_10 = None
        expand_45: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_21, [1, 16, 512, 512]);  where_21 = None
        view_232: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_45, [16, 512, 512]);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_226: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_291, [512, 1024])
        permute_114: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_62: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg173_1, view_226, permute_114);  arg173_1 = view_226 = permute_114 = None
        view_227: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_62, [1, 512, 1024]);  addmm_62 = None
        view_228: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_227, [1, 512, -1, 64]);  view_227 = None
        permute_115: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_303: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_115, torch.float32);  permute_115 = None
        expand_46: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_303, [1, 16, 512, 64]);  convert_element_type_303 = None
        view_233: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_46, [16, 512, 64]);  expand_46 = None
        bmm_21: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_232, view_233);  view_232 = view_233 = None
        view_234: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 16, 512, 64]);  bmm_21 = None
        convert_element_type_305: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_234, torch.float16);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_305, [0, 2, 1, 3]);  convert_element_type_305 = None
        clone_31: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_31, [1, 512, -1]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_235, [512, 1024]);  view_235 = None
        permute_118: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        addmm_63: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg175_1, view_236, permute_118);  arg175_1 = view_236 = permute_118 = None
        view_237: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_63, [1, 512, 1024]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_237, convert_element_type_291);  view_237 = convert_element_type_291 = None
        convert_element_type_309: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_87, torch.float32);  add_87 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_309, [2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_309, getitem_43);  convert_element_type_309 = getitem_43 = None
        add_88: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_94: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_95: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_94, arg176_1);  mul_94 = arg176_1 = None
        add_89: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_95, arg177_1);  mul_95 = arg177_1 = None
        convert_element_type_310: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_89, torch.float16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_310, [512, 1024])
        permute_119: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
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
        permute_120: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_65: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg181_1, view_240, permute_120);  arg181_1 = view_240 = permute_120 = None
        view_241: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_65, [1, 512, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_91: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_241, convert_element_type_310);  view_241 = convert_element_type_310 = None
        convert_element_type_319: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_91, torch.float32);  add_91 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_319, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_33: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_319, getitem_45);  convert_element_type_319 = getitem_45 = None
        add_92: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_99: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_100: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_99, arg182_1);  mul_99 = arg182_1 = None
        add_93: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_100, arg183_1);  mul_100 = arg183_1 = None
        convert_element_type_320: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_93, torch.float16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_242: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_320, [512, 1024])
        permute_121: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_66: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg185_1, view_242, permute_121);  arg185_1 = view_242 = permute_121 = None
        view_243: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_66, [1, 512, 1024]);  addmm_66 = None
        view_244: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_243, [1, 512, -1, 64]);  view_243 = None
        permute_122: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_330: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_122, torch.float32);  permute_122 = None
        mul_101: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_330, 0.3535533905932738);  convert_element_type_330 = None
        expand_47: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_101, [1, 16, 512, 64]);  mul_101 = None
        view_251: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_47, [16, 512, 64]);  expand_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_245: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_320, [512, 1024])
        permute_123: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_67: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg187_1, view_245, permute_123);  arg187_1 = view_245 = permute_123 = None
        view_246: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_67, [1, 512, 1024]);  addmm_67 = None
        view_247: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_246, [1, 512, -1, 64]);  view_246 = None
        permute_124: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_331: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_124, torch.float32);  permute_124 = None
        permute_127: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_331, [0, 1, 3, 2]);  convert_element_type_331 = None
        mul_102: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_127, 0.3535533905932738);  permute_127 = None
        expand_48: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_102, [1, 16, 64, 512]);  mul_102 = None
        view_252: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_48, [16, 64, 512]);  expand_48 = None
        bmm_22: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_251, view_252);  view_251 = view_252 = None
        view_253: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [1, 16, 512, 512]);  bmm_22 = None
        full_default_34: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_34, full_default_33);  full_default_34 = full_default_33 = None
        add_94: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_253, where_22);  view_253 = where_22 = None
        eq_11: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_94, -inf)
        logical_not_22: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        full_default_35: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_11: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_94, [-1], True)
        sub_34: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_94, amax_11);  add_94 = amax_11 = None
        exp_11: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        where_23: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_23, full_default_35, div_11);  logical_not_23 = full_default_35 = div_11 = None
        expand_49: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_23, [1, 16, 512, 512]);  where_23 = None
        view_254: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_49, [16, 512, 512]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_248: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_320, [512, 1024])
        permute_125: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_68: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg189_1, view_248, permute_125);  arg189_1 = view_248 = permute_125 = None
        view_249: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_68, [1, 512, 1024]);  addmm_68 = None
        view_250: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_249, [1, 512, -1, 64]);  view_249 = None
        permute_126: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_332: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_126, torch.float32);  permute_126 = None
        expand_50: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_332, [1, 16, 512, 64]);  convert_element_type_332 = None
        view_255: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_50, [16, 512, 64]);  expand_50 = None
        bmm_23: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_254, view_255);  view_254 = view_255 = None
        view_256: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 16, 512, 64]);  bmm_23 = None
        convert_element_type_334: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_256, torch.float16);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_334, [0, 2, 1, 3]);  convert_element_type_334 = None
        clone_34: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_257: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_34, [1, 512, -1]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_258: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_257, [512, 1024]);  view_257 = None
        permute_129: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_69: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg191_1, view_258, permute_129);  arg191_1 = view_258 = permute_129 = None
        view_259: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_69, [1, 512, 1024]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_95: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_259, convert_element_type_320);  view_259 = convert_element_type_320 = None
        convert_element_type_338: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_95, torch.float32);  add_95 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_338, [2], correction = 0, keepdim = True)
        getitem_46: "f32[1, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_338, getitem_47);  convert_element_type_338 = getitem_47 = None
        add_96: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_103: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_104: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_103, arg192_1);  mul_103 = arg192_1 = None
        add_97: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_104, arg193_1);  mul_104 = arg193_1 = None
        convert_element_type_339: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_97, torch.float16);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_339, [512, 1024])
        permute_130: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
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
        permute_131: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        addmm_71: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg197_1, view_262, permute_131);  arg197_1 = view_262 = permute_131 = None
        view_263: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_71, [1, 512, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_99: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_263, convert_element_type_339);  view_263 = convert_element_type_339 = None
        convert_element_type_348: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_99, torch.float32);  add_99 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_348, [2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_36: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_348, getitem_49);  convert_element_type_348 = getitem_49 = None
        add_100: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_108: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_109: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_108, arg198_1);  mul_108 = arg198_1 = None
        add_101: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_109, arg199_1);  mul_109 = arg199_1 = None
        convert_element_type_349: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_101, torch.float16);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_264: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_349, [512, 1024])
        permute_132: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_72: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg201_1, view_264, permute_132);  arg201_1 = view_264 = permute_132 = None
        view_265: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_72, [1, 512, 1024]);  addmm_72 = None
        view_266: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_265, [1, 512, -1, 64]);  view_265 = None
        permute_133: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_359: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_133, torch.float32);  permute_133 = None
        mul_110: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_359, 0.3535533905932738);  convert_element_type_359 = None
        expand_51: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_110, [1, 16, 512, 64]);  mul_110 = None
        view_273: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_51, [16, 512, 64]);  expand_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_267: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_349, [512, 1024])
        permute_134: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_73: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg203_1, view_267, permute_134);  arg203_1 = view_267 = permute_134 = None
        view_268: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_73, [1, 512, 1024]);  addmm_73 = None
        view_269: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_268, [1, 512, -1, 64]);  view_268 = None
        permute_135: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_360: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_135, torch.float32);  permute_135 = None
        permute_138: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_360, [0, 1, 3, 2]);  convert_element_type_360 = None
        mul_111: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_138, 0.3535533905932738);  permute_138 = None
        expand_52: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_111, [1, 16, 64, 512]);  mul_111 = None
        view_274: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_52, [16, 64, 512]);  expand_52 = None
        bmm_24: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_273, view_274);  view_273 = view_274 = None
        view_275: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_24, [1, 16, 512, 512]);  bmm_24 = None
        full_default_37: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_37, full_default_36);  full_default_37 = full_default_36 = None
        add_102: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_275, where_24);  view_275 = where_24 = None
        eq_12: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_102, -inf)
        logical_not_24: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_12);  eq_12 = None
        any_13: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_24, -1, True);  logical_not_24 = None
        logical_not_25: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_13);  any_13 = None
        full_default_38: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_12: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_102, [-1], True)
        sub_37: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_102, amax_12);  add_102 = amax_12 = None
        exp_12: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_13: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        where_25: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_25, full_default_38, div_12);  logical_not_25 = full_default_38 = div_12 = None
        expand_53: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_25, [1, 16, 512, 512]);  where_25 = None
        view_276: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_53, [16, 512, 512]);  expand_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_270: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_349, [512, 1024])
        permute_136: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        addmm_74: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg205_1, view_270, permute_136);  arg205_1 = view_270 = permute_136 = None
        view_271: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_74, [1, 512, 1024]);  addmm_74 = None
        view_272: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_271, [1, 512, -1, 64]);  view_271 = None
        permute_137: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_361: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_137, torch.float32);  permute_137 = None
        expand_54: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_361, [1, 16, 512, 64]);  convert_element_type_361 = None
        view_277: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_54, [16, 512, 64]);  expand_54 = None
        bmm_25: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_276, view_277);  view_276 = view_277 = None
        view_278: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_25, [1, 16, 512, 64]);  bmm_25 = None
        convert_element_type_363: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_278, torch.float16);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_139: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_363, [0, 2, 1, 3]);  convert_element_type_363 = None
        clone_37: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_279: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_37, [1, 512, -1]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_280: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_279, [512, 1024]);  view_279 = None
        permute_140: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_75: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg207_1, view_280, permute_140);  arg207_1 = view_280 = permute_140 = None
        view_281: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_75, [1, 512, 1024]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_103: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_281, convert_element_type_349);  view_281 = convert_element_type_349 = None
        convert_element_type_367: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_103, torch.float32);  add_103 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_367, [2], correction = 0, keepdim = True)
        getitem_50: "f32[1, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_38: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_367, getitem_51);  convert_element_type_367 = getitem_51 = None
        add_104: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_112: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
        mul_113: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_112, arg208_1);  mul_112 = arg208_1 = None
        add_105: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_113, arg209_1);  mul_113 = arg209_1 = None
        convert_element_type_368: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_105, torch.float16);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_282: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_368, [512, 1024])
        permute_141: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_76: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg211_1, view_282, permute_141);  arg211_1 = view_282 = permute_141 = None
        view_283: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_76, [1, 512, 3072]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_372: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        mul_114: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_372, 0.5)
        mul_115: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_372, 0.7071067811865476);  convert_element_type_372 = None
        erf_12: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_115);  mul_115 = None
        add_106: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_116: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_114, add_106);  mul_114 = add_106 = None
        convert_element_type_373: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_116, torch.float16);  mul_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_284: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_373, [512, 3072]);  convert_element_type_373 = None
        permute_142: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        addmm_77: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg213_1, view_284, permute_142);  arg213_1 = view_284 = permute_142 = None
        view_285: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_77, [1, 512, 1024]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_107: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_285, convert_element_type_368);  view_285 = convert_element_type_368 = None
        convert_element_type_377: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32);  add_107 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_377, [2], correction = 0, keepdim = True)
        getitem_52: "f32[1, 512, 1]" = var_mean_26[0]
        getitem_53: "f32[1, 512, 1]" = var_mean_26[1];  var_mean_26 = None
        sub_39: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_377, getitem_53);  convert_element_type_377 = getitem_53 = None
        add_108: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_26: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_117: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
        mul_118: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_117, arg214_1);  mul_117 = arg214_1 = None
        add_109: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_118, arg215_1);  mul_118 = arg215_1 = None
        convert_element_type_378: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_109, torch.float16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_286: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_378, [512, 1024])
        permute_143: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_78: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg217_1, view_286, permute_143);  arg217_1 = view_286 = permute_143 = None
        view_287: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_78, [1, 512, 1024]);  addmm_78 = None
        view_288: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_287, [1, 512, -1, 64]);  view_287 = None
        permute_144: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_388: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_144, torch.float32);  permute_144 = None
        mul_119: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_388, 0.3535533905932738);  convert_element_type_388 = None
        expand_55: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_119, [1, 16, 512, 64]);  mul_119 = None
        view_295: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_55, [16, 512, 64]);  expand_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_289: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_378, [512, 1024])
        permute_145: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_79: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg219_1, view_289, permute_145);  arg219_1 = view_289 = permute_145 = None
        view_290: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_79, [1, 512, 1024]);  addmm_79 = None
        view_291: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_290, [1, 512, -1, 64]);  view_290 = None
        permute_146: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_389: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_146, torch.float32);  permute_146 = None
        permute_149: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_389, [0, 1, 3, 2]);  convert_element_type_389 = None
        mul_120: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_149, 0.3535533905932738);  permute_149 = None
        expand_56: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_120, [1, 16, 64, 512]);  mul_120 = None
        view_296: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_56, [16, 64, 512]);  expand_56 = None
        bmm_26: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_295, view_296);  view_295 = view_296 = None
        view_297: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_26, [1, 16, 512, 512]);  bmm_26 = None
        full_default_40: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_40, full_default_39);  full_default_40 = full_default_39 = None
        add_110: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_297, where_26);  view_297 = where_26 = None
        eq_13: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_110, -inf)
        logical_not_26: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_13);  eq_13 = None
        any_14: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_26, -1, True);  logical_not_26 = None
        logical_not_27: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_14);  any_14 = None
        full_default_41: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_13: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_110, [-1], True)
        sub_40: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_110, amax_13);  add_110 = amax_13 = None
        exp_13: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_14: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        where_27: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_27, full_default_41, div_13);  logical_not_27 = full_default_41 = div_13 = None
        expand_57: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_27, [1, 16, 512, 512]);  where_27 = None
        view_298: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_57, [16, 512, 512]);  expand_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_292: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_378, [512, 1024])
        permute_147: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_80: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg221_1, view_292, permute_147);  arg221_1 = view_292 = permute_147 = None
        view_293: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_80, [1, 512, 1024]);  addmm_80 = None
        view_294: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_293, [1, 512, -1, 64]);  view_293 = None
        permute_148: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_390: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_148, torch.float32);  permute_148 = None
        expand_58: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_390, [1, 16, 512, 64]);  convert_element_type_390 = None
        view_299: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_58, [16, 512, 64]);  expand_58 = None
        bmm_27: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_298, view_299);  view_298 = view_299 = None
        view_300: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_27, [1, 16, 512, 64]);  bmm_27 = None
        convert_element_type_392: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_300, torch.float16);  view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_150: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_392, [0, 2, 1, 3]);  convert_element_type_392 = None
        clone_40: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_301: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_40, [1, 512, -1]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_302: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_301, [512, 1024]);  view_301 = None
        permute_151: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        addmm_81: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg223_1, view_302, permute_151);  arg223_1 = view_302 = permute_151 = None
        view_303: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_81, [1, 512, 1024]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_111: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_303, convert_element_type_378);  view_303 = convert_element_type_378 = None
        convert_element_type_396: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_111, torch.float32);  add_111 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_396, [2], correction = 0, keepdim = True)
        getitem_54: "f32[1, 512, 1]" = var_mean_27[0]
        getitem_55: "f32[1, 512, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_41: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_396, getitem_55);  convert_element_type_396 = getitem_55 = None
        add_112: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_27: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_121: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
        mul_122: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_121, arg224_1);  mul_121 = arg224_1 = None
        add_113: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_122, arg225_1);  mul_122 = arg225_1 = None
        convert_element_type_397: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_113, torch.float16);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_304: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_397, [512, 1024])
        permute_152: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_82: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg227_1, view_304, permute_152);  arg227_1 = view_304 = permute_152 = None
        view_305: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_82, [1, 512, 3072]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_401: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_305, torch.float32);  view_305 = None
        mul_123: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_401, 0.5)
        mul_124: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_401, 0.7071067811865476);  convert_element_type_401 = None
        erf_13: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_124);  mul_124 = None
        add_114: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_125: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_123, add_114);  mul_123 = add_114 = None
        convert_element_type_402: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_125, torch.float16);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_306: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_402, [512, 3072]);  convert_element_type_402 = None
        permute_153: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        addmm_83: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg229_1, view_306, permute_153);  arg229_1 = view_306 = permute_153 = None
        view_307: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_83, [1, 512, 1024]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_115: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_307, convert_element_type_397);  view_307 = convert_element_type_397 = None
        convert_element_type_406: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_115, torch.float32);  add_115 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_406, [2], correction = 0, keepdim = True)
        getitem_56: "f32[1, 512, 1]" = var_mean_28[0]
        getitem_57: "f32[1, 512, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_42: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_406, getitem_57);  convert_element_type_406 = getitem_57 = None
        add_116: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_28: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_126: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
        mul_127: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_126, arg230_1);  mul_126 = arg230_1 = None
        add_117: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_127, arg231_1);  mul_127 = arg231_1 = None
        convert_element_type_407: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_117, torch.float16);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_308: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_407, [512, 1024])
        permute_154: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_84: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg233_1, view_308, permute_154);  arg233_1 = view_308 = permute_154 = None
        view_309: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_84, [1, 512, 1024]);  addmm_84 = None
        view_310: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_309, [1, 512, -1, 64]);  view_309 = None
        permute_155: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_417: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_155, torch.float32);  permute_155 = None
        mul_128: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_417, 0.3535533905932738);  convert_element_type_417 = None
        expand_59: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_128, [1, 16, 512, 64]);  mul_128 = None
        view_317: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_59, [16, 512, 64]);  expand_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_311: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_407, [512, 1024])
        permute_156: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        addmm_85: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg235_1, view_311, permute_156);  arg235_1 = view_311 = permute_156 = None
        view_312: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_85, [1, 512, 1024]);  addmm_85 = None
        view_313: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_312, [1, 512, -1, 64]);  view_312 = None
        permute_157: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_418: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_157, torch.float32);  permute_157 = None
        permute_160: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_418, [0, 1, 3, 2]);  convert_element_type_418 = None
        mul_129: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_160, 0.3535533905932738);  permute_160 = None
        expand_60: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_129, [1, 16, 64, 512]);  mul_129 = None
        view_318: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_60, [16, 64, 512]);  expand_60 = None
        bmm_28: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_317, view_318);  view_317 = view_318 = None
        view_319: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_28, [1, 16, 512, 512]);  bmm_28 = None
        full_default_43: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_43, full_default_42);  full_default_43 = full_default_42 = None
        add_118: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_319, where_28);  view_319 = where_28 = None
        eq_14: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_118, -inf)
        logical_not_28: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_14);  eq_14 = None
        any_15: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_28, -1, True);  logical_not_28 = None
        logical_not_29: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_15);  any_15 = None
        full_default_44: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_14: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_118, [-1], True)
        sub_43: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_118, amax_14);  add_118 = amax_14 = None
        exp_14: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_15: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        where_29: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_29, full_default_44, div_14);  logical_not_29 = full_default_44 = div_14 = None
        expand_61: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_29, [1, 16, 512, 512]);  where_29 = None
        view_320: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_61, [16, 512, 512]);  expand_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_314: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_407, [512, 1024])
        permute_158: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        addmm_86: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg237_1, view_314, permute_158);  arg237_1 = view_314 = permute_158 = None
        view_315: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_86, [1, 512, 1024]);  addmm_86 = None
        view_316: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_315, [1, 512, -1, 64]);  view_315 = None
        permute_159: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_419: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_159, torch.float32);  permute_159 = None
        expand_62: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_419, [1, 16, 512, 64]);  convert_element_type_419 = None
        view_321: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_62, [16, 512, 64]);  expand_62 = None
        bmm_29: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_320, view_321);  view_320 = view_321 = None
        view_322: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_29, [1, 16, 512, 64]);  bmm_29 = None
        convert_element_type_421: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_322, torch.float16);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_161: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_421, [0, 2, 1, 3]);  convert_element_type_421 = None
        clone_43: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_323: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_43, [1, 512, -1]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_324: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_323, [512, 1024]);  view_323 = None
        permute_162: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        addmm_87: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg239_1, view_324, permute_162);  arg239_1 = view_324 = permute_162 = None
        view_325: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_87, [1, 512, 1024]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_119: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_325, convert_element_type_407);  view_325 = convert_element_type_407 = None
        convert_element_type_425: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_119, torch.float32);  add_119 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_425, [2], correction = 0, keepdim = True)
        getitem_58: "f32[1, 512, 1]" = var_mean_29[0]
        getitem_59: "f32[1, 512, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_44: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_425, getitem_59);  convert_element_type_425 = getitem_59 = None
        add_120: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_29: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_130: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
        mul_131: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_130, arg240_1);  mul_130 = arg240_1 = None
        add_121: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_131, arg241_1);  mul_131 = arg241_1 = None
        convert_element_type_426: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_121, torch.float16);  add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_326: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_426, [512, 1024])
        permute_163: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        addmm_88: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg243_1, view_326, permute_163);  arg243_1 = view_326 = permute_163 = None
        view_327: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_88, [1, 512, 3072]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_430: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_327, torch.float32);  view_327 = None
        mul_132: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_430, 0.5)
        mul_133: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_430, 0.7071067811865476);  convert_element_type_430 = None
        erf_14: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_122: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_134: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_132, add_122);  mul_132 = add_122 = None
        convert_element_type_431: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_134, torch.float16);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_328: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_431, [512, 3072]);  convert_element_type_431 = None
        permute_164: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        addmm_89: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg245_1, view_328, permute_164);  arg245_1 = view_328 = permute_164 = None
        view_329: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_89, [1, 512, 1024]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_123: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_329, convert_element_type_426);  view_329 = convert_element_type_426 = None
        convert_element_type_435: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_123, torch.float32);  add_123 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_435, [2], correction = 0, keepdim = True)
        getitem_60: "f32[1, 512, 1]" = var_mean_30[0]
        getitem_61: "f32[1, 512, 1]" = var_mean_30[1];  var_mean_30 = None
        sub_45: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_435, getitem_61);  convert_element_type_435 = getitem_61 = None
        add_124: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_30: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_135: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
        mul_136: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_135, arg246_1);  mul_135 = arg246_1 = None
        add_125: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_136, arg247_1);  mul_136 = arg247_1 = None
        convert_element_type_436: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_125, torch.float16);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_330: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_436, [512, 1024])
        permute_165: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        addmm_90: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg249_1, view_330, permute_165);  arg249_1 = view_330 = permute_165 = None
        view_331: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_90, [1, 512, 1024]);  addmm_90 = None
        view_332: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_331, [1, 512, -1, 64]);  view_331 = None
        permute_166: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_446: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_166, torch.float32);  permute_166 = None
        mul_137: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_446, 0.3535533905932738);  convert_element_type_446 = None
        expand_63: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_137, [1, 16, 512, 64]);  mul_137 = None
        view_339: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_63, [16, 512, 64]);  expand_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_333: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_436, [512, 1024])
        permute_167: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_91: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg251_1, view_333, permute_167);  arg251_1 = view_333 = permute_167 = None
        view_334: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_91, [1, 512, 1024]);  addmm_91 = None
        view_335: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_334, [1, 512, -1, 64]);  view_334 = None
        permute_168: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_447: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_168, torch.float32);  permute_168 = None
        permute_171: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_447, [0, 1, 3, 2]);  convert_element_type_447 = None
        mul_138: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_171, 0.3535533905932738);  permute_171 = None
        expand_64: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_138, [1, 16, 64, 512]);  mul_138 = None
        view_340: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_64, [16, 64, 512]);  expand_64 = None
        bmm_30: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_339, view_340);  view_339 = view_340 = None
        view_341: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_30, [1, 16, 512, 512]);  bmm_30 = None
        full_default_46: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_30: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_46, full_default_45);  full_default_46 = full_default_45 = None
        add_126: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_341, where_30);  view_341 = where_30 = None
        eq_15: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_126, -inf)
        logical_not_30: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_15);  eq_15 = None
        any_16: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_30, -1, True);  logical_not_30 = None
        logical_not_31: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_16);  any_16 = None
        full_default_47: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_15: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_126, [-1], True)
        sub_46: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_126, amax_15);  add_126 = amax_15 = None
        exp_15: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_16: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        where_31: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_31, full_default_47, div_15);  logical_not_31 = full_default_47 = div_15 = None
        expand_65: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_31, [1, 16, 512, 512]);  where_31 = None
        view_342: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_65, [16, 512, 512]);  expand_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_336: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_436, [512, 1024])
        permute_169: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        addmm_92: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg253_1, view_336, permute_169);  arg253_1 = view_336 = permute_169 = None
        view_337: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_92, [1, 512, 1024]);  addmm_92 = None
        view_338: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_337, [1, 512, -1, 64]);  view_337 = None
        permute_170: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_448: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_170, torch.float32);  permute_170 = None
        expand_66: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_448, [1, 16, 512, 64]);  convert_element_type_448 = None
        view_343: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_66, [16, 512, 64]);  expand_66 = None
        bmm_31: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_342, view_343);  view_342 = view_343 = None
        view_344: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_31, [1, 16, 512, 64]);  bmm_31 = None
        convert_element_type_450: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_344, torch.float16);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_172: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_450, [0, 2, 1, 3]);  convert_element_type_450 = None
        clone_46: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_345: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_46, [1, 512, -1]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_346: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_345, [512, 1024]);  view_345 = None
        permute_173: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        addmm_93: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg255_1, view_346, permute_173);  arg255_1 = view_346 = permute_173 = None
        view_347: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_93, [1, 512, 1024]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_127: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_347, convert_element_type_436);  view_347 = convert_element_type_436 = None
        convert_element_type_454: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_127, torch.float32);  add_127 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_454, [2], correction = 0, keepdim = True)
        getitem_62: "f32[1, 512, 1]" = var_mean_31[0]
        getitem_63: "f32[1, 512, 1]" = var_mean_31[1];  var_mean_31 = None
        sub_47: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_454, getitem_63);  convert_element_type_454 = getitem_63 = None
        add_128: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_31: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_139: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
        mul_140: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_139, arg256_1);  mul_139 = arg256_1 = None
        add_129: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_140, arg257_1);  mul_140 = arg257_1 = None
        convert_element_type_455: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_129, torch.float16);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_348: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_455, [512, 1024])
        permute_174: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        addmm_94: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg259_1, view_348, permute_174);  arg259_1 = view_348 = permute_174 = None
        view_349: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_94, [1, 512, 3072]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_459: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        mul_141: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_459, 0.5)
        mul_142: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_459, 0.7071067811865476);  convert_element_type_459 = None
        erf_15: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_130: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_143: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_141, add_130);  mul_141 = add_130 = None
        convert_element_type_460: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_143, torch.float16);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_350: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_460, [512, 3072]);  convert_element_type_460 = None
        permute_175: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        addmm_95: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg261_1, view_350, permute_175);  arg261_1 = view_350 = permute_175 = None
        view_351: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_95, [1, 512, 1024]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_131: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_351, convert_element_type_455);  view_351 = convert_element_type_455 = None
        convert_element_type_464: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_131, torch.float32);  add_131 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_464, [2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 512, 1]" = var_mean_32[0]
        getitem_65: "f32[1, 512, 1]" = var_mean_32[1];  var_mean_32 = None
        sub_48: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_464, getitem_65);  convert_element_type_464 = getitem_65 = None
        add_132: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_32: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        mul_144: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
        mul_145: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_144, arg262_1);  mul_144 = arg262_1 = None
        add_133: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_145, arg263_1);  mul_145 = arg263_1 = None
        convert_element_type_465: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_133, torch.float16);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_352: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_465, [512, 1024])
        permute_176: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_96: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg265_1, view_352, permute_176);  arg265_1 = view_352 = permute_176 = None
        view_353: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_96, [1, 512, 1024]);  addmm_96 = None
        view_354: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_353, [1, 512, -1, 64]);  view_353 = None
        permute_177: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_475: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_177, torch.float32);  permute_177 = None
        mul_146: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_475, 0.3535533905932738);  convert_element_type_475 = None
        expand_67: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_146, [1, 16, 512, 64]);  mul_146 = None
        view_361: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_67, [16, 512, 64]);  expand_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_355: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_465, [512, 1024])
        permute_178: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_97: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg267_1, view_355, permute_178);  arg267_1 = view_355 = permute_178 = None
        view_356: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_97, [1, 512, 1024]);  addmm_97 = None
        view_357: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_356, [1, 512, -1, 64]);  view_356 = None
        permute_179: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_476: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_179, torch.float32);  permute_179 = None
        permute_182: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_476, [0, 1, 3, 2]);  convert_element_type_476 = None
        mul_147: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_182, 0.3535533905932738);  permute_182 = None
        expand_68: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_147, [1, 16, 64, 512]);  mul_147 = None
        view_362: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_68, [16, 64, 512]);  expand_68 = None
        bmm_32: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_361, view_362);  view_361 = view_362 = None
        view_363: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_32, [1, 16, 512, 512]);  bmm_32 = None
        full_default_49: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_49, full_default_48);  full_default_49 = full_default_48 = None
        add_134: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_363, where_32);  view_363 = where_32 = None
        eq_16: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_134, -inf)
        logical_not_32: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_16);  eq_16 = None
        any_17: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_32, -1, True);  logical_not_32 = None
        logical_not_33: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_17);  any_17 = None
        full_default_50: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_16: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_134, [-1], True)
        sub_49: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_134, amax_16);  add_134 = amax_16 = None
        exp_16: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_17: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        where_33: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_33, full_default_50, div_16);  logical_not_33 = full_default_50 = div_16 = None
        expand_69: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_33, [1, 16, 512, 512]);  where_33 = None
        view_364: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_69, [16, 512, 512]);  expand_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_358: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_465, [512, 1024])
        permute_180: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_98: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg269_1, view_358, permute_180);  arg269_1 = view_358 = permute_180 = None
        view_359: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_98, [1, 512, 1024]);  addmm_98 = None
        view_360: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_359, [1, 512, -1, 64]);  view_359 = None
        permute_181: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_477: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_181, torch.float32);  permute_181 = None
        expand_70: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_477, [1, 16, 512, 64]);  convert_element_type_477 = None
        view_365: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_70, [16, 512, 64]);  expand_70 = None
        bmm_33: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_364, view_365);  view_364 = view_365 = None
        view_366: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_33, [1, 16, 512, 64]);  bmm_33 = None
        convert_element_type_479: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_366, torch.float16);  view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_183: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_479, [0, 2, 1, 3]);  convert_element_type_479 = None
        clone_49: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_367: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_49, [1, 512, -1]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_368: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_367, [512, 1024]);  view_367 = None
        permute_184: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        addmm_99: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg271_1, view_368, permute_184);  arg271_1 = view_368 = permute_184 = None
        view_369: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_99, [1, 512, 1024]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_135: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_369, convert_element_type_465);  view_369 = convert_element_type_465 = None
        convert_element_type_483: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_135, torch.float32);  add_135 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_483, [2], correction = 0, keepdim = True)
        getitem_66: "f32[1, 512, 1]" = var_mean_33[0]
        getitem_67: "f32[1, 512, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_50: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_483, getitem_67);  convert_element_type_483 = getitem_67 = None
        add_136: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_33: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_148: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
        mul_149: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_148, arg272_1);  mul_148 = arg272_1 = None
        add_137: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_149, arg273_1);  mul_149 = arg273_1 = None
        convert_element_type_484: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_137, torch.float16);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_370: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_484, [512, 1024])
        permute_185: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_100: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg275_1, view_370, permute_185);  arg275_1 = view_370 = permute_185 = None
        view_371: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_100, [1, 512, 3072]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_488: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_371, torch.float32);  view_371 = None
        mul_150: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_488, 0.5)
        mul_151: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_488, 0.7071067811865476);  convert_element_type_488 = None
        erf_16: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_138: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_152: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_150, add_138);  mul_150 = add_138 = None
        convert_element_type_489: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_152, torch.float16);  mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_372: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_489, [512, 3072]);  convert_element_type_489 = None
        permute_186: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
        addmm_101: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg277_1, view_372, permute_186);  arg277_1 = view_372 = permute_186 = None
        view_373: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_101, [1, 512, 1024]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_139: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_373, convert_element_type_484);  view_373 = convert_element_type_484 = None
        convert_element_type_493: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_139, torch.float32);  add_139 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_493, [2], correction = 0, keepdim = True)
        getitem_68: "f32[1, 512, 1]" = var_mean_34[0]
        getitem_69: "f32[1, 512, 1]" = var_mean_34[1];  var_mean_34 = None
        sub_51: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_493, getitem_69);  convert_element_type_493 = getitem_69 = None
        add_140: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
        rsqrt_34: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_153: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = rsqrt_34 = None
        mul_154: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_153, arg278_1);  mul_153 = arg278_1 = None
        add_141: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_154, arg279_1);  mul_154 = arg279_1 = None
        convert_element_type_494: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_141, torch.float16);  add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_374: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_494, [512, 1024])
        permute_187: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        addmm_102: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg281_1, view_374, permute_187);  arg281_1 = view_374 = permute_187 = None
        view_375: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_102, [1, 512, 1024]);  addmm_102 = None
        view_376: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_375, [1, 512, -1, 64]);  view_375 = None
        permute_188: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_504: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_188, torch.float32);  permute_188 = None
        mul_155: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_504, 0.3535533905932738);  convert_element_type_504 = None
        expand_71: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_155, [1, 16, 512, 64]);  mul_155 = None
        view_383: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_71, [16, 512, 64]);  expand_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_377: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_494, [512, 1024])
        permute_189: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        addmm_103: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg283_1, view_377, permute_189);  arg283_1 = view_377 = permute_189 = None
        view_378: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_103, [1, 512, 1024]);  addmm_103 = None
        view_379: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_378, [1, 512, -1, 64]);  view_378 = None
        permute_190: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_505: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_190, torch.float32);  permute_190 = None
        permute_193: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_505, [0, 1, 3, 2]);  convert_element_type_505 = None
        mul_156: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_193, 0.3535533905932738);  permute_193 = None
        expand_72: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_156, [1, 16, 64, 512]);  mul_156 = None
        view_384: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_72, [16, 64, 512]);  expand_72 = None
        bmm_34: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_383, view_384);  view_383 = view_384 = None
        view_385: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_34, [1, 16, 512, 512]);  bmm_34 = None
        full_default_52: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_34: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_52, full_default_51);  full_default_52 = full_default_51 = None
        add_142: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_385, where_34);  view_385 = where_34 = None
        eq_17: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_142, -inf)
        logical_not_34: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_17);  eq_17 = None
        any_18: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_34, -1, True);  logical_not_34 = None
        logical_not_35: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_18);  any_18 = None
        full_default_53: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_17: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_142, [-1], True)
        sub_52: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_142, amax_17);  add_142 = amax_17 = None
        exp_17: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_18: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        where_35: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_35, full_default_53, div_17);  logical_not_35 = full_default_53 = div_17 = None
        expand_73: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_35, [1, 16, 512, 512]);  where_35 = None
        view_386: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_73, [16, 512, 512]);  expand_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_380: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_494, [512, 1024])
        permute_191: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_104: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg285_1, view_380, permute_191);  arg285_1 = view_380 = permute_191 = None
        view_381: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_104, [1, 512, 1024]);  addmm_104 = None
        view_382: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_381, [1, 512, -1, 64]);  view_381 = None
        permute_192: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_506: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_192, torch.float32);  permute_192 = None
        expand_74: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_506, [1, 16, 512, 64]);  convert_element_type_506 = None
        view_387: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_74, [16, 512, 64]);  expand_74 = None
        bmm_35: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_386, view_387);  view_386 = view_387 = None
        view_388: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_35, [1, 16, 512, 64]);  bmm_35 = None
        convert_element_type_508: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_388, torch.float16);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_194: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_508, [0, 2, 1, 3]);  convert_element_type_508 = None
        clone_52: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_389: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_52, [1, 512, -1]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_390: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_389, [512, 1024]);  view_389 = None
        permute_195: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        addmm_105: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg287_1, view_390, permute_195);  arg287_1 = view_390 = permute_195 = None
        view_391: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_105, [1, 512, 1024]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_143: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_391, convert_element_type_494);  view_391 = convert_element_type_494 = None
        convert_element_type_512: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_143, torch.float32);  add_143 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_512, [2], correction = 0, keepdim = True)
        getitem_70: "f32[1, 512, 1]" = var_mean_35[0]
        getitem_71: "f32[1, 512, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_53: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_512, getitem_71);  convert_element_type_512 = getitem_71 = None
        add_144: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_35: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_157: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
        mul_158: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_157, arg288_1);  mul_157 = arg288_1 = None
        add_145: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_158, arg289_1);  mul_158 = arg289_1 = None
        convert_element_type_513: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_145, torch.float16);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_513, [512, 1024])
        permute_196: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        addmm_106: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg291_1, view_392, permute_196);  arg291_1 = view_392 = permute_196 = None
        view_393: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_106, [1, 512, 3072]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_517: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        mul_159: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_517, 0.5)
        mul_160: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_517, 0.7071067811865476);  convert_element_type_517 = None
        erf_17: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
        add_146: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_161: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_159, add_146);  mul_159 = add_146 = None
        convert_element_type_518: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_161, torch.float16);  mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_394: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_518, [512, 3072]);  convert_element_type_518 = None
        permute_197: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
        addmm_107: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg293_1, view_394, permute_197);  arg293_1 = view_394 = permute_197 = None
        view_395: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_107, [1, 512, 1024]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_147: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_395, convert_element_type_513);  view_395 = convert_element_type_513 = None
        convert_element_type_522: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_147, torch.float32);  add_147 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_522, [2], correction = 0, keepdim = True)
        getitem_72: "f32[1, 512, 1]" = var_mean_36[0]
        getitem_73: "f32[1, 512, 1]" = var_mean_36[1];  var_mean_36 = None
        sub_54: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_522, getitem_73);  convert_element_type_522 = getitem_73 = None
        add_148: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_36: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_162: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = rsqrt_36 = None
        mul_163: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_162, arg294_1);  mul_162 = arg294_1 = None
        add_149: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_163, arg295_1);  mul_163 = arg295_1 = None
        convert_element_type_523: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_149, torch.float16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_396: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_523, [512, 1024])
        permute_198: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_108: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg297_1, view_396, permute_198);  arg297_1 = view_396 = permute_198 = None
        view_397: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_108, [1, 512, 1024]);  addmm_108 = None
        view_398: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_397, [1, 512, -1, 64]);  view_397 = None
        permute_199: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_533: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_199, torch.float32);  permute_199 = None
        mul_164: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_533, 0.3535533905932738);  convert_element_type_533 = None
        expand_75: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_164, [1, 16, 512, 64]);  mul_164 = None
        view_405: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_75, [16, 512, 64]);  expand_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_399: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_523, [512, 1024])
        permute_200: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_109: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg299_1, view_399, permute_200);  arg299_1 = view_399 = permute_200 = None
        view_400: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_109, [1, 512, 1024]);  addmm_109 = None
        view_401: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_400, [1, 512, -1, 64]);  view_400 = None
        permute_201: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_534: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_201, torch.float32);  permute_201 = None
        permute_204: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_534, [0, 1, 3, 2]);  convert_element_type_534 = None
        mul_165: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_204, 0.3535533905932738);  permute_204 = None
        expand_76: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_165, [1, 16, 64, 512]);  mul_165 = None
        view_406: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_76, [16, 64, 512]);  expand_76 = None
        bmm_36: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_405, view_406);  view_405 = view_406 = None
        view_407: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_36, [1, 16, 512, 512]);  bmm_36 = None
        full_default_55: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_36: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_55, full_default_54);  full_default_55 = full_default_54 = None
        add_150: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_407, where_36);  view_407 = where_36 = None
        eq_18: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_150, -inf)
        logical_not_36: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_18);  eq_18 = None
        any_19: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_36, -1, True);  logical_not_36 = None
        logical_not_37: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_19);  any_19 = None
        full_default_56: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_18: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_150, [-1], True)
        sub_55: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_150, amax_18);  add_150 = amax_18 = None
        exp_18: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_19: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        where_37: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_37, full_default_56, div_18);  logical_not_37 = full_default_56 = div_18 = None
        expand_77: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_37, [1, 16, 512, 512]);  where_37 = None
        view_408: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_77, [16, 512, 512]);  expand_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_402: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_523, [512, 1024])
        permute_202: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        addmm_110: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg301_1, view_402, permute_202);  arg301_1 = view_402 = permute_202 = None
        view_403: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_110, [1, 512, 1024]);  addmm_110 = None
        view_404: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_403, [1, 512, -1, 64]);  view_403 = None
        permute_203: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_535: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_203, torch.float32);  permute_203 = None
        expand_78: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_535, [1, 16, 512, 64]);  convert_element_type_535 = None
        view_409: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_78, [16, 512, 64]);  expand_78 = None
        bmm_37: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_408, view_409);  view_408 = view_409 = None
        view_410: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_37, [1, 16, 512, 64]);  bmm_37 = None
        convert_element_type_537: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_410, torch.float16);  view_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_205: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_537, [0, 2, 1, 3]);  convert_element_type_537 = None
        clone_55: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_411: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_55, [1, 512, -1]);  clone_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_412: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_411, [512, 1024]);  view_411 = None
        permute_206: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
        addmm_111: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg303_1, view_412, permute_206);  arg303_1 = view_412 = permute_206 = None
        view_413: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_111, [1, 512, 1024]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_151: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_413, convert_element_type_523);  view_413 = convert_element_type_523 = None
        convert_element_type_541: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_151, torch.float32);  add_151 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_541, [2], correction = 0, keepdim = True)
        getitem_74: "f32[1, 512, 1]" = var_mean_37[0]
        getitem_75: "f32[1, 512, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_56: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_541, getitem_75);  convert_element_type_541 = getitem_75 = None
        add_152: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
        rsqrt_37: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        mul_166: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
        mul_167: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_166, arg304_1);  mul_166 = arg304_1 = None
        add_153: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_167, arg305_1);  mul_167 = arg305_1 = None
        convert_element_type_542: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_153, torch.float16);  add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_414: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_542, [512, 1024])
        permute_207: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        addmm_112: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg307_1, view_414, permute_207);  arg307_1 = view_414 = permute_207 = None
        view_415: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_112, [1, 512, 3072]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_546: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        mul_168: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_546, 0.5)
        mul_169: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_546, 0.7071067811865476);  convert_element_type_546 = None
        erf_18: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_169);  mul_169 = None
        add_154: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_170: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_168, add_154);  mul_168 = add_154 = None
        convert_element_type_547: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_170, torch.float16);  mul_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_416: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_547, [512, 3072]);  convert_element_type_547 = None
        permute_208: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
        addmm_113: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg309_1, view_416, permute_208);  arg309_1 = view_416 = permute_208 = None
        view_417: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_113, [1, 512, 1024]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_155: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_417, convert_element_type_542);  view_417 = convert_element_type_542 = None
        convert_element_type_551: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_155, torch.float32);  add_155 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_551, [2], correction = 0, keepdim = True)
        getitem_76: "f32[1, 512, 1]" = var_mean_38[0]
        getitem_77: "f32[1, 512, 1]" = var_mean_38[1];  var_mean_38 = None
        sub_57: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_551, getitem_77);  convert_element_type_551 = getitem_77 = None
        add_156: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
        rsqrt_38: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        mul_171: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = rsqrt_38 = None
        mul_172: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_171, arg310_1);  mul_171 = arg310_1 = None
        add_157: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_172, arg311_1);  mul_172 = arg311_1 = None
        convert_element_type_552: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_157, torch.float16);  add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_418: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_552, [512, 1024])
        permute_209: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
        addmm_114: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg313_1, view_418, permute_209);  arg313_1 = view_418 = permute_209 = None
        view_419: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_114, [1, 512, 1024]);  addmm_114 = None
        view_420: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_419, [1, 512, -1, 64]);  view_419 = None
        permute_210: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_562: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_210, torch.float32);  permute_210 = None
        mul_173: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_562, 0.3535533905932738);  convert_element_type_562 = None
        expand_79: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_173, [1, 16, 512, 64]);  mul_173 = None
        view_427: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_79, [16, 512, 64]);  expand_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_421: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_552, [512, 1024])
        permute_211: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
        addmm_115: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg315_1, view_421, permute_211);  arg315_1 = view_421 = permute_211 = None
        view_422: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_115, [1, 512, 1024]);  addmm_115 = None
        view_423: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_422, [1, 512, -1, 64]);  view_422 = None
        permute_212: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_563: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_212, torch.float32);  permute_212 = None
        permute_215: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_563, [0, 1, 3, 2]);  convert_element_type_563 = None
        mul_174: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_215, 0.3535533905932738);  permute_215 = None
        expand_80: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_174, [1, 16, 64, 512]);  mul_174 = None
        view_428: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_80, [16, 64, 512]);  expand_80 = None
        bmm_38: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_427, view_428);  view_427 = view_428 = None
        view_429: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_38, [1, 16, 512, 512]);  bmm_38 = None
        full_default_58: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_38: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_58, full_default_57);  full_default_58 = full_default_57 = None
        add_158: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_429, where_38);  view_429 = where_38 = None
        eq_19: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_158, -inf)
        logical_not_38: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_19);  eq_19 = None
        any_20: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_38, -1, True);  logical_not_38 = None
        logical_not_39: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_20);  any_20 = None
        full_default_59: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_19: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_158, [-1], True)
        sub_58: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_158, amax_19);  add_158 = amax_19 = None
        exp_19: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_20: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        where_39: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_39, full_default_59, div_19);  logical_not_39 = full_default_59 = div_19 = None
        expand_81: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_39, [1, 16, 512, 512]);  where_39 = None
        view_430: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_81, [16, 512, 512]);  expand_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_424: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_552, [512, 1024])
        permute_213: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
        addmm_116: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg317_1, view_424, permute_213);  arg317_1 = view_424 = permute_213 = None
        view_425: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_116, [1, 512, 1024]);  addmm_116 = None
        view_426: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_425, [1, 512, -1, 64]);  view_425 = None
        permute_214: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_564: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_214, torch.float32);  permute_214 = None
        expand_82: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_564, [1, 16, 512, 64]);  convert_element_type_564 = None
        view_431: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_82, [16, 512, 64]);  expand_82 = None
        bmm_39: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_430, view_431);  view_430 = view_431 = None
        view_432: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_39, [1, 16, 512, 64]);  bmm_39 = None
        convert_element_type_566: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_432, torch.float16);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_216: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_566, [0, 2, 1, 3]);  convert_element_type_566 = None
        clone_58: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_433: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_58, [1, 512, -1]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_434: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_433, [512, 1024]);  view_433 = None
        permute_217: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
        addmm_117: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg319_1, view_434, permute_217);  arg319_1 = view_434 = permute_217 = None
        view_435: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_117, [1, 512, 1024]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_159: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_435, convert_element_type_552);  view_435 = convert_element_type_552 = None
        convert_element_type_570: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_159, torch.float32);  add_159 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_570, [2], correction = 0, keepdim = True)
        getitem_78: "f32[1, 512, 1]" = var_mean_39[0]
        getitem_79: "f32[1, 512, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_59: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_570, getitem_79);  convert_element_type_570 = getitem_79 = None
        add_160: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_39: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        mul_175: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
        mul_176: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_175, arg320_1);  mul_175 = arg320_1 = None
        add_161: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_176, arg321_1);  mul_176 = arg321_1 = None
        convert_element_type_571: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_161, torch.float16);  add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_436: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_571, [512, 1024])
        permute_218: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
        addmm_118: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg323_1, view_436, permute_218);  arg323_1 = view_436 = permute_218 = None
        view_437: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_118, [1, 512, 3072]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_575: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None
        mul_177: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_575, 0.5)
        mul_178: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_575, 0.7071067811865476);  convert_element_type_575 = None
        erf_19: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
        add_162: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_179: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_177, add_162);  mul_177 = add_162 = None
        convert_element_type_576: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_179, torch.float16);  mul_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_438: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_576, [512, 3072]);  convert_element_type_576 = None
        permute_219: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
        addmm_119: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg325_1, view_438, permute_219);  arg325_1 = view_438 = permute_219 = None
        view_439: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_119, [1, 512, 1024]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_163: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_439, convert_element_type_571);  view_439 = convert_element_type_571 = None
        convert_element_type_580: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_163, torch.float32);  add_163 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_580, [2], correction = 0, keepdim = True)
        getitem_80: "f32[1, 512, 1]" = var_mean_40[0]
        getitem_81: "f32[1, 512, 1]" = var_mean_40[1];  var_mean_40 = None
        sub_60: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_580, getitem_81);  convert_element_type_580 = getitem_81 = None
        add_164: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_40: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        mul_180: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = rsqrt_40 = None
        mul_181: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_180, arg326_1);  mul_180 = arg326_1 = None
        add_165: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_181, arg327_1);  mul_181 = arg327_1 = None
        convert_element_type_581: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_165, torch.float16);  add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_440: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_581, [512, 1024])
        permute_220: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_120: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg329_1, view_440, permute_220);  arg329_1 = view_440 = permute_220 = None
        view_441: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_120, [1, 512, 1024]);  addmm_120 = None
        view_442: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_441, [1, 512, -1, 64]);  view_441 = None
        permute_221: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_591: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_221, torch.float32);  permute_221 = None
        mul_182: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_591, 0.3535533905932738);  convert_element_type_591 = None
        expand_83: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_182, [1, 16, 512, 64]);  mul_182 = None
        view_449: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_83, [16, 512, 64]);  expand_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_443: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_581, [512, 1024])
        permute_222: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
        addmm_121: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg331_1, view_443, permute_222);  arg331_1 = view_443 = permute_222 = None
        view_444: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_121, [1, 512, 1024]);  addmm_121 = None
        view_445: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_444, [1, 512, -1, 64]);  view_444 = None
        permute_223: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_592: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_223, torch.float32);  permute_223 = None
        permute_226: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_592, [0, 1, 3, 2]);  convert_element_type_592 = None
        mul_183: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_226, 0.3535533905932738);  permute_226 = None
        expand_84: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_183, [1, 16, 64, 512]);  mul_183 = None
        view_450: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_84, [16, 64, 512]);  expand_84 = None
        bmm_40: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_449, view_450);  view_449 = view_450 = None
        view_451: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_40, [1, 16, 512, 512]);  bmm_40 = None
        full_default_61: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_40: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_61, full_default_60);  full_default_61 = full_default_60 = None
        add_166: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_451, where_40);  view_451 = where_40 = None
        eq_20: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_166, -inf)
        logical_not_40: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_20);  eq_20 = None
        any_21: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_40, -1, True);  logical_not_40 = None
        logical_not_41: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_21);  any_21 = None
        full_default_62: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_20: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_166, [-1], True)
        sub_61: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_166, amax_20);  add_166 = amax_20 = None
        exp_20: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_21: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        where_41: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_41, full_default_62, div_20);  logical_not_41 = full_default_62 = div_20 = None
        expand_85: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_41, [1, 16, 512, 512]);  where_41 = None
        view_452: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_85, [16, 512, 512]);  expand_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_446: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_581, [512, 1024])
        permute_224: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
        addmm_122: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg333_1, view_446, permute_224);  arg333_1 = view_446 = permute_224 = None
        view_447: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_122, [1, 512, 1024]);  addmm_122 = None
        view_448: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_447, [1, 512, -1, 64]);  view_447 = None
        permute_225: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_593: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_225, torch.float32);  permute_225 = None
        expand_86: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_593, [1, 16, 512, 64]);  convert_element_type_593 = None
        view_453: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_86, [16, 512, 64]);  expand_86 = None
        bmm_41: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_452, view_453);  view_452 = view_453 = None
        view_454: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_41, [1, 16, 512, 64]);  bmm_41 = None
        convert_element_type_595: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_454, torch.float16);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_227: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_595, [0, 2, 1, 3]);  convert_element_type_595 = None
        clone_61: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_455: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_61, [1, 512, -1]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_456: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_455, [512, 1024]);  view_455 = None
        permute_228: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
        addmm_123: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg335_1, view_456, permute_228);  arg335_1 = view_456 = permute_228 = None
        view_457: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_123, [1, 512, 1024]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_167: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_457, convert_element_type_581);  view_457 = convert_element_type_581 = None
        convert_element_type_599: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_167, torch.float32);  add_167 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_599, [2], correction = 0, keepdim = True)
        getitem_82: "f32[1, 512, 1]" = var_mean_41[0]
        getitem_83: "f32[1, 512, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_62: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_599, getitem_83);  convert_element_type_599 = getitem_83 = None
        add_168: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
        rsqrt_41: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_184: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
        mul_185: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_184, arg336_1);  mul_184 = arg336_1 = None
        add_169: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_185, arg337_1);  mul_185 = arg337_1 = None
        convert_element_type_600: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_169, torch.float16);  add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_458: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_600, [512, 1024])
        permute_229: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
        addmm_124: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg339_1, view_458, permute_229);  arg339_1 = view_458 = permute_229 = None
        view_459: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_124, [1, 512, 3072]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_604: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        mul_186: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_604, 0.5)
        mul_187: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_604, 0.7071067811865476);  convert_element_type_604 = None
        erf_20: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_187);  mul_187 = None
        add_170: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_188: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_186, add_170);  mul_186 = add_170 = None
        convert_element_type_605: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_188, torch.float16);  mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_460: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_605, [512, 3072]);  convert_element_type_605 = None
        permute_230: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
        addmm_125: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg341_1, view_460, permute_230);  arg341_1 = view_460 = permute_230 = None
        view_461: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_125, [1, 512, 1024]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_171: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_461, convert_element_type_600);  view_461 = convert_element_type_600 = None
        convert_element_type_609: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_171, torch.float32);  add_171 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_609, [2], correction = 0, keepdim = True)
        getitem_84: "f32[1, 512, 1]" = var_mean_42[0]
        getitem_85: "f32[1, 512, 1]" = var_mean_42[1];  var_mean_42 = None
        sub_63: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_609, getitem_85);  convert_element_type_609 = getitem_85 = None
        add_172: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
        rsqrt_42: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        mul_189: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = rsqrt_42 = None
        mul_190: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_189, arg342_1);  mul_189 = arg342_1 = None
        add_173: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_190, arg343_1);  mul_190 = arg343_1 = None
        convert_element_type_610: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_173, torch.float16);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_462: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_610, [512, 1024])
        permute_231: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
        addmm_126: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg345_1, view_462, permute_231);  arg345_1 = view_462 = permute_231 = None
        view_463: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_126, [1, 512, 1024]);  addmm_126 = None
        view_464: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_463, [1, 512, -1, 64]);  view_463 = None
        permute_232: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_620: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_232, torch.float32);  permute_232 = None
        mul_191: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_620, 0.3535533905932738);  convert_element_type_620 = None
        expand_87: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_191, [1, 16, 512, 64]);  mul_191 = None
        view_471: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_87, [16, 512, 64]);  expand_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_465: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_610, [512, 1024])
        permute_233: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg346_1, [1, 0]);  arg346_1 = None
        addmm_127: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg347_1, view_465, permute_233);  arg347_1 = view_465 = permute_233 = None
        view_466: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_127, [1, 512, 1024]);  addmm_127 = None
        view_467: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_466, [1, 512, -1, 64]);  view_466 = None
        permute_234: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_621: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_234, torch.float32);  permute_234 = None
        permute_237: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_621, [0, 1, 3, 2]);  convert_element_type_621 = None
        mul_192: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_237, 0.3535533905932738);  permute_237 = None
        expand_88: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_192, [1, 16, 64, 512]);  mul_192 = None
        view_472: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_88, [16, 64, 512]);  expand_88 = None
        bmm_42: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_471, view_472);  view_471 = view_472 = None
        view_473: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_42, [1, 16, 512, 512]);  bmm_42 = None
        full_default_64: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_42: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_64, full_default_63);  full_default_64 = full_default_63 = None
        add_174: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_473, where_42);  view_473 = where_42 = None
        eq_21: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_174, -inf)
        logical_not_42: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_21);  eq_21 = None
        any_22: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_42, -1, True);  logical_not_42 = None
        logical_not_43: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_22);  any_22 = None
        full_default_65: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_21: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_174, [-1], True)
        sub_64: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_174, amax_21);  add_174 = amax_21 = None
        exp_21: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_22: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        where_43: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_43, full_default_65, div_21);  logical_not_43 = full_default_65 = div_21 = None
        expand_89: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_43, [1, 16, 512, 512]);  where_43 = None
        view_474: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_89, [16, 512, 512]);  expand_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_468: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_610, [512, 1024])
        permute_235: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg348_1, [1, 0]);  arg348_1 = None
        addmm_128: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg349_1, view_468, permute_235);  arg349_1 = view_468 = permute_235 = None
        view_469: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_128, [1, 512, 1024]);  addmm_128 = None
        view_470: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_469, [1, 512, -1, 64]);  view_469 = None
        permute_236: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_622: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_236, torch.float32);  permute_236 = None
        expand_90: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_622, [1, 16, 512, 64]);  convert_element_type_622 = None
        view_475: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_90, [16, 512, 64]);  expand_90 = None
        bmm_43: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_474, view_475);  view_474 = view_475 = None
        view_476: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_43, [1, 16, 512, 64]);  bmm_43 = None
        convert_element_type_624: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_476, torch.float16);  view_476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_238: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_624, [0, 2, 1, 3]);  convert_element_type_624 = None
        clone_64: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_477: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_64, [1, 512, -1]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_478: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_477, [512, 1024]);  view_477 = None
        permute_239: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
        addmm_129: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg351_1, view_478, permute_239);  arg351_1 = view_478 = permute_239 = None
        view_479: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_129, [1, 512, 1024]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_175: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_479, convert_element_type_610);  view_479 = convert_element_type_610 = None
        convert_element_type_628: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_175, torch.float32);  add_175 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_628, [2], correction = 0, keepdim = True)
        getitem_86: "f32[1, 512, 1]" = var_mean_43[0]
        getitem_87: "f32[1, 512, 1]" = var_mean_43[1];  var_mean_43 = None
        sub_65: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_628, getitem_87);  convert_element_type_628 = getitem_87 = None
        add_176: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_43: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        mul_193: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
        mul_194: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_193, arg352_1);  mul_193 = arg352_1 = None
        add_177: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_194, arg353_1);  mul_194 = arg353_1 = None
        convert_element_type_629: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_177, torch.float16);  add_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_480: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_629, [512, 1024])
        permute_240: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg354_1, [1, 0]);  arg354_1 = None
        addmm_130: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg355_1, view_480, permute_240);  arg355_1 = view_480 = permute_240 = None
        view_481: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_130, [1, 512, 3072]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_633: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_481, torch.float32);  view_481 = None
        mul_195: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_633, 0.5)
        mul_196: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_633, 0.7071067811865476);  convert_element_type_633 = None
        erf_21: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_196);  mul_196 = None
        add_178: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_197: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_195, add_178);  mul_195 = add_178 = None
        convert_element_type_634: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_197, torch.float16);  mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_482: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_634, [512, 3072]);  convert_element_type_634 = None
        permute_241: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
        addmm_131: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg357_1, view_482, permute_241);  arg357_1 = view_482 = permute_241 = None
        view_483: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_131, [1, 512, 1024]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_179: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_483, convert_element_type_629);  view_483 = convert_element_type_629 = None
        convert_element_type_638: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_179, torch.float32);  add_179 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_638, [2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 512, 1]" = var_mean_44[0]
        getitem_89: "f32[1, 512, 1]" = var_mean_44[1];  var_mean_44 = None
        sub_66: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_638, getitem_89);  convert_element_type_638 = getitem_89 = None
        add_180: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_44: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        mul_198: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = rsqrt_44 = None
        mul_199: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_198, arg358_1);  mul_198 = arg358_1 = None
        add_181: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_199, arg359_1);  mul_199 = arg359_1 = None
        convert_element_type_639: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_181, torch.float16);  add_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_484: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_639, [512, 1024])
        permute_242: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
        addmm_132: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg361_1, view_484, permute_242);  arg361_1 = view_484 = permute_242 = None
        view_485: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_132, [1, 512, 1024]);  addmm_132 = None
        view_486: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_485, [1, 512, -1, 64]);  view_485 = None
        permute_243: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_649: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_243, torch.float32);  permute_243 = None
        mul_200: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_649, 0.3535533905932738);  convert_element_type_649 = None
        expand_91: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_200, [1, 16, 512, 64]);  mul_200 = None
        view_493: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_91, [16, 512, 64]);  expand_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_487: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_639, [512, 1024])
        permute_244: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
        addmm_133: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg363_1, view_487, permute_244);  arg363_1 = view_487 = permute_244 = None
        view_488: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_133, [1, 512, 1024]);  addmm_133 = None
        view_489: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_488, [1, 512, -1, 64]);  view_488 = None
        permute_245: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_650: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_245, torch.float32);  permute_245 = None
        permute_248: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_650, [0, 1, 3, 2]);  convert_element_type_650 = None
        mul_201: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_248, 0.3535533905932738);  permute_248 = None
        expand_92: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_201, [1, 16, 64, 512]);  mul_201 = None
        view_494: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_92, [16, 64, 512]);  expand_92 = None
        bmm_44: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_493, view_494);  view_493 = view_494 = None
        view_495: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_44, [1, 16, 512, 512]);  bmm_44 = None
        full_default_67: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_44: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_67, full_default_66);  full_default_67 = full_default_66 = None
        add_182: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_495, where_44);  view_495 = where_44 = None
        eq_22: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_182, -inf)
        logical_not_44: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_22);  eq_22 = None
        any_23: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_44, -1, True);  logical_not_44 = None
        logical_not_45: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_23);  any_23 = None
        full_default_68: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_22: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_182, [-1], True)
        sub_67: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_182, amax_22);  add_182 = amax_22 = None
        exp_22: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_23: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        where_45: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_45, full_default_68, div_22);  logical_not_45 = full_default_68 = div_22 = None
        expand_93: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_45, [1, 16, 512, 512]);  where_45 = None
        view_496: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_93, [16, 512, 512]);  expand_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_490: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_639, [512, 1024])
        permute_246: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg364_1, [1, 0]);  arg364_1 = None
        addmm_134: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg365_1, view_490, permute_246);  arg365_1 = view_490 = permute_246 = None
        view_491: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_134, [1, 512, 1024]);  addmm_134 = None
        view_492: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_491, [1, 512, -1, 64]);  view_491 = None
        permute_247: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_651: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_247, torch.float32);  permute_247 = None
        expand_94: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_651, [1, 16, 512, 64]);  convert_element_type_651 = None
        view_497: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_94, [16, 512, 64]);  expand_94 = None
        bmm_45: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_496, view_497);  view_496 = view_497 = None
        view_498: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_45, [1, 16, 512, 64]);  bmm_45 = None
        convert_element_type_653: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_498, torch.float16);  view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_249: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_653, [0, 2, 1, 3]);  convert_element_type_653 = None
        clone_67: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_499: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_67, [1, 512, -1]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_500: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_499, [512, 1024]);  view_499 = None
        permute_250: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg366_1, [1, 0]);  arg366_1 = None
        addmm_135: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg367_1, view_500, permute_250);  arg367_1 = view_500 = permute_250 = None
        view_501: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_135, [1, 512, 1024]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_183: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_501, convert_element_type_639);  view_501 = convert_element_type_639 = None
        convert_element_type_657: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_183, torch.float32);  add_183 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_657, [2], correction = 0, keepdim = True)
        getitem_90: "f32[1, 512, 1]" = var_mean_45[0]
        getitem_91: "f32[1, 512, 1]" = var_mean_45[1];  var_mean_45 = None
        sub_68: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_657, getitem_91);  convert_element_type_657 = getitem_91 = None
        add_184: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
        rsqrt_45: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        mul_202: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
        mul_203: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_202, arg368_1);  mul_202 = arg368_1 = None
        add_185: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_203, arg369_1);  mul_203 = arg369_1 = None
        convert_element_type_658: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_185, torch.float16);  add_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_502: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_658, [512, 1024])
        permute_251: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
        addmm_136: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg371_1, view_502, permute_251);  arg371_1 = view_502 = permute_251 = None
        view_503: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_136, [1, 512, 3072]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_662: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None
        mul_204: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_662, 0.5)
        mul_205: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_662, 0.7071067811865476);  convert_element_type_662 = None
        erf_22: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_205);  mul_205 = None
        add_186: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_206: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_204, add_186);  mul_204 = add_186 = None
        convert_element_type_663: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_206, torch.float16);  mul_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_663, [512, 3072]);  convert_element_type_663 = None
        permute_252: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg372_1, [1, 0]);  arg372_1 = None
        addmm_137: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg373_1, view_504, permute_252);  arg373_1 = view_504 = permute_252 = None
        view_505: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_137, [1, 512, 1024]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_187: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_505, convert_element_type_658);  view_505 = convert_element_type_658 = None
        convert_element_type_667: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_187, torch.float32);  add_187 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_667, [2], correction = 0, keepdim = True)
        getitem_92: "f32[1, 512, 1]" = var_mean_46[0]
        getitem_93: "f32[1, 512, 1]" = var_mean_46[1];  var_mean_46 = None
        sub_69: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_667, getitem_93);  convert_element_type_667 = getitem_93 = None
        add_188: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
        rsqrt_46: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_207: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = rsqrt_46 = None
        mul_208: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_207, arg374_1);  mul_207 = arg374_1 = None
        add_189: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_208, arg375_1);  mul_208 = arg375_1 = None
        convert_element_type_668: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_189, torch.float16);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_506: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_668, [512, 1024])
        permute_253: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
        addmm_138: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg377_1, view_506, permute_253);  arg377_1 = view_506 = permute_253 = None
        view_507: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_138, [1, 512, 1024]);  addmm_138 = None
        view_508: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_507, [1, 512, -1, 64]);  view_507 = None
        permute_254: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_678: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_254, torch.float32);  permute_254 = None
        mul_209: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(convert_element_type_678, 0.3535533905932738);  convert_element_type_678 = None
        expand_95: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(mul_209, [1, 16, 512, 64]);  mul_209 = None
        view_515: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_95, [16, 512, 64]);  expand_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_509: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_668, [512, 1024])
        permute_255: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg378_1, [1, 0]);  arg378_1 = None
        addmm_139: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg379_1, view_509, permute_255);  arg379_1 = view_509 = permute_255 = None
        view_510: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_139, [1, 512, 1024]);  addmm_139 = None
        view_511: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_510, [1, 512, -1, 64]);  view_510 = None
        permute_256: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_679: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_256, torch.float32);  permute_256 = None
        permute_259: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_679, [0, 1, 3, 2]);  convert_element_type_679 = None
        mul_210: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(permute_259, 0.3535533905932738);  permute_259 = None
        expand_96: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(mul_210, [1, 16, 64, 512]);  mul_210 = None
        view_516: "f32[16, 64, 512]" = torch.ops.aten.reshape.default(expand_96, [16, 64, 512]);  expand_96 = None
        bmm_46: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_515, view_516);  view_515 = view_516 = None
        view_517: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_46, [1, 16, 512, 512]);  bmm_46 = None
        full_default_70: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_69: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_46: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_70, full_default_69);  expand_2 = full_default_70 = full_default_69 = None
        add_190: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_517, where_46);  view_517 = where_46 = None
        eq_23: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_190, -inf)
        logical_not_46: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_23);  eq_23 = None
        any_24: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_46, -1, True);  logical_not_46 = None
        logical_not_47: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_24);  any_24 = None
        full_default_71: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_23: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_190, [-1], True)
        sub_70: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_190, amax_23);  add_190 = amax_23 = None
        exp_23: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        sum_24: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        where_47: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_47, full_default_71, div_23);  logical_not_47 = full_default_71 = div_23 = None
        expand_97: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_47, [1, 16, 512, 512]);  where_47 = None
        view_518: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(expand_97, [16, 512, 512]);  expand_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_512: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_668, [512, 1024])
        permute_257: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
        addmm_140: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg381_1, view_512, permute_257);  arg381_1 = view_512 = permute_257 = None
        view_513: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_140, [1, 512, 1024]);  addmm_140 = None
        view_514: "f16[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_513, [1, 512, -1, 64]);  view_513 = None
        permute_258: "f16[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_680: "f32[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_258, torch.float32);  permute_258 = None
        expand_98: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_680, [1, 16, 512, 64]);  convert_element_type_680 = None
        view_519: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(expand_98, [16, 512, 64]);  expand_98 = None
        bmm_47: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_518, view_519);  view_518 = view_519 = None
        view_520: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_47, [1, 16, 512, 64]);  bmm_47 = None
        convert_element_type_682: "f16[1, 16, 512, 64]" = torch.ops.prims.convert_element_type.default(view_520, torch.float16);  view_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_260: "f16[1, 512, 16, 64]" = torch.ops.aten.permute.default(convert_element_type_682, [0, 2, 1, 3]);  convert_element_type_682 = None
        clone_70: "f16[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_521: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_70, [1, 512, -1]);  clone_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_522: "f16[512, 1024]" = torch.ops.aten.reshape.default(view_521, [512, 1024]);  view_521 = None
        permute_261: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg382_1, [1, 0]);  arg382_1 = None
        addmm_141: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg383_1, view_522, permute_261);  arg383_1 = view_522 = permute_261 = None
        view_523: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_141, [1, 512, 1024]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_191: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_523, convert_element_type_668);  view_523 = convert_element_type_668 = None
        convert_element_type_686: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_191, torch.float32);  add_191 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_686, [2], correction = 0, keepdim = True)
        getitem_94: "f32[1, 512, 1]" = var_mean_47[0]
        getitem_95: "f32[1, 512, 1]" = var_mean_47[1];  var_mean_47 = None
        sub_71: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_686, getitem_95);  convert_element_type_686 = getitem_95 = None
        add_192: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_47: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_211: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
        mul_212: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_211, arg384_1);  mul_211 = arg384_1 = None
        add_193: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_212, arg385_1);  mul_212 = arg385_1 = None
        convert_element_type_687: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_193, torch.float16);  add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_524: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_687, [512, 1024])
        permute_262: "f16[1024, 3072]" = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
        addmm_142: "f16[512, 3072]" = torch.ops.aten.addmm.default(arg387_1, view_524, permute_262);  arg387_1 = view_524 = permute_262 = None
        view_525: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_142, [1, 512, 3072]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_691: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_525, torch.float32);  view_525 = None
        mul_213: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_691, 0.5)
        mul_214: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_691, 0.7071067811865476);  convert_element_type_691 = None
        erf_23: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_214);  mul_214 = None
        add_194: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_215: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_213, add_194);  mul_213 = add_194 = None
        convert_element_type_692: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_215, torch.float16);  mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_526: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_692, [512, 3072]);  convert_element_type_692 = None
        permute_263: "f16[3072, 1024]" = torch.ops.aten.permute.default(arg388_1, [1, 0]);  arg388_1 = None
        addmm_143: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg389_1, view_526, permute_263);  arg389_1 = view_526 = permute_263 = None
        view_527: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_143, [1, 512, 1024]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_195: "f16[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_527, convert_element_type_687);  view_527 = convert_element_type_687 = None
        convert_element_type_696: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_195, torch.float32);  add_195 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_696, [2], correction = 0, keepdim = True)
        getitem_96: "f32[1, 512, 1]" = var_mean_48[0]
        getitem_97: "f32[1, 512, 1]" = var_mean_48[1];  var_mean_48 = None
        sub_72: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_696, getitem_97);  convert_element_type_696 = getitem_97 = None
        add_196: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_48: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        mul_216: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = rsqrt_48 = None
        mul_217: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_216, arg390_1);  mul_216 = arg390_1 = None
        add_197: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_217, arg391_1);  mul_217 = arg391_1 = None
        convert_element_type_697: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_197, torch.float16);  add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        view_528: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_697, [512, 1024]);  convert_element_type_697 = None
        permute_264: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg392_1, [1, 0]);  arg392_1 = None
        addmm_144: "f16[512, 1024]" = torch.ops.aten.addmm.default(arg393_1, view_528, permute_264);  arg393_1 = view_528 = permute_264 = None
        view_529: "f16[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_144, [1, 512, 1024]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_701: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None
        mul_218: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_701, 0.5)
        mul_219: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_701, 0.7071067811865476);  convert_element_type_701 = None
        erf_24: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
        add_198: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_220: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_218, add_198);  mul_218 = add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_default: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_220, torch.float32);  mul_220 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem_98: "f32[1, 512, 1]" = var_mean_49[0]
        getitem_99: "f32[1, 512, 1]" = var_mean_49[1];  var_mean_49 = None
        sub_73: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_99);  convert_element_type_default = getitem_99 = None
        add_199: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
        rsqrt_49: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        mul_221: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_49);  sub_73 = rsqrt_49 = None
        mul_222: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_221, arg394_1);  mul_221 = arg394_1 = None
        add_200: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_222, arg395_1);  mul_222 = arg395_1 = None
        convert_element_type_704: "f16[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_200, torch.float16);  add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        view_530: "f16[512, 1024]" = torch.ops.aten.reshape.default(convert_element_type_704, [512, 1024]);  convert_element_type_704 = None
        permute_265: "f16[1024, 30522]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_145: "f16[512, 30522]" = torch.ops.aten.addmm.default(arg396_1, view_530, permute_265);  arg396_1 = view_530 = permute_265 = None
        view_531: "f16[1, 512, 30522]" = torch.ops.aten.reshape.default(addmm_145, [1, 512, 30522]);  addmm_145 = None
        return (view_531,)
