class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 1024]", arg1_1: "f16[50257, 1280]", arg2_1: "f16[1024, 1280]", arg3_1: "f16[1280]", arg4_1: "f16[1280]", arg5_1: "f16[3840]", arg6_1: "f16[1280, 3840]", arg7_1: "f16[1280]", arg8_1: "f16[1280, 1280]", arg9_1: "f16[1280]", arg10_1: "f16[1280]", arg11_1: "f16[5120]", arg12_1: "f16[1280, 5120]", arg13_1: "f16[1280]", arg14_1: "f16[5120, 1280]", arg15_1: "f16[1280]", arg16_1: "f16[1280]", arg17_1: "f16[3840]", arg18_1: "f16[1280, 3840]", arg19_1: "f16[1280]", arg20_1: "f16[1280, 1280]", arg21_1: "f16[1280]", arg22_1: "f16[1280]", arg23_1: "f16[5120]", arg24_1: "f16[1280, 5120]", arg25_1: "f16[1280]", arg26_1: "f16[5120, 1280]", arg27_1: "f16[1280]", arg28_1: "f16[1280]", arg29_1: "f16[3840]", arg30_1: "f16[1280, 3840]", arg31_1: "f16[1280]", arg32_1: "f16[1280, 1280]", arg33_1: "f16[1280]", arg34_1: "f16[1280]", arg35_1: "f16[5120]", arg36_1: "f16[1280, 5120]", arg37_1: "f16[1280]", arg38_1: "f16[5120, 1280]", arg39_1: "f16[1280]", arg40_1: "f16[1280]", arg41_1: "f16[3840]", arg42_1: "f16[1280, 3840]", arg43_1: "f16[1280]", arg44_1: "f16[1280, 1280]", arg45_1: "f16[1280]", arg46_1: "f16[1280]", arg47_1: "f16[5120]", arg48_1: "f16[1280, 5120]", arg49_1: "f16[1280]", arg50_1: "f16[5120, 1280]", arg51_1: "f16[1280]", arg52_1: "f16[1280]", arg53_1: "f16[3840]", arg54_1: "f16[1280, 3840]", arg55_1: "f16[1280]", arg56_1: "f16[1280, 1280]", arg57_1: "f16[1280]", arg58_1: "f16[1280]", arg59_1: "f16[5120]", arg60_1: "f16[1280, 5120]", arg61_1: "f16[1280]", arg62_1: "f16[5120, 1280]", arg63_1: "f16[1280]", arg64_1: "f16[1280]", arg65_1: "f16[3840]", arg66_1: "f16[1280, 3840]", arg67_1: "f16[1280]", arg68_1: "f16[1280, 1280]", arg69_1: "f16[1280]", arg70_1: "f16[1280]", arg71_1: "f16[5120]", arg72_1: "f16[1280, 5120]", arg73_1: "f16[1280]", arg74_1: "f16[5120, 1280]", arg75_1: "f16[1280]", arg76_1: "f16[1280]", arg77_1: "f16[3840]", arg78_1: "f16[1280, 3840]", arg79_1: "f16[1280]", arg80_1: "f16[1280, 1280]", arg81_1: "f16[1280]", arg82_1: "f16[1280]", arg83_1: "f16[5120]", arg84_1: "f16[1280, 5120]", arg85_1: "f16[1280]", arg86_1: "f16[5120, 1280]", arg87_1: "f16[1280]", arg88_1: "f16[1280]", arg89_1: "f16[3840]", arg90_1: "f16[1280, 3840]", arg91_1: "f16[1280]", arg92_1: "f16[1280, 1280]", arg93_1: "f16[1280]", arg94_1: "f16[1280]", arg95_1: "f16[5120]", arg96_1: "f16[1280, 5120]", arg97_1: "f16[1280]", arg98_1: "f16[5120, 1280]", arg99_1: "f16[1280]", arg100_1: "f16[1280]", arg101_1: "f16[3840]", arg102_1: "f16[1280, 3840]", arg103_1: "f16[1280]", arg104_1: "f16[1280, 1280]", arg105_1: "f16[1280]", arg106_1: "f16[1280]", arg107_1: "f16[5120]", arg108_1: "f16[1280, 5120]", arg109_1: "f16[1280]", arg110_1: "f16[5120, 1280]", arg111_1: "f16[1280]", arg112_1: "f16[1280]", arg113_1: "f16[3840]", arg114_1: "f16[1280, 3840]", arg115_1: "f16[1280]", arg116_1: "f16[1280, 1280]", arg117_1: "f16[1280]", arg118_1: "f16[1280]", arg119_1: "f16[5120]", arg120_1: "f16[1280, 5120]", arg121_1: "f16[1280]", arg122_1: "f16[5120, 1280]", arg123_1: "f16[1280]", arg124_1: "f16[1280]", arg125_1: "f16[3840]", arg126_1: "f16[1280, 3840]", arg127_1: "f16[1280]", arg128_1: "f16[1280, 1280]", arg129_1: "f16[1280]", arg130_1: "f16[1280]", arg131_1: "f16[5120]", arg132_1: "f16[1280, 5120]", arg133_1: "f16[1280]", arg134_1: "f16[5120, 1280]", arg135_1: "f16[1280]", arg136_1: "f16[1280]", arg137_1: "f16[3840]", arg138_1: "f16[1280, 3840]", arg139_1: "f16[1280]", arg140_1: "f16[1280, 1280]", arg141_1: "f16[1280]", arg142_1: "f16[1280]", arg143_1: "f16[5120]", arg144_1: "f16[1280, 5120]", arg145_1: "f16[1280]", arg146_1: "f16[5120, 1280]", arg147_1: "f16[1280]", arg148_1: "f16[1280]", arg149_1: "f16[3840]", arg150_1: "f16[1280, 3840]", arg151_1: "f16[1280]", arg152_1: "f16[1280, 1280]", arg153_1: "f16[1280]", arg154_1: "f16[1280]", arg155_1: "f16[5120]", arg156_1: "f16[1280, 5120]", arg157_1: "f16[1280]", arg158_1: "f16[5120, 1280]", arg159_1: "f16[1280]", arg160_1: "f16[1280]", arg161_1: "f16[3840]", arg162_1: "f16[1280, 3840]", arg163_1: "f16[1280]", arg164_1: "f16[1280, 1280]", arg165_1: "f16[1280]", arg166_1: "f16[1280]", arg167_1: "f16[5120]", arg168_1: "f16[1280, 5120]", arg169_1: "f16[1280]", arg170_1: "f16[5120, 1280]", arg171_1: "f16[1280]", arg172_1: "f16[1280]", arg173_1: "f16[3840]", arg174_1: "f16[1280, 3840]", arg175_1: "f16[1280]", arg176_1: "f16[1280, 1280]", arg177_1: "f16[1280]", arg178_1: "f16[1280]", arg179_1: "f16[5120]", arg180_1: "f16[1280, 5120]", arg181_1: "f16[1280]", arg182_1: "f16[5120, 1280]", arg183_1: "f16[1280]", arg184_1: "f16[1280]", arg185_1: "f16[3840]", arg186_1: "f16[1280, 3840]", arg187_1: "f16[1280]", arg188_1: "f16[1280, 1280]", arg189_1: "f16[1280]", arg190_1: "f16[1280]", arg191_1: "f16[5120]", arg192_1: "f16[1280, 5120]", arg193_1: "f16[1280]", arg194_1: "f16[5120, 1280]", arg195_1: "f16[1280]", arg196_1: "f16[1280]", arg197_1: "f16[3840]", arg198_1: "f16[1280, 3840]", arg199_1: "f16[1280]", arg200_1: "f16[1280, 1280]", arg201_1: "f16[1280]", arg202_1: "f16[1280]", arg203_1: "f16[5120]", arg204_1: "f16[1280, 5120]", arg205_1: "f16[1280]", arg206_1: "f16[5120, 1280]", arg207_1: "f16[1280]", arg208_1: "f16[1280]", arg209_1: "f16[3840]", arg210_1: "f16[1280, 3840]", arg211_1: "f16[1280]", arg212_1: "f16[1280, 1280]", arg213_1: "f16[1280]", arg214_1: "f16[1280]", arg215_1: "f16[5120]", arg216_1: "f16[1280, 5120]", arg217_1: "f16[1280]", arg218_1: "f16[5120, 1280]", arg219_1: "f16[1280]", arg220_1: "f16[1280]", arg221_1: "f16[3840]", arg222_1: "f16[1280, 3840]", arg223_1: "f16[1280]", arg224_1: "f16[1280, 1280]", arg225_1: "f16[1280]", arg226_1: "f16[1280]", arg227_1: "f16[5120]", arg228_1: "f16[1280, 5120]", arg229_1: "f16[1280]", arg230_1: "f16[5120, 1280]", arg231_1: "f16[1280]", arg232_1: "f16[1280]", arg233_1: "f16[3840]", arg234_1: "f16[1280, 3840]", arg235_1: "f16[1280]", arg236_1: "f16[1280, 1280]", arg237_1: "f16[1280]", arg238_1: "f16[1280]", arg239_1: "f16[5120]", arg240_1: "f16[1280, 5120]", arg241_1: "f16[1280]", arg242_1: "f16[5120, 1280]", arg243_1: "f16[1280]", arg244_1: "f16[1280]", arg245_1: "f16[3840]", arg246_1: "f16[1280, 3840]", arg247_1: "f16[1280]", arg248_1: "f16[1280, 1280]", arg249_1: "f16[1280]", arg250_1: "f16[1280]", arg251_1: "f16[5120]", arg252_1: "f16[1280, 5120]", arg253_1: "f16[1280]", arg254_1: "f16[5120, 1280]", arg255_1: "f16[1280]", arg256_1: "f16[1280]", arg257_1: "f16[3840]", arg258_1: "f16[1280, 3840]", arg259_1: "f16[1280]", arg260_1: "f16[1280, 1280]", arg261_1: "f16[1280]", arg262_1: "f16[1280]", arg263_1: "f16[5120]", arg264_1: "f16[1280, 5120]", arg265_1: "f16[1280]", arg266_1: "f16[5120, 1280]", arg267_1: "f16[1280]", arg268_1: "f16[1280]", arg269_1: "f16[3840]", arg270_1: "f16[1280, 3840]", arg271_1: "f16[1280]", arg272_1: "f16[1280, 1280]", arg273_1: "f16[1280]", arg274_1: "f16[1280]", arg275_1: "f16[5120]", arg276_1: "f16[1280, 5120]", arg277_1: "f16[1280]", arg278_1: "f16[5120, 1280]", arg279_1: "f16[1280]", arg280_1: "f16[1280]", arg281_1: "f16[3840]", arg282_1: "f16[1280, 3840]", arg283_1: "f16[1280]", arg284_1: "f16[1280, 1280]", arg285_1: "f16[1280]", arg286_1: "f16[1280]", arg287_1: "f16[5120]", arg288_1: "f16[1280, 5120]", arg289_1: "f16[1280]", arg290_1: "f16[5120, 1280]", arg291_1: "f16[1280]", arg292_1: "f16[1280]", arg293_1: "f16[3840]", arg294_1: "f16[1280, 3840]", arg295_1: "f16[1280]", arg296_1: "f16[1280, 1280]", arg297_1: "f16[1280]", arg298_1: "f16[1280]", arg299_1: "f16[5120]", arg300_1: "f16[1280, 5120]", arg301_1: "f16[1280]", arg302_1: "f16[5120, 1280]", arg303_1: "f16[1280]", arg304_1: "f16[1280]", arg305_1: "f16[3840]", arg306_1: "f16[1280, 3840]", arg307_1: "f16[1280]", arg308_1: "f16[1280, 1280]", arg309_1: "f16[1280]", arg310_1: "f16[1280]", arg311_1: "f16[5120]", arg312_1: "f16[1280, 5120]", arg313_1: "f16[1280]", arg314_1: "f16[5120, 1280]", arg315_1: "f16[1280]", arg316_1: "f16[1280]", arg317_1: "f16[3840]", arg318_1: "f16[1280, 3840]", arg319_1: "f16[1280]", arg320_1: "f16[1280, 1280]", arg321_1: "f16[1280]", arg322_1: "f16[1280]", arg323_1: "f16[5120]", arg324_1: "f16[1280, 5120]", arg325_1: "f16[1280]", arg326_1: "f16[5120, 1280]", arg327_1: "f16[1280]", arg328_1: "f16[1280]", arg329_1: "f16[3840]", arg330_1: "f16[1280, 3840]", arg331_1: "f16[1280]", arg332_1: "f16[1280, 1280]", arg333_1: "f16[1280]", arg334_1: "f16[1280]", arg335_1: "f16[5120]", arg336_1: "f16[1280, 5120]", arg337_1: "f16[1280]", arg338_1: "f16[5120, 1280]", arg339_1: "f16[1280]", arg340_1: "f16[1280]", arg341_1: "f16[3840]", arg342_1: "f16[1280, 3840]", arg343_1: "f16[1280]", arg344_1: "f16[1280, 1280]", arg345_1: "f16[1280]", arg346_1: "f16[1280]", arg347_1: "f16[5120]", arg348_1: "f16[1280, 5120]", arg349_1: "f16[1280]", arg350_1: "f16[5120, 1280]", arg351_1: "f16[1280]", arg352_1: "f16[1280]", arg353_1: "f16[3840]", arg354_1: "f16[1280, 3840]", arg355_1: "f16[1280]", arg356_1: "f16[1280, 1280]", arg357_1: "f16[1280]", arg358_1: "f16[1280]", arg359_1: "f16[5120]", arg360_1: "f16[1280, 5120]", arg361_1: "f16[1280]", arg362_1: "f16[5120, 1280]", arg363_1: "f16[1280]", arg364_1: "f16[1280]", arg365_1: "f16[3840]", arg366_1: "f16[1280, 3840]", arg367_1: "f16[1280]", arg368_1: "f16[1280, 1280]", arg369_1: "f16[1280]", arg370_1: "f16[1280]", arg371_1: "f16[5120]", arg372_1: "f16[1280, 5120]", arg373_1: "f16[1280]", arg374_1: "f16[5120, 1280]", arg375_1: "f16[1280]", arg376_1: "f16[1280]", arg377_1: "f16[3840]", arg378_1: "f16[1280, 3840]", arg379_1: "f16[1280]", arg380_1: "f16[1280, 1280]", arg381_1: "f16[1280]", arg382_1: "f16[1280]", arg383_1: "f16[5120]", arg384_1: "f16[1280, 5120]", arg385_1: "f16[1280]", arg386_1: "f16[5120, 1280]", arg387_1: "f16[1280]", arg388_1: "f16[1280]", arg389_1: "f16[3840]", arg390_1: "f16[1280, 3840]", arg391_1: "f16[1280]", arg392_1: "f16[1280, 1280]", arg393_1: "f16[1280]", arg394_1: "f16[1280]", arg395_1: "f16[5120]", arg396_1: "f16[1280, 5120]", arg397_1: "f16[1280]", arg398_1: "f16[5120, 1280]", arg399_1: "f16[1280]", arg400_1: "f16[1280]", arg401_1: "f16[3840]", arg402_1: "f16[1280, 3840]", arg403_1: "f16[1280]", arg404_1: "f16[1280, 1280]", arg405_1: "f16[1280]", arg406_1: "f16[1280]", arg407_1: "f16[5120]", arg408_1: "f16[1280, 5120]", arg409_1: "f16[1280]", arg410_1: "f16[5120, 1280]", arg411_1: "f16[1280]", arg412_1: "f16[1280]", arg413_1: "f16[3840]", arg414_1: "f16[1280, 3840]", arg415_1: "f16[1280]", arg416_1: "f16[1280, 1280]", arg417_1: "f16[1280]", arg418_1: "f16[1280]", arg419_1: "f16[5120]", arg420_1: "f16[1280, 5120]", arg421_1: "f16[1280]", arg422_1: "f16[5120, 1280]", arg423_1: "f16[1280]", arg424_1: "f16[1280]", arg425_1: "f16[3840]", arg426_1: "f16[1280, 3840]", arg427_1: "f16[1280]", arg428_1: "f16[1280, 1280]", arg429_1: "f16[1280]", arg430_1: "f16[1280]", arg431_1: "f16[5120]", arg432_1: "f16[1280, 5120]", arg433_1: "f16[1280]", arg434_1: "f16[5120, 1280]", arg435_1: "f16[1280]", arg436_1: "f16[1280]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f16[1, 1024, 1280]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:581 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:582 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f16[1, 1024, 1280]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_1: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 1024, 1]" = var_mean[0]
        getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
        sub_2: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_4: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_1: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_5: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type_1: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_5, torch.float16);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_1: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_1, [-1, 1280]);  convert_element_type_1 = None
        addmm: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg5_1, view_1, arg6_1);  arg5_1 = view_1 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_2: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm, [1, 1024, 3840]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_2, 1280, 2);  view_2 = None
        getitem_2: "f16[1, 1024, 1280]" = split[0]
        getitem_3: "f16[1, 1024, 1280]" = split[1]
        getitem_4: "f16[1, 1024, 1280]" = split[2];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_5: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_2, [1, 1024, -1, 64]);  getitem_2 = None
        permute_2: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_3: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_3, [1, 1024, -1, 64]);  getitem_3 = None
        permute: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_4: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_4, [1, 1024, -1, 64]);  getitem_4 = None
        permute_1: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[1024]" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_7: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_8: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        unsqueeze_9: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[1024]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_4: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_5: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[1, 1025]" = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = unsqueeze = None
        slice_3: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 1025)
        slice_2: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 1024);  cat = None
        sub_1: "i64[1, 1024]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[1, 1024]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[1, 1, 1024, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[1, 1, 1, 1024]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[1, 1, 1024, 1024]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[1, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 1024, 1024]" = torch.ops.aten.expand.default(bitwise_and_1, [1, -1, 1024, 1024]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_2, permute, permute_1, where, False, scale = 0.125);  permute_2 = permute = permute_1 = where = None
        getitem_5: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_3, [1, 1024, -1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_6, [-1, 1280]);  view_6 = None
        addmm_1: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg7_1, view_7, arg8_1);  arg7_1 = view_7 = arg8_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_8: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_1, [1, 1024, 1280]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_6: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_8, add_1);  view_8 = add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_8: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_14: "f32[1, 1024, 1]" = var_mean_1[0]
        getitem_15: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_3: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_15);  convert_element_type_8 = getitem_15 = None
        add_7: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_2: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_3: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
        add_8: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None
        convert_element_type_9: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_8, torch.float16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_9: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_9, [-1, 1280]);  convert_element_type_9 = None
        addmm_2: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg11_1, view_9, arg12_1);  arg11_1 = view_9 = arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_10: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_2, [1, 1024, 5120]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        pow_1: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_5: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_10, mul_5);  view_10 = mul_5 = None
        mul_6: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_10: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_4, add_10);  mul_4 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_11: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_7, [-1, 5120]);  mul_7 = None
        addmm_3: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg13_1, view_11, arg14_1);  arg13_1 = view_11 = arg14_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_12: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_3, [1, 1024, 1280]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_11: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_6, view_12);  add_6 = view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_16: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_16, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 1024, 1]" = var_mean_2[0]
        getitem_17: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_4: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_17);  convert_element_type_16 = getitem_17 = None
        add_12: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_8: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
        add_13: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None
        convert_element_type_17: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_13, torch.float16);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_13: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_17, [-1, 1280]);  convert_element_type_17 = None
        addmm_4: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg17_1, view_13, arg18_1);  arg17_1 = view_13 = arg18_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_14: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_4, [1, 1024, 3840]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_14, 1280, 2);  view_14 = None
        getitem_18: "f16[1, 1024, 1280]" = split_1[0]
        getitem_19: "f16[1, 1024, 1280]" = split_1[1]
        getitem_20: "f16[1, 1024, 1280]" = split_1[2];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_17: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_18, [1, 1024, -1, 64]);  getitem_18 = None
        permute_6: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_15: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_19, [1, 1024, -1, 64]);  getitem_19 = None
        permute_4: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_16: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_20, [1, 1024, -1, 64]);  getitem_20 = None
        permute_5: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_4: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_6, permute_4, permute_5, where_1, False, scale = 0.125);  permute_6 = permute_4 = permute_5 = where_1 = None
        getitem_21: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_7, [1, 1024, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_18, [-1, 1280]);  view_18 = None
        addmm_5: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg19_1, view_19, arg20_1);  arg19_1 = view_19 = arg20_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_20: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_5, [1, 1024, 1280]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_14: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_20, add_11);  view_20 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_24: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 1024, 1]" = var_mean_3[0]
        getitem_31: "f32[1, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_31);  convert_element_type_24 = getitem_31 = None
        add_15: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_10: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_11: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_16: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None
        convert_element_type_25: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_16, torch.float16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_21: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_25, [-1, 1280]);  convert_element_type_25 = None
        addmm_6: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg23_1, view_21, arg24_1);  arg23_1 = view_21 = arg24_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_22: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_6, [1, 1024, 5120]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_2: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_13: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_22, mul_13);  view_22 = mul_13 = None
        mul_14: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_18: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_12, add_18);  mul_12 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_23: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_15, [-1, 5120]);  mul_15 = None
        addmm_7: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg25_1, view_23, arg26_1);  arg25_1 = view_23 = arg26_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_24: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_7, [1, 1024, 1280]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_19: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_14, view_24);  add_14 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_32: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_32, [2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 1024, 1]" = var_mean_4[0]
        getitem_33: "f32[1, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_32, getitem_33);  convert_element_type_32 = getitem_33 = None
        add_20: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_16: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_17: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_16, arg27_1);  mul_16 = arg27_1 = None
        add_21: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_17, arg28_1);  mul_17 = arg28_1 = None
        convert_element_type_33: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_21, torch.float16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_25: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_33, [-1, 1280]);  convert_element_type_33 = None
        addmm_8: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg29_1, view_25, arg30_1);  arg29_1 = view_25 = arg30_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_26: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_8, [1, 1024, 3840]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_26, 1280, 2);  view_26 = None
        getitem_34: "f16[1, 1024, 1280]" = split_2[0]
        getitem_35: "f16[1, 1024, 1280]" = split_2[1]
        getitem_36: "f16[1, 1024, 1280]" = split_2[2];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_29: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_34, [1, 1024, -1, 64]);  getitem_34 = None
        permute_10: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_27: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_35, [1, 1024, -1, 64]);  getitem_35 = None
        permute_8: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_28: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_36, [1, 1024, -1, 64]);  getitem_36 = None
        permute_9: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_6: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_10, permute_8, permute_9, where_2, False, scale = 0.125);  permute_10 = permute_8 = permute_9 = where_2 = None
        getitem_37: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3]);  getitem_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_11, [1, 1024, -1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_30, [-1, 1280]);  view_30 = None
        addmm_9: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg31_1, view_31, arg32_1);  arg31_1 = view_31 = arg32_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_32: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_9, [1, 1024, 1280]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_22: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_32, add_19);  view_32 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_40: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_22, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_46: "f32[1, 1024, 1]" = var_mean_5[0]
        getitem_47: "f32[1, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_7: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_47);  convert_element_type_40 = getitem_47 = None
        add_23: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_18: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_19: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_18, arg33_1);  mul_18 = arg33_1 = None
        add_24: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_19, arg34_1);  mul_19 = arg34_1 = None
        convert_element_type_41: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_24, torch.float16);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_33: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_41, [-1, 1280]);  convert_element_type_41 = None
        addmm_10: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg35_1, view_33, arg36_1);  arg35_1 = view_33 = arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_34: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_10, [1, 1024, 5120]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        pow_3: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_21: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_34, mul_21);  view_34 = mul_21 = None
        mul_22: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_26: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_20, add_26);  mul_20 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_35: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_23, [-1, 5120]);  mul_23 = None
        addmm_11: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg37_1, view_35, arg38_1);  arg37_1 = view_35 = arg38_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_36: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_11, [1, 1024, 1280]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_27: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_22, view_36);  add_22 = view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_48: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 1024, 1]" = var_mean_6[0]
        getitem_49: "f32[1, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_8: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_49);  convert_element_type_48 = getitem_49 = None
        add_28: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_24: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_25: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_24, arg39_1);  mul_24 = arg39_1 = None
        add_29: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_25, arg40_1);  mul_25 = arg40_1 = None
        convert_element_type_49: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_29, torch.float16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_37: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_49, [-1, 1280]);  convert_element_type_49 = None
        addmm_12: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg41_1, view_37, arg42_1);  arg41_1 = view_37 = arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_38: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_12, [1, 1024, 3840]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_38, 1280, 2);  view_38 = None
        getitem_50: "f16[1, 1024, 1280]" = split_3[0]
        getitem_51: "f16[1, 1024, 1280]" = split_3[1]
        getitem_52: "f16[1, 1024, 1280]" = split_3[2];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_41: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_50, [1, 1024, -1, 64]);  getitem_50 = None
        permute_14: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_39: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_51, [1, 1024, -1, 64]);  getitem_51 = None
        permute_12: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_40: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_52, [1, 1024, -1, 64]);  getitem_52 = None
        permute_13: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_8: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_14, permute_12, permute_13, where_3, False, scale = 0.125);  permute_14 = permute_12 = permute_13 = where_3 = None
        getitem_53: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_15, [1, 1024, -1]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_42, [-1, 1280]);  view_42 = None
        addmm_13: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg43_1, view_43, arg44_1);  arg43_1 = view_43 = arg44_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_44: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_13, [1, 1024, 1280]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_30: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_44, add_27);  view_44 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_56: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_56, [2], correction = 0, keepdim = True)
        getitem_62: "f32[1, 1024, 1]" = var_mean_7[0]
        getitem_63: "f32[1, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_9: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_56, getitem_63);  convert_element_type_56 = getitem_63 = None
        add_31: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_26: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = rsqrt_7 = None
        mul_27: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_26, arg45_1);  mul_26 = arg45_1 = None
        add_32: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_27, arg46_1);  mul_27 = arg46_1 = None
        convert_element_type_57: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_32, torch.float16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_45: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_57, [-1, 1280]);  convert_element_type_57 = None
        addmm_14: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg47_1, view_45, arg48_1);  arg47_1 = view_45 = arg48_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_46: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_14, [1, 1024, 5120]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        pow_4: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_29: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_33: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_46, mul_29);  view_46 = mul_29 = None
        mul_30: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_3: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_34: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_28, add_34);  mul_28 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_47: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_31, [-1, 5120]);  mul_31 = None
        addmm_15: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg49_1, view_47, arg50_1);  arg49_1 = view_47 = arg50_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_48: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_15, [1, 1024, 1280]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_35: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_30, view_48);  add_30 = view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_64: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_64, [2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 1024, 1]" = var_mean_8[0]
        getitem_65: "f32[1, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_10: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_64, getitem_65);  convert_element_type_64 = getitem_65 = None
        add_36: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_32: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = rsqrt_8 = None
        mul_33: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_32, arg51_1);  mul_32 = arg51_1 = None
        add_37: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_33, arg52_1);  mul_33 = arg52_1 = None
        convert_element_type_65: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_49: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_65, [-1, 1280]);  convert_element_type_65 = None
        addmm_16: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg53_1, view_49, arg54_1);  arg53_1 = view_49 = arg54_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_50: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_16, [1, 1024, 3840]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_50, 1280, 2);  view_50 = None
        getitem_66: "f16[1, 1024, 1280]" = split_4[0]
        getitem_67: "f16[1, 1024, 1280]" = split_4[1]
        getitem_68: "f16[1, 1024, 1280]" = split_4[2];  split_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_53: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_66, [1, 1024, -1, 64]);  getitem_66 = None
        permute_18: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_51: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_67, [1, 1024, -1, 64]);  getitem_67 = None
        permute_16: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_52: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_68, [1, 1024, -1, 64]);  getitem_68 = None
        permute_17: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_10: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_18, permute_16, permute_17, where_4, False, scale = 0.125);  permute_18 = permute_16 = permute_17 = where_4 = None
        getitem_69: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3]);  getitem_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_19, [1, 1024, -1]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_54, [-1, 1280]);  view_54 = None
        addmm_17: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg55_1, view_55, arg56_1);  arg55_1 = view_55 = arg56_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_56: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_17, [1, 1024, 1280]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_38: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_56, add_35);  view_56 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_72: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_78: "f32[1, 1024, 1]" = var_mean_9[0]
        getitem_79: "f32[1, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_11: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_79);  convert_element_type_72 = getitem_79 = None
        add_39: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_34: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = rsqrt_9 = None
        mul_35: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_34, arg57_1);  mul_34 = arg57_1 = None
        add_40: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_35, arg58_1);  mul_35 = arg58_1 = None
        convert_element_type_73: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_40, torch.float16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_57: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_73, [-1, 1280]);  convert_element_type_73 = None
        addmm_18: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg59_1, view_57, arg60_1);  arg59_1 = view_57 = arg60_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_58: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_18, [1, 1024, 5120]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        pow_5: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_37: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_58, mul_37);  view_58 = mul_37 = None
        mul_38: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_42: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_36, add_42);  mul_36 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_59: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_39, [-1, 5120]);  mul_39 = None
        addmm_19: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg61_1, view_59, arg62_1);  arg61_1 = view_59 = arg62_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_60: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_19, [1, 1024, 1280]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_43: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_38, view_60);  add_38 = view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_80: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_43, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True)
        getitem_80: "f32[1, 1024, 1]" = var_mean_10[0]
        getitem_81: "f32[1, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_12: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_81);  convert_element_type_80 = getitem_81 = None
        add_44: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_40: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = rsqrt_10 = None
        mul_41: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_40, arg63_1);  mul_40 = arg63_1 = None
        add_45: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_41, arg64_1);  mul_41 = arg64_1 = None
        convert_element_type_81: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_45, torch.float16);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_61: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_81, [-1, 1280]);  convert_element_type_81 = None
        addmm_20: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg65_1, view_61, arg66_1);  arg65_1 = view_61 = arg66_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_62: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_20, [1, 1024, 3840]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_62, 1280, 2);  view_62 = None
        getitem_82: "f16[1, 1024, 1280]" = split_5[0]
        getitem_83: "f16[1, 1024, 1280]" = split_5[1]
        getitem_84: "f16[1, 1024, 1280]" = split_5[2];  split_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_65: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_82, [1, 1024, -1, 64]);  getitem_82 = None
        permute_22: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_63: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_83, [1, 1024, -1, 64]);  getitem_83 = None
        permute_20: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_64: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_84, [1, 1024, -1, 64]);  getitem_84 = None
        permute_21: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_12: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_22, permute_20, permute_21, where_5, False, scale = 0.125);  permute_22 = permute_20 = permute_21 = where_5 = None
        getitem_85: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_23, [1, 1024, -1]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_66, [-1, 1280]);  view_66 = None
        addmm_21: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg67_1, view_67, arg68_1);  arg67_1 = view_67 = arg68_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_68: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_21, [1, 1024, 1280]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_46: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_68, add_43);  view_68 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_88: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_46, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_88, [2], correction = 0, keepdim = True)
        getitem_94: "f32[1, 1024, 1]" = var_mean_11[0]
        getitem_95: "f32[1, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_13: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_88, getitem_95);  convert_element_type_88 = getitem_95 = None
        add_47: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_42: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = rsqrt_11 = None
        mul_43: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_42, arg69_1);  mul_42 = arg69_1 = None
        add_48: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_43, arg70_1);  mul_43 = arg70_1 = None
        convert_element_type_89: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_48, torch.float16);  add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_69: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_89, [-1, 1280]);  convert_element_type_89 = None
        addmm_22: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg71_1, view_69, arg72_1);  arg71_1 = view_69 = arg72_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_70: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_22, [1, 1024, 5120]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        pow_6: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_45: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_49: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_70, mul_45);  view_70 = mul_45 = None
        mul_46: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_5: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_50: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_44, add_50);  mul_44 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_71: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_47, [-1, 5120]);  mul_47 = None
        addmm_23: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg73_1, view_71, arg74_1);  arg73_1 = view_71 = arg74_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_72: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_23, [1, 1024, 1280]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_51: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_46, view_72);  add_46 = view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_96: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_96: "f32[1, 1024, 1]" = var_mean_12[0]
        getitem_97: "f32[1, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_14: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_97);  convert_element_type_96 = getitem_97 = None
        add_52: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_48: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = rsqrt_12 = None
        mul_49: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_53: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None
        convert_element_type_97: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_53, torch.float16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_73: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_97, [-1, 1280]);  convert_element_type_97 = None
        addmm_24: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg77_1, view_73, arg78_1);  arg77_1 = view_73 = arg78_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_74: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_24, [1, 1024, 3840]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_6 = torch.ops.aten.split.Tensor(view_74, 1280, 2);  view_74 = None
        getitem_98: "f16[1, 1024, 1280]" = split_6[0]
        getitem_99: "f16[1, 1024, 1280]" = split_6[1]
        getitem_100: "f16[1, 1024, 1280]" = split_6[2];  split_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_77: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_98, [1, 1024, -1, 64]);  getitem_98 = None
        permute_26: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_75: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_99, [1, 1024, -1, 64]);  getitem_99 = None
        permute_24: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_76: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_100, [1, 1024, -1, 64]);  getitem_100 = None
        permute_25: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_14: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_26, permute_24, permute_25, where_6, False, scale = 0.125);  permute_26 = permute_24 = permute_25 = where_6 = None
        getitem_101: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_78: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_27, [1, 1024, -1]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_79: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_78, [-1, 1280]);  view_78 = None
        addmm_25: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg79_1, view_79, arg80_1);  arg79_1 = view_79 = arg80_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_80: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_25, [1, 1024, 1280]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_54: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_80, add_51);  view_80 = add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_104: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_54, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_104, [2], correction = 0, keepdim = True)
        getitem_110: "f32[1, 1024, 1]" = var_mean_13[0]
        getitem_111: "f32[1, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_15: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_104, getitem_111);  convert_element_type_104 = getitem_111 = None
        add_55: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_50: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = rsqrt_13 = None
        mul_51: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_50, arg81_1);  mul_50 = arg81_1 = None
        add_56: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_51, arg82_1);  mul_51 = arg82_1 = None
        convert_element_type_105: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_56, torch.float16);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_81: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_105, [-1, 1280]);  convert_element_type_105 = None
        addmm_26: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg83_1, view_81, arg84_1);  arg83_1 = view_81 = arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_82: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_26, [1, 1024, 5120]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
        pow_7: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 3.0)
        mul_53: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_57: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_82, mul_53);  view_82 = mul_53 = None
        mul_54: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_6: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_58: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_52, add_58);  mul_52 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_83: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_55, [-1, 5120]);  mul_55 = None
        addmm_27: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg85_1, view_83, arg86_1);  arg85_1 = view_83 = arg86_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_84: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_27, [1, 1024, 1280]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_59: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_54, view_84);  add_54 = view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_112: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_112, [2], correction = 0, keepdim = True)
        getitem_112: "f32[1, 1024, 1]" = var_mean_14[0]
        getitem_113: "f32[1, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_16: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_112, getitem_113);  convert_element_type_112 = getitem_113 = None
        add_60: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_56: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = rsqrt_14 = None
        mul_57: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_56, arg87_1);  mul_56 = arg87_1 = None
        add_61: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_57, arg88_1);  mul_57 = arg88_1 = None
        convert_element_type_113: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_61, torch.float16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_85: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_113, [-1, 1280]);  convert_element_type_113 = None
        addmm_28: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg89_1, view_85, arg90_1);  arg89_1 = view_85 = arg90_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_86: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_28, [1, 1024, 3840]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_7 = torch.ops.aten.split.Tensor(view_86, 1280, 2);  view_86 = None
        getitem_114: "f16[1, 1024, 1280]" = split_7[0]
        getitem_115: "f16[1, 1024, 1280]" = split_7[1]
        getitem_116: "f16[1, 1024, 1280]" = split_7[2];  split_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_89: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_114, [1, 1024, -1, 64]);  getitem_114 = None
        permute_30: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_87: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_115, [1, 1024, -1, 64]);  getitem_115 = None
        permute_28: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_88: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_116, [1, 1024, -1, 64]);  getitem_116 = None
        permute_29: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_16: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_30, permute_28, permute_29, where_7, False, scale = 0.125);  permute_30 = permute_28 = permute_29 = where_7 = None
        getitem_117: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_90: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_31, [1, 1024, -1]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_91: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_90, [-1, 1280]);  view_90 = None
        addmm_29: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg91_1, view_91, arg92_1);  arg91_1 = view_91 = arg92_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_92: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_29, [1, 1024, 1280]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_62: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_92, add_59);  view_92 = add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_120: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_126: "f32[1, 1024, 1]" = var_mean_15[0]
        getitem_127: "f32[1, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_17: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_127);  convert_element_type_120 = getitem_127 = None
        add_63: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
        rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_58: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_15);  sub_17 = rsqrt_15 = None
        mul_59: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_58, arg93_1);  mul_58 = arg93_1 = None
        add_64: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_59, arg94_1);  mul_59 = arg94_1 = None
        convert_element_type_121: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_64, torch.float16);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_93: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_121, [-1, 1280]);  convert_element_type_121 = None
        addmm_30: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg95_1, view_93, arg96_1);  arg95_1 = view_93 = arg96_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_94: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_30, [1, 1024, 5120]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
        pow_8: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_94, 3.0)
        mul_61: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_65: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_94, mul_61);  view_94 = mul_61 = None
        mul_62: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_7: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_66: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_60, add_66);  mul_60 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_95: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_63, [-1, 5120]);  mul_63 = None
        addmm_31: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg97_1, view_95, arg98_1);  arg97_1 = view_95 = arg98_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_96: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_31, [1, 1024, 1280]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_67: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_62, view_96);  add_62 = view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_128: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_67, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_128, [2], correction = 0, keepdim = True)
        getitem_128: "f32[1, 1024, 1]" = var_mean_16[0]
        getitem_129: "f32[1, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_18: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_128, getitem_129);  convert_element_type_128 = getitem_129 = None
        add_68: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_64: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_16);  sub_18 = rsqrt_16 = None
        mul_65: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_64, arg99_1);  mul_64 = arg99_1 = None
        add_69: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_65, arg100_1);  mul_65 = arg100_1 = None
        convert_element_type_129: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_69, torch.float16);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_97: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_129, [-1, 1280]);  convert_element_type_129 = None
        addmm_32: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg101_1, view_97, arg102_1);  arg101_1 = view_97 = arg102_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_98: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_32, [1, 1024, 3840]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_8 = torch.ops.aten.split.Tensor(view_98, 1280, 2);  view_98 = None
        getitem_130: "f16[1, 1024, 1280]" = split_8[0]
        getitem_131: "f16[1, 1024, 1280]" = split_8[1]
        getitem_132: "f16[1, 1024, 1280]" = split_8[2];  split_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_101: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_130, [1, 1024, -1, 64]);  getitem_130 = None
        permute_34: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_99: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_131, [1, 1024, -1, 64]);  getitem_131 = None
        permute_32: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_100: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_132, [1, 1024, -1, 64]);  getitem_132 = None
        permute_33: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_18: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_18, full_default_17);  full_default_18 = full_default_17 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_34, permute_32, permute_33, where_8, False, scale = 0.125);  permute_34 = permute_32 = permute_33 = where_8 = None
        getitem_133: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_102: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_35, [1, 1024, -1]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_103: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_102, [-1, 1280]);  view_102 = None
        addmm_33: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg103_1, view_103, arg104_1);  arg103_1 = view_103 = arg104_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_104: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_33, [1, 1024, 1280]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_70: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_104, add_67);  view_104 = add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_136: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_70, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_136, [2], correction = 0, keepdim = True)
        getitem_142: "f32[1, 1024, 1]" = var_mean_17[0]
        getitem_143: "f32[1, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_19: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_136, getitem_143);  convert_element_type_136 = getitem_143 = None
        add_71: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
        rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_66: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_17);  sub_19 = rsqrt_17 = None
        mul_67: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_66, arg105_1);  mul_66 = arg105_1 = None
        add_72: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_67, arg106_1);  mul_67 = arg106_1 = None
        convert_element_type_137: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_72, torch.float16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_105: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_137, [-1, 1280]);  convert_element_type_137 = None
        addmm_34: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg107_1, view_105, arg108_1);  arg107_1 = view_105 = arg108_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_106: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_34, [1, 1024, 5120]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        pow_9: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 3.0)
        mul_69: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_73: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_106, mul_69);  view_106 = mul_69 = None
        mul_70: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_8: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_74: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_68, add_74);  mul_68 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_107: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_71, [-1, 5120]);  mul_71 = None
        addmm_35: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg109_1, view_107, arg110_1);  arg109_1 = view_107 = arg110_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_108: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_35, [1, 1024, 1280]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_75: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_70, view_108);  add_70 = view_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_144: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_75, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_144: "f32[1, 1024, 1]" = var_mean_18[0]
        getitem_145: "f32[1, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_20: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_145);  convert_element_type_144 = getitem_145 = None
        add_76: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_72: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_18);  sub_20 = rsqrt_18 = None
        mul_73: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_72, arg111_1);  mul_72 = arg111_1 = None
        add_77: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_73, arg112_1);  mul_73 = arg112_1 = None
        convert_element_type_145: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_77, torch.float16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_109: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_145, [-1, 1280]);  convert_element_type_145 = None
        addmm_36: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg113_1, view_109, arg114_1);  arg113_1 = view_109 = arg114_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_110: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_36, [1, 1024, 3840]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_9 = torch.ops.aten.split.Tensor(view_110, 1280, 2);  view_110 = None
        getitem_146: "f16[1, 1024, 1280]" = split_9[0]
        getitem_147: "f16[1, 1024, 1280]" = split_9[1]
        getitem_148: "f16[1, 1024, 1280]" = split_9[2];  split_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_113: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_146, [1, 1024, -1, 64]);  getitem_146 = None
        permute_38: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_111: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_147, [1, 1024, -1, 64]);  getitem_147 = None
        permute_36: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_112: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_148, [1, 1024, -1, 64]);  getitem_148 = None
        permute_37: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_20: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_20, full_default_19);  full_default_20 = full_default_19 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_38, permute_36, permute_37, where_9, False, scale = 0.125);  permute_38 = permute_36 = permute_37 = where_9 = None
        getitem_149: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3]);  getitem_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_114: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_39, [1, 1024, -1]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_115: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_114, [-1, 1280]);  view_114 = None
        addmm_37: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg115_1, view_115, arg116_1);  arg115_1 = view_115 = arg116_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_116: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_37, [1, 1024, 1280]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_78: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_116, add_75);  view_116 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_152: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_78, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_152, [2], correction = 0, keepdim = True)
        getitem_158: "f32[1, 1024, 1]" = var_mean_19[0]
        getitem_159: "f32[1, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_21: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_152, getitem_159);  convert_element_type_152 = getitem_159 = None
        add_79: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_74: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_19);  sub_21 = rsqrt_19 = None
        mul_75: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_74, arg117_1);  mul_74 = arg117_1 = None
        add_80: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_75, arg118_1);  mul_75 = arg118_1 = None
        convert_element_type_153: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_80, torch.float16);  add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_117: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_153, [-1, 1280]);  convert_element_type_153 = None
        addmm_38: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg119_1, view_117, arg120_1);  arg119_1 = view_117 = arg120_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_118: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_38, [1, 1024, 5120]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        pow_10: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_118, 3.0)
        mul_77: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_81: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_118, mul_77);  view_118 = mul_77 = None
        mul_78: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_9: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_82: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_76, add_82);  mul_76 = add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_119: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_79, [-1, 5120]);  mul_79 = None
        addmm_39: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg121_1, view_119, arg122_1);  arg121_1 = view_119 = arg122_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_120: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_39, [1, 1024, 1280]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_83: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_78, view_120);  add_78 = view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_160: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_160, [2], correction = 0, keepdim = True)
        getitem_160: "f32[1, 1024, 1]" = var_mean_20[0]
        getitem_161: "f32[1, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_22: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_160, getitem_161);  convert_element_type_160 = getitem_161 = None
        add_84: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_80: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_20);  sub_22 = rsqrt_20 = None
        mul_81: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_80, arg123_1);  mul_80 = arg123_1 = None
        add_85: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_81, arg124_1);  mul_81 = arg124_1 = None
        convert_element_type_161: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_85, torch.float16);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_121: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_161, [-1, 1280]);  convert_element_type_161 = None
        addmm_40: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg125_1, view_121, arg126_1);  arg125_1 = view_121 = arg126_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_122: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_40, [1, 1024, 3840]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_10 = torch.ops.aten.split.Tensor(view_122, 1280, 2);  view_122 = None
        getitem_162: "f16[1, 1024, 1280]" = split_10[0]
        getitem_163: "f16[1, 1024, 1280]" = split_10[1]
        getitem_164: "f16[1, 1024, 1280]" = split_10[2];  split_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_125: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_162, [1, 1024, -1, 64]);  getitem_162 = None
        permute_42: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_123: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_163, [1, 1024, -1, 64]);  getitem_163 = None
        permute_40: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_124: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_164, [1, 1024, -1, 64]);  getitem_164 = None
        permute_41: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_22: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_42, permute_40, permute_41, where_10, False, scale = 0.125);  permute_42 = permute_40 = permute_41 = where_10 = None
        getitem_165: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_126: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_43, [1, 1024, -1]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_127: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_126, [-1, 1280]);  view_126 = None
        addmm_41: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg127_1, view_127, arg128_1);  arg127_1 = view_127 = arg128_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_128: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_41, [1, 1024, 1280]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_86: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_128, add_83);  view_128 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_168: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_86, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_168, [2], correction = 0, keepdim = True)
        getitem_174: "f32[1, 1024, 1]" = var_mean_21[0]
        getitem_175: "f32[1, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_23: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_168, getitem_175);  convert_element_type_168 = getitem_175 = None
        add_87: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_82: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_21);  sub_23 = rsqrt_21 = None
        mul_83: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_82, arg129_1);  mul_82 = arg129_1 = None
        add_88: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_83, arg130_1);  mul_83 = arg130_1 = None
        convert_element_type_169: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_88, torch.float16);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_129: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_169, [-1, 1280]);  convert_element_type_169 = None
        addmm_42: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg131_1, view_129, arg132_1);  arg131_1 = view_129 = arg132_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_130: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_42, [1, 1024, 5120]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_130, 0.5)
        pow_11: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_130, 3.0)
        mul_85: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_89: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_130, mul_85);  view_130 = mul_85 = None
        mul_86: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_10: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_90: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_84, add_90);  mul_84 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_131: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_87, [-1, 5120]);  mul_87 = None
        addmm_43: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg133_1, view_131, arg134_1);  arg133_1 = view_131 = arg134_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_132: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_43, [1, 1024, 1280]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_91: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_86, view_132);  add_86 = view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_176: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_91, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_176, [2], correction = 0, keepdim = True)
        getitem_176: "f32[1, 1024, 1]" = var_mean_22[0]
        getitem_177: "f32[1, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_24: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_176, getitem_177);  convert_element_type_176 = getitem_177 = None
        add_92: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_88: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_22);  sub_24 = rsqrt_22 = None
        mul_89: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_88, arg135_1);  mul_88 = arg135_1 = None
        add_93: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_89, arg136_1);  mul_89 = arg136_1 = None
        convert_element_type_177: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_93, torch.float16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_133: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_177, [-1, 1280]);  convert_element_type_177 = None
        addmm_44: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg137_1, view_133, arg138_1);  arg137_1 = view_133 = arg138_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_134: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_44, [1, 1024, 3840]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_11 = torch.ops.aten.split.Tensor(view_134, 1280, 2);  view_134 = None
        getitem_178: "f16[1, 1024, 1280]" = split_11[0]
        getitem_179: "f16[1, 1024, 1280]" = split_11[1]
        getitem_180: "f16[1, 1024, 1280]" = split_11[2];  split_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_137: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_178, [1, 1024, -1, 64]);  getitem_178 = None
        permute_46: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_135: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_179, [1, 1024, -1, 64]);  getitem_179 = None
        permute_44: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_136: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_180, [1, 1024, -1, 64]);  getitem_180 = None
        permute_45: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_24: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_24, full_default_23);  full_default_24 = full_default_23 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_46, permute_44, permute_45, where_11, False, scale = 0.125);  permute_46 = permute_44 = permute_45 = where_11 = None
        getitem_181: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_138: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_47, [1, 1024, -1]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_139: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_138, [-1, 1280]);  view_138 = None
        addmm_45: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg139_1, view_139, arg140_1);  arg139_1 = view_139 = arg140_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_140: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_45, [1, 1024, 1280]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_94: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_140, add_91);  view_140 = add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_184: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_94, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_184, [2], correction = 0, keepdim = True)
        getitem_190: "f32[1, 1024, 1]" = var_mean_23[0]
        getitem_191: "f32[1, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_25: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_184, getitem_191);  convert_element_type_184 = getitem_191 = None
        add_95: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_90: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_23);  sub_25 = rsqrt_23 = None
        mul_91: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_90, arg141_1);  mul_90 = arg141_1 = None
        add_96: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_91, arg142_1);  mul_91 = arg142_1 = None
        convert_element_type_185: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_96, torch.float16);  add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_141: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_185, [-1, 1280]);  convert_element_type_185 = None
        addmm_46: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg143_1, view_141, arg144_1);  arg143_1 = view_141 = arg144_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_142: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_46, [1, 1024, 5120]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        pow_12: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 3.0)
        mul_93: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_97: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_142, mul_93);  view_142 = mul_93 = None
        mul_94: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_11: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_98: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_92, add_98);  mul_92 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_143: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_95, [-1, 5120]);  mul_95 = None
        addmm_47: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg145_1, view_143, arg146_1);  arg145_1 = view_143 = arg146_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_144: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_47, [1, 1024, 1280]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_99: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_94, view_144);  add_94 = view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_192: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_99, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_192, [2], correction = 0, keepdim = True)
        getitem_192: "f32[1, 1024, 1]" = var_mean_24[0]
        getitem_193: "f32[1, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_26: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_192, getitem_193);  convert_element_type_192 = getitem_193 = None
        add_100: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_24: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_96: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_24);  sub_26 = rsqrt_24 = None
        mul_97: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_96, arg147_1);  mul_96 = arg147_1 = None
        add_101: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_97, arg148_1);  mul_97 = arg148_1 = None
        convert_element_type_193: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_101, torch.float16);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_145: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_193, [-1, 1280]);  convert_element_type_193 = None
        addmm_48: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg149_1, view_145, arg150_1);  arg149_1 = view_145 = arg150_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_146: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_48, [1, 1024, 3840]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_12 = torch.ops.aten.split.Tensor(view_146, 1280, 2);  view_146 = None
        getitem_194: "f16[1, 1024, 1280]" = split_12[0]
        getitem_195: "f16[1, 1024, 1280]" = split_12[1]
        getitem_196: "f16[1, 1024, 1280]" = split_12[2];  split_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_149: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_194, [1, 1024, -1, 64]);  getitem_194 = None
        permute_50: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_147: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_195, [1, 1024, -1, 64]);  getitem_195 = None
        permute_48: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_148: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_196, [1, 1024, -1, 64]);  getitem_196 = None
        permute_49: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_26: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_26, full_default_25);  full_default_26 = full_default_25 = None
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_50, permute_48, permute_49, where_12, False, scale = 0.125);  permute_50 = permute_48 = permute_49 = where_12 = None
        getitem_197: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_12[0];  _scaled_dot_product_cudnn_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_197, [0, 2, 1, 3]);  getitem_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_150: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_51, [1, 1024, -1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_151: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_150, [-1, 1280]);  view_150 = None
        addmm_49: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg151_1, view_151, arg152_1);  arg151_1 = view_151 = arg152_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_152: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_49, [1, 1024, 1280]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_102: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_152, add_99);  view_152 = add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_200: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_102, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_200, [2], correction = 0, keepdim = True)
        getitem_206: "f32[1, 1024, 1]" = var_mean_25[0]
        getitem_207: "f32[1, 1024, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_27: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_200, getitem_207);  convert_element_type_200 = getitem_207 = None
        add_103: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_25: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_98: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_25);  sub_27 = rsqrt_25 = None
        mul_99: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_98, arg153_1);  mul_98 = arg153_1 = None
        add_104: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_99, arg154_1);  mul_99 = arg154_1 = None
        convert_element_type_201: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_104, torch.float16);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_153: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_201, [-1, 1280]);  convert_element_type_201 = None
        addmm_50: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg155_1, view_153, arg156_1);  arg155_1 = view_153 = arg156_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_154: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_50, [1, 1024, 5120]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_100: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_154, 0.5)
        pow_13: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_154, 3.0)
        mul_101: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_105: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_154, mul_101);  view_154 = mul_101 = None
        mul_102: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_105, 0.7978845608028654);  add_105 = None
        tanh_12: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_102);  mul_102 = None
        add_106: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_103: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_100, add_106);  mul_100 = add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_155: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_103, [-1, 5120]);  mul_103 = None
        addmm_51: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg157_1, view_155, arg158_1);  arg157_1 = view_155 = arg158_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_156: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_51, [1, 1024, 1280]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_107: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_102, view_156);  add_102 = view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_208: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_208, [2], correction = 0, keepdim = True)
        getitem_208: "f32[1, 1024, 1]" = var_mean_26[0]
        getitem_209: "f32[1, 1024, 1]" = var_mean_26[1];  var_mean_26 = None
        sub_28: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_208, getitem_209);  convert_element_type_208 = getitem_209 = None
        add_108: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_26: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_104: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_26);  sub_28 = rsqrt_26 = None
        mul_105: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_104, arg159_1);  mul_104 = arg159_1 = None
        add_109: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_105, arg160_1);  mul_105 = arg160_1 = None
        convert_element_type_209: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_109, torch.float16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_157: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_209, [-1, 1280]);  convert_element_type_209 = None
        addmm_52: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg161_1, view_157, arg162_1);  arg161_1 = view_157 = arg162_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_158: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_52, [1, 1024, 3840]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_13 = torch.ops.aten.split.Tensor(view_158, 1280, 2);  view_158 = None
        getitem_210: "f16[1, 1024, 1280]" = split_13[0]
        getitem_211: "f16[1, 1024, 1280]" = split_13[1]
        getitem_212: "f16[1, 1024, 1280]" = split_13[2];  split_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_161: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_210, [1, 1024, -1, 64]);  getitem_210 = None
        permute_54: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_159: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_211, [1, 1024, -1, 64]);  getitem_211 = None
        permute_52: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_160: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_212, [1, 1024, -1, 64]);  getitem_212 = None
        permute_53: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_28: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_28, full_default_27);  full_default_28 = full_default_27 = None
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_54, permute_52, permute_53, where_13, False, scale = 0.125);  permute_54 = permute_52 = permute_53 = where_13 = None
        getitem_213: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_13[0];  _scaled_dot_product_cudnn_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_55: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_213, [0, 2, 1, 3]);  getitem_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_162: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_55, [1, 1024, -1]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_163: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_162, [-1, 1280]);  view_162 = None
        addmm_53: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg163_1, view_163, arg164_1);  arg163_1 = view_163 = arg164_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_164: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_53, [1, 1024, 1280]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_110: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_164, add_107);  view_164 = add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_216: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_110, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_216, [2], correction = 0, keepdim = True)
        getitem_222: "f32[1, 1024, 1]" = var_mean_27[0]
        getitem_223: "f32[1, 1024, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_29: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_216, getitem_223);  convert_element_type_216 = getitem_223 = None
        add_111: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-05);  getitem_222 = None
        rsqrt_27: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_106: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_27);  sub_29 = rsqrt_27 = None
        mul_107: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_106, arg165_1);  mul_106 = arg165_1 = None
        add_112: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_107, arg166_1);  mul_107 = arg166_1 = None
        convert_element_type_217: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_112, torch.float16);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_165: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_217, [-1, 1280]);  convert_element_type_217 = None
        addmm_54: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg167_1, view_165, arg168_1);  arg167_1 = view_165 = arg168_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_166: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_54, [1, 1024, 5120]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_108: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_166, 0.5)
        pow_14: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_166, 3.0)
        mul_109: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_113: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_166, mul_109);  view_166 = mul_109 = None
        mul_110: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_113, 0.7978845608028654);  add_113 = None
        tanh_13: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_110);  mul_110 = None
        add_114: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_111: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_108, add_114);  mul_108 = add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_167: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_111, [-1, 5120]);  mul_111 = None
        addmm_55: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg169_1, view_167, arg170_1);  arg169_1 = view_167 = arg170_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_168: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_55, [1, 1024, 1280]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_115: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_110, view_168);  add_110 = view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_224: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_115, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_224, [2], correction = 0, keepdim = True)
        getitem_224: "f32[1, 1024, 1]" = var_mean_28[0]
        getitem_225: "f32[1, 1024, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_30: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_224, getitem_225);  convert_element_type_224 = getitem_225 = None
        add_116: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_28: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_112: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_28);  sub_30 = rsqrt_28 = None
        mul_113: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_112, arg171_1);  mul_112 = arg171_1 = None
        add_117: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_113, arg172_1);  mul_113 = arg172_1 = None
        convert_element_type_225: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_117, torch.float16);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_169: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_225, [-1, 1280]);  convert_element_type_225 = None
        addmm_56: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg173_1, view_169, arg174_1);  arg173_1 = view_169 = arg174_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_170: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_56, [1, 1024, 3840]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_14 = torch.ops.aten.split.Tensor(view_170, 1280, 2);  view_170 = None
        getitem_226: "f16[1, 1024, 1280]" = split_14[0]
        getitem_227: "f16[1, 1024, 1280]" = split_14[1]
        getitem_228: "f16[1, 1024, 1280]" = split_14[2];  split_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_173: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_226, [1, 1024, -1, 64]);  getitem_226 = None
        permute_58: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_171: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_227, [1, 1024, -1, 64]);  getitem_227 = None
        permute_56: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_172: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_228, [1, 1024, -1, 64]);  getitem_228 = None
        permute_57: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_30: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_30, full_default_29);  full_default_30 = full_default_29 = None
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_58, permute_56, permute_57, where_14, False, scale = 0.125);  permute_58 = permute_56 = permute_57 = where_14 = None
        getitem_229: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_14[0];  _scaled_dot_product_cudnn_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_59: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_229, [0, 2, 1, 3]);  getitem_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_174: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_59, [1, 1024, -1]);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_175: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_174, [-1, 1280]);  view_174 = None
        addmm_57: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg175_1, view_175, arg176_1);  arg175_1 = view_175 = arg176_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_176: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_57, [1, 1024, 1280]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_118: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_176, add_115);  view_176 = add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_232: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_118, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_232, [2], correction = 0, keepdim = True)
        getitem_238: "f32[1, 1024, 1]" = var_mean_29[0]
        getitem_239: "f32[1, 1024, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_31: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_232, getitem_239);  convert_element_type_232 = getitem_239 = None
        add_119: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_238, 1e-05);  getitem_238 = None
        rsqrt_29: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_114: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_29);  sub_31 = rsqrt_29 = None
        mul_115: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_114, arg177_1);  mul_114 = arg177_1 = None
        add_120: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_115, arg178_1);  mul_115 = arg178_1 = None
        convert_element_type_233: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_120, torch.float16);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_177: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_233, [-1, 1280]);  convert_element_type_233 = None
        addmm_58: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg179_1, view_177, arg180_1);  arg179_1 = view_177 = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_178: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_58, [1, 1024, 5120]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
        pow_15: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_178, 3.0)
        mul_117: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_121: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_178, mul_117);  view_178 = mul_117 = None
        mul_118: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_121, 0.7978845608028654);  add_121 = None
        tanh_14: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_122: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_119: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_116, add_122);  mul_116 = add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_179: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_119, [-1, 5120]);  mul_119 = None
        addmm_59: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg181_1, view_179, arg182_1);  arg181_1 = view_179 = arg182_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_180: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_59, [1, 1024, 1280]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_123: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_118, view_180);  add_118 = view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_240: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_123, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_240, [2], correction = 0, keepdim = True)
        getitem_240: "f32[1, 1024, 1]" = var_mean_30[0]
        getitem_241: "f32[1, 1024, 1]" = var_mean_30[1];  var_mean_30 = None
        sub_32: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_240, getitem_241);  convert_element_type_240 = getitem_241 = None
        add_124: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_30: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_120: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_30);  sub_32 = rsqrt_30 = None
        mul_121: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_120, arg183_1);  mul_120 = arg183_1 = None
        add_125: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_121, arg184_1);  mul_121 = arg184_1 = None
        convert_element_type_241: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_125, torch.float16);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_181: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_241, [-1, 1280]);  convert_element_type_241 = None
        addmm_60: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg185_1, view_181, arg186_1);  arg185_1 = view_181 = arg186_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_182: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_60, [1, 1024, 3840]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_15 = torch.ops.aten.split.Tensor(view_182, 1280, 2);  view_182 = None
        getitem_242: "f16[1, 1024, 1280]" = split_15[0]
        getitem_243: "f16[1, 1024, 1280]" = split_15[1]
        getitem_244: "f16[1, 1024, 1280]" = split_15[2];  split_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_185: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_242, [1, 1024, -1, 64]);  getitem_242 = None
        permute_62: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_183: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_243, [1, 1024, -1, 64]);  getitem_243 = None
        permute_60: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_184: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_244, [1, 1024, -1, 64]);  getitem_244 = None
        permute_61: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_32: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_32, full_default_31);  full_default_32 = full_default_31 = None
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_62, permute_60, permute_61, where_15, False, scale = 0.125);  permute_62 = permute_60 = permute_61 = where_15 = None
        getitem_245: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_15[0];  _scaled_dot_product_cudnn_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_245, [0, 2, 1, 3]);  getitem_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_186: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_63, [1, 1024, -1]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_187: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_186, [-1, 1280]);  view_186 = None
        addmm_61: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg187_1, view_187, arg188_1);  arg187_1 = view_187 = arg188_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_188: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_61, [1, 1024, 1280]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_126: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_188, add_123);  view_188 = add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_248: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_126, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_248, [2], correction = 0, keepdim = True)
        getitem_254: "f32[1, 1024, 1]" = var_mean_31[0]
        getitem_255: "f32[1, 1024, 1]" = var_mean_31[1];  var_mean_31 = None
        sub_33: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_248, getitem_255);  convert_element_type_248 = getitem_255 = None
        add_127: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-05);  getitem_254 = None
        rsqrt_31: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        mul_122: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_31);  sub_33 = rsqrt_31 = None
        mul_123: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_122, arg189_1);  mul_122 = arg189_1 = None
        add_128: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_123, arg190_1);  mul_123 = arg190_1 = None
        convert_element_type_249: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_128, torch.float16);  add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_189: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_249, [-1, 1280]);  convert_element_type_249 = None
        addmm_62: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg191_1, view_189, arg192_1);  arg191_1 = view_189 = arg192_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_190: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_62, [1, 1024, 5120]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_124: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_190, 0.5)
        pow_16: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_190, 3.0)
        mul_125: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_129: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_190, mul_125);  view_190 = mul_125 = None
        mul_126: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_129, 0.7978845608028654);  add_129 = None
        tanh_15: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
        add_130: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_127: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_124, add_130);  mul_124 = add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_191: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_127, [-1, 5120]);  mul_127 = None
        addmm_63: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg193_1, view_191, arg194_1);  arg193_1 = view_191 = arg194_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_192: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_63, [1, 1024, 1280]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_131: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_126, view_192);  add_126 = view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_256: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_131, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_256, [2], correction = 0, keepdim = True)
        getitem_256: "f32[1, 1024, 1]" = var_mean_32[0]
        getitem_257: "f32[1, 1024, 1]" = var_mean_32[1];  var_mean_32 = None
        sub_34: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_256, getitem_257);  convert_element_type_256 = getitem_257 = None
        add_132: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_256, 1e-05);  getitem_256 = None
        rsqrt_32: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        mul_128: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_32);  sub_34 = rsqrt_32 = None
        mul_129: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_128, arg195_1);  mul_128 = arg195_1 = None
        add_133: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_129, arg196_1);  mul_129 = arg196_1 = None
        convert_element_type_257: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_133, torch.float16);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_193: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_257, [-1, 1280]);  convert_element_type_257 = None
        addmm_64: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg197_1, view_193, arg198_1);  arg197_1 = view_193 = arg198_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_194: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_64, [1, 1024, 3840]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_16 = torch.ops.aten.split.Tensor(view_194, 1280, 2);  view_194 = None
        getitem_258: "f16[1, 1024, 1280]" = split_16[0]
        getitem_259: "f16[1, 1024, 1280]" = split_16[1]
        getitem_260: "f16[1, 1024, 1280]" = split_16[2];  split_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_197: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_258, [1, 1024, -1, 64]);  getitem_258 = None
        permute_66: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_197, [0, 2, 1, 3]);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_195: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_259, [1, 1024, -1, 64]);  getitem_259 = None
        permute_64: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_196: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_260, [1, 1024, -1, 64]);  getitem_260 = None
        permute_65: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_34: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_34, full_default_33);  full_default_34 = full_default_33 = None
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_66, permute_64, permute_65, where_16, False, scale = 0.125);  permute_66 = permute_64 = permute_65 = where_16 = None
        getitem_261: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_16[0];  _scaled_dot_product_cudnn_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_67: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_261, [0, 2, 1, 3]);  getitem_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_198: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_67, [1, 1024, -1]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_199: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_198, [-1, 1280]);  view_198 = None
        addmm_65: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg199_1, view_199, arg200_1);  arg199_1 = view_199 = arg200_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_200: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_65, [1, 1024, 1280]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_134: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_200, add_131);  view_200 = add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_264: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_134, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_264, [2], correction = 0, keepdim = True)
        getitem_270: "f32[1, 1024, 1]" = var_mean_33[0]
        getitem_271: "f32[1, 1024, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_35: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_264, getitem_271);  convert_element_type_264 = getitem_271 = None
        add_135: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_270, 1e-05);  getitem_270 = None
        rsqrt_33: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        mul_130: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_33);  sub_35 = rsqrt_33 = None
        mul_131: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_130, arg201_1);  mul_130 = arg201_1 = None
        add_136: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_131, arg202_1);  mul_131 = arg202_1 = None
        convert_element_type_265: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_136, torch.float16);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_201: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_265, [-1, 1280]);  convert_element_type_265 = None
        addmm_66: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg203_1, view_201, arg204_1);  arg203_1 = view_201 = arg204_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_202: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_66, [1, 1024, 5120]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_132: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_202, 0.5)
        pow_17: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_202, 3.0)
        mul_133: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_137: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_202, mul_133);  view_202 = mul_133 = None
        mul_134: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_137, 0.7978845608028654);  add_137 = None
        tanh_16: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_134);  mul_134 = None
        add_138: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
        mul_135: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_132, add_138);  mul_132 = add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_203: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_135, [-1, 5120]);  mul_135 = None
        addmm_67: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg205_1, view_203, arg206_1);  arg205_1 = view_203 = arg206_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_204: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_67, [1, 1024, 1280]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_139: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_134, view_204);  add_134 = view_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_272: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_139, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_272, [2], correction = 0, keepdim = True)
        getitem_272: "f32[1, 1024, 1]" = var_mean_34[0]
        getitem_273: "f32[1, 1024, 1]" = var_mean_34[1];  var_mean_34 = None
        sub_36: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_272, getitem_273);  convert_element_type_272 = getitem_273 = None
        add_140: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_272, 1e-05);  getitem_272 = None
        rsqrt_34: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_136: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_34);  sub_36 = rsqrt_34 = None
        mul_137: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_136, arg207_1);  mul_136 = arg207_1 = None
        add_141: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_137, arg208_1);  mul_137 = arg208_1 = None
        convert_element_type_273: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_141, torch.float16);  add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_205: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_273, [-1, 1280]);  convert_element_type_273 = None
        addmm_68: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg209_1, view_205, arg210_1);  arg209_1 = view_205 = arg210_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_206: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_68, [1, 1024, 3840]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_17 = torch.ops.aten.split.Tensor(view_206, 1280, 2);  view_206 = None
        getitem_274: "f16[1, 1024, 1280]" = split_17[0]
        getitem_275: "f16[1, 1024, 1280]" = split_17[1]
        getitem_276: "f16[1, 1024, 1280]" = split_17[2];  split_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_209: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_274, [1, 1024, -1, 64]);  getitem_274 = None
        permute_70: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_207: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_275, [1, 1024, -1, 64]);  getitem_275 = None
        permute_68: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_208: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_276, [1, 1024, -1, 64]);  getitem_276 = None
        permute_69: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_36: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_36, full_default_35);  full_default_36 = full_default_35 = None
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_70, permute_68, permute_69, where_17, False, scale = 0.125);  permute_70 = permute_68 = permute_69 = where_17 = None
        getitem_277: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_17[0];  _scaled_dot_product_cudnn_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_71: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_277, [0, 2, 1, 3]);  getitem_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_210: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_71, [1, 1024, -1]);  permute_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_211: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_210, [-1, 1280]);  view_210 = None
        addmm_69: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg211_1, view_211, arg212_1);  arg211_1 = view_211 = arg212_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_212: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_69, [1, 1024, 1280]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_142: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_212, add_139);  view_212 = add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_280: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_142, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_280, [2], correction = 0, keepdim = True)
        getitem_286: "f32[1, 1024, 1]" = var_mean_35[0]
        getitem_287: "f32[1, 1024, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_37: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_280, getitem_287);  convert_element_type_280 = getitem_287 = None
        add_143: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_286, 1e-05);  getitem_286 = None
        rsqrt_35: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        mul_138: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_35);  sub_37 = rsqrt_35 = None
        mul_139: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_138, arg213_1);  mul_138 = arg213_1 = None
        add_144: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_139, arg214_1);  mul_139 = arg214_1 = None
        convert_element_type_281: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_144, torch.float16);  add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_213: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_281, [-1, 1280]);  convert_element_type_281 = None
        addmm_70: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg215_1, view_213, arg216_1);  arg215_1 = view_213 = arg216_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_214: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_70, [1, 1024, 5120]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_214, 0.5)
        pow_18: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_214, 3.0)
        mul_141: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_145: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_214, mul_141);  view_214 = mul_141 = None
        mul_142: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_145, 0.7978845608028654);  add_145 = None
        tanh_17: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_146: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
        mul_143: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_140, add_146);  mul_140 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_215: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_143, [-1, 5120]);  mul_143 = None
        addmm_71: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg217_1, view_215, arg218_1);  arg217_1 = view_215 = arg218_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_216: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_71, [1, 1024, 1280]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_147: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_142, view_216);  add_142 = view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_288: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_147, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_288, [2], correction = 0, keepdim = True)
        getitem_288: "f32[1, 1024, 1]" = var_mean_36[0]
        getitem_289: "f32[1, 1024, 1]" = var_mean_36[1];  var_mean_36 = None
        sub_38: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_288, getitem_289);  convert_element_type_288 = getitem_289 = None
        add_148: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_288, 1e-05);  getitem_288 = None
        rsqrt_36: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_144: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_36);  sub_38 = rsqrt_36 = None
        mul_145: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_144, arg219_1);  mul_144 = arg219_1 = None
        add_149: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_145, arg220_1);  mul_145 = arg220_1 = None
        convert_element_type_289: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_149, torch.float16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_217: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_289, [-1, 1280]);  convert_element_type_289 = None
        addmm_72: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg221_1, view_217, arg222_1);  arg221_1 = view_217 = arg222_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_218: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_72, [1, 1024, 3840]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_18 = torch.ops.aten.split.Tensor(view_218, 1280, 2);  view_218 = None
        getitem_290: "f16[1, 1024, 1280]" = split_18[0]
        getitem_291: "f16[1, 1024, 1280]" = split_18[1]
        getitem_292: "f16[1, 1024, 1280]" = split_18[2];  split_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_221: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_290, [1, 1024, -1, 64]);  getitem_290 = None
        permute_74: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_219: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_291, [1, 1024, -1, 64]);  getitem_291 = None
        permute_72: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_220: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_292, [1, 1024, -1, 64]);  getitem_292 = None
        permute_73: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_38: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_38, full_default_37);  full_default_38 = full_default_37 = None
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_74, permute_72, permute_73, where_18, False, scale = 0.125);  permute_74 = permute_72 = permute_73 = where_18 = None
        getitem_293: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_18[0];  _scaled_dot_product_cudnn_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_75: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_293, [0, 2, 1, 3]);  getitem_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_222: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_75, [1, 1024, -1]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_223: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_222, [-1, 1280]);  view_222 = None
        addmm_73: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg223_1, view_223, arg224_1);  arg223_1 = view_223 = arg224_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_224: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_73, [1, 1024, 1280]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_150: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_224, add_147);  view_224 = add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_296: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_150, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_296, [2], correction = 0, keepdim = True)
        getitem_302: "f32[1, 1024, 1]" = var_mean_37[0]
        getitem_303: "f32[1, 1024, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_39: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_296, getitem_303);  convert_element_type_296 = getitem_303 = None
        add_151: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_302, 1e-05);  getitem_302 = None
        rsqrt_37: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        mul_146: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_37);  sub_39 = rsqrt_37 = None
        mul_147: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_146, arg225_1);  mul_146 = arg225_1 = None
        add_152: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_147, arg226_1);  mul_147 = arg226_1 = None
        convert_element_type_297: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_152, torch.float16);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_225: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_297, [-1, 1280]);  convert_element_type_297 = None
        addmm_74: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg227_1, view_225, arg228_1);  arg227_1 = view_225 = arg228_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_226: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_74, [1, 1024, 5120]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_148: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_226, 0.5)
        pow_19: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_226, 3.0)
        mul_149: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_153: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_226, mul_149);  view_226 = mul_149 = None
        mul_150: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_153, 0.7978845608028654);  add_153 = None
        tanh_18: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
        add_154: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
        mul_151: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_148, add_154);  mul_148 = add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_227: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_151, [-1, 5120]);  mul_151 = None
        addmm_75: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg229_1, view_227, arg230_1);  arg229_1 = view_227 = arg230_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_228: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_75, [1, 1024, 1280]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_155: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_150, view_228);  add_150 = view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_304: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_155, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_304, [2], correction = 0, keepdim = True)
        getitem_304: "f32[1, 1024, 1]" = var_mean_38[0]
        getitem_305: "f32[1, 1024, 1]" = var_mean_38[1];  var_mean_38 = None
        sub_40: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_304, getitem_305);  convert_element_type_304 = getitem_305 = None
        add_156: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_304, 1e-05);  getitem_304 = None
        rsqrt_38: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        mul_152: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_38);  sub_40 = rsqrt_38 = None
        mul_153: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_152, arg231_1);  mul_152 = arg231_1 = None
        add_157: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_153, arg232_1);  mul_153 = arg232_1 = None
        convert_element_type_305: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_157, torch.float16);  add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_229: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_305, [-1, 1280]);  convert_element_type_305 = None
        addmm_76: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg233_1, view_229, arg234_1);  arg233_1 = view_229 = arg234_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_230: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_76, [1, 1024, 3840]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_19 = torch.ops.aten.split.Tensor(view_230, 1280, 2);  view_230 = None
        getitem_306: "f16[1, 1024, 1280]" = split_19[0]
        getitem_307: "f16[1, 1024, 1280]" = split_19[1]
        getitem_308: "f16[1, 1024, 1280]" = split_19[2];  split_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_233: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_306, [1, 1024, -1, 64]);  getitem_306 = None
        permute_78: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_231: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_307, [1, 1024, -1, 64]);  getitem_307 = None
        permute_76: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_232: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_308, [1, 1024, -1, 64]);  getitem_308 = None
        permute_77: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_40: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_40, full_default_39);  full_default_40 = full_default_39 = None
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_78, permute_76, permute_77, where_19, False, scale = 0.125);  permute_78 = permute_76 = permute_77 = where_19 = None
        getitem_309: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_19[0];  _scaled_dot_product_cudnn_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_79: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_309, [0, 2, 1, 3]);  getitem_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_234: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_79, [1, 1024, -1]);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_235: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_234, [-1, 1280]);  view_234 = None
        addmm_77: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg235_1, view_235, arg236_1);  arg235_1 = view_235 = arg236_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_236: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_77, [1, 1024, 1280]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_158: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_236, add_155);  view_236 = add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_312: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_158, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_312, [2], correction = 0, keepdim = True)
        getitem_318: "f32[1, 1024, 1]" = var_mean_39[0]
        getitem_319: "f32[1, 1024, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_41: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_312, getitem_319);  convert_element_type_312 = getitem_319 = None
        add_159: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_318, 1e-05);  getitem_318 = None
        rsqrt_39: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        mul_154: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_39);  sub_41 = rsqrt_39 = None
        mul_155: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_154, arg237_1);  mul_154 = arg237_1 = None
        add_160: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_155, arg238_1);  mul_155 = arg238_1 = None
        convert_element_type_313: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_160, torch.float16);  add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_237: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_313, [-1, 1280]);  convert_element_type_313 = None
        addmm_78: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg239_1, view_237, arg240_1);  arg239_1 = view_237 = arg240_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_238: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_78, [1, 1024, 5120]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_156: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_238, 0.5)
        pow_20: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_238, 3.0)
        mul_157: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_161: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_238, mul_157);  view_238 = mul_157 = None
        mul_158: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_161, 0.7978845608028654);  add_161 = None
        tanh_19: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_162: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
        mul_159: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_156, add_162);  mul_156 = add_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_239: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_159, [-1, 5120]);  mul_159 = None
        addmm_79: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg241_1, view_239, arg242_1);  arg241_1 = view_239 = arg242_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_240: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_79, [1, 1024, 1280]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_163: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_158, view_240);  add_158 = view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_320: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_163, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_320, [2], correction = 0, keepdim = True)
        getitem_320: "f32[1, 1024, 1]" = var_mean_40[0]
        getitem_321: "f32[1, 1024, 1]" = var_mean_40[1];  var_mean_40 = None
        sub_42: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_320, getitem_321);  convert_element_type_320 = getitem_321 = None
        add_164: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_320, 1e-05);  getitem_320 = None
        rsqrt_40: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        mul_160: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_40);  sub_42 = rsqrt_40 = None
        mul_161: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_160, arg243_1);  mul_160 = arg243_1 = None
        add_165: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_161, arg244_1);  mul_161 = arg244_1 = None
        convert_element_type_321: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_165, torch.float16);  add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_241: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_321, [-1, 1280]);  convert_element_type_321 = None
        addmm_80: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg245_1, view_241, arg246_1);  arg245_1 = view_241 = arg246_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_242: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_80, [1, 1024, 3840]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_20 = torch.ops.aten.split.Tensor(view_242, 1280, 2);  view_242 = None
        getitem_322: "f16[1, 1024, 1280]" = split_20[0]
        getitem_323: "f16[1, 1024, 1280]" = split_20[1]
        getitem_324: "f16[1, 1024, 1280]" = split_20[2];  split_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_245: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_322, [1, 1024, -1, 64]);  getitem_322 = None
        permute_82: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_243: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_323, [1, 1024, -1, 64]);  getitem_323 = None
        permute_80: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_244: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_324, [1, 1024, -1, 64]);  getitem_324 = None
        permute_81: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_42: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_42, full_default_41);  full_default_42 = full_default_41 = None
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_82, permute_80, permute_81, where_20, False, scale = 0.125);  permute_82 = permute_80 = permute_81 = where_20 = None
        getitem_325: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_20[0];  _scaled_dot_product_cudnn_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_83: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_325, [0, 2, 1, 3]);  getitem_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_246: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_83, [1, 1024, -1]);  permute_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_247: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_246, [-1, 1280]);  view_246 = None
        addmm_81: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg247_1, view_247, arg248_1);  arg247_1 = view_247 = arg248_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_248: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_81, [1, 1024, 1280]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_166: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_248, add_163);  view_248 = add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_328: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_166, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_328, [2], correction = 0, keepdim = True)
        getitem_334: "f32[1, 1024, 1]" = var_mean_41[0]
        getitem_335: "f32[1, 1024, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_43: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_328, getitem_335);  convert_element_type_328 = getitem_335 = None
        add_167: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_334, 1e-05);  getitem_334 = None
        rsqrt_41: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
        mul_162: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_41);  sub_43 = rsqrt_41 = None
        mul_163: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_162, arg249_1);  mul_162 = arg249_1 = None
        add_168: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_163, arg250_1);  mul_163 = arg250_1 = None
        convert_element_type_329: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_168, torch.float16);  add_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_249: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_329, [-1, 1280]);  convert_element_type_329 = None
        addmm_82: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg251_1, view_249, arg252_1);  arg251_1 = view_249 = arg252_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_250: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_82, [1, 1024, 5120]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_164: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_250, 0.5)
        pow_21: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_250, 3.0)
        mul_165: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_169: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_250, mul_165);  view_250 = mul_165 = None
        mul_166: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_169, 0.7978845608028654);  add_169 = None
        tanh_20: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_170: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
        mul_167: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_164, add_170);  mul_164 = add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_251: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_167, [-1, 5120]);  mul_167 = None
        addmm_83: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg253_1, view_251, arg254_1);  arg253_1 = view_251 = arg254_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_252: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_83, [1, 1024, 1280]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_171: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_166, view_252);  add_166 = view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_336: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_171, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_336, [2], correction = 0, keepdim = True)
        getitem_336: "f32[1, 1024, 1]" = var_mean_42[0]
        getitem_337: "f32[1, 1024, 1]" = var_mean_42[1];  var_mean_42 = None
        sub_44: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_336, getitem_337);  convert_element_type_336 = getitem_337 = None
        add_172: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_336, 1e-05);  getitem_336 = None
        rsqrt_42: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        mul_168: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_42);  sub_44 = rsqrt_42 = None
        mul_169: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_168, arg255_1);  mul_168 = arg255_1 = None
        add_173: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_169, arg256_1);  mul_169 = arg256_1 = None
        convert_element_type_337: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_173, torch.float16);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_253: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_337, [-1, 1280]);  convert_element_type_337 = None
        addmm_84: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg257_1, view_253, arg258_1);  arg257_1 = view_253 = arg258_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_254: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_84, [1, 1024, 3840]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_21 = torch.ops.aten.split.Tensor(view_254, 1280, 2);  view_254 = None
        getitem_338: "f16[1, 1024, 1280]" = split_21[0]
        getitem_339: "f16[1, 1024, 1280]" = split_21[1]
        getitem_340: "f16[1, 1024, 1280]" = split_21[2];  split_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_257: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_338, [1, 1024, -1, 64]);  getitem_338 = None
        permute_86: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_255: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_339, [1, 1024, -1, 64]);  getitem_339 = None
        permute_84: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_256: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_340, [1, 1024, -1, 64]);  getitem_340 = None
        permute_85: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_44: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_44, full_default_43);  full_default_44 = full_default_43 = None
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_86, permute_84, permute_85, where_21, False, scale = 0.125);  permute_86 = permute_84 = permute_85 = where_21 = None
        getitem_341: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_21[0];  _scaled_dot_product_cudnn_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_87: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_341, [0, 2, 1, 3]);  getitem_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_258: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_87, [1, 1024, -1]);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_259: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_258, [-1, 1280]);  view_258 = None
        addmm_85: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg259_1, view_259, arg260_1);  arg259_1 = view_259 = arg260_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_260: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_85, [1, 1024, 1280]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_174: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_260, add_171);  view_260 = add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_344: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_174, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_344, [2], correction = 0, keepdim = True)
        getitem_350: "f32[1, 1024, 1]" = var_mean_43[0]
        getitem_351: "f32[1, 1024, 1]" = var_mean_43[1];  var_mean_43 = None
        sub_45: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_344, getitem_351);  convert_element_type_344 = getitem_351 = None
        add_175: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_350, 1e-05);  getitem_350 = None
        rsqrt_43: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        mul_170: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_43);  sub_45 = rsqrt_43 = None
        mul_171: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_170, arg261_1);  mul_170 = arg261_1 = None
        add_176: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_171, arg262_1);  mul_171 = arg262_1 = None
        convert_element_type_345: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_176, torch.float16);  add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_261: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_345, [-1, 1280]);  convert_element_type_345 = None
        addmm_86: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg263_1, view_261, arg264_1);  arg263_1 = view_261 = arg264_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_262: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_86, [1, 1024, 5120]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_172: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_262, 0.5)
        pow_22: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_262, 3.0)
        mul_173: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_177: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_262, mul_173);  view_262 = mul_173 = None
        mul_174: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_177, 0.7978845608028654);  add_177 = None
        tanh_21: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
        add_178: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
        mul_175: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_172, add_178);  mul_172 = add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_263: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_175, [-1, 5120]);  mul_175 = None
        addmm_87: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg265_1, view_263, arg266_1);  arg265_1 = view_263 = arg266_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_264: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_87, [1, 1024, 1280]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_179: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_174, view_264);  add_174 = view_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_352: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_179, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_352, [2], correction = 0, keepdim = True)
        getitem_352: "f32[1, 1024, 1]" = var_mean_44[0]
        getitem_353: "f32[1, 1024, 1]" = var_mean_44[1];  var_mean_44 = None
        sub_46: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_352, getitem_353);  convert_element_type_352 = getitem_353 = None
        add_180: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_352, 1e-05);  getitem_352 = None
        rsqrt_44: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        mul_176: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_44);  sub_46 = rsqrt_44 = None
        mul_177: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_176, arg267_1);  mul_176 = arg267_1 = None
        add_181: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_177, arg268_1);  mul_177 = arg268_1 = None
        convert_element_type_353: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_181, torch.float16);  add_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_265: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_353, [-1, 1280]);  convert_element_type_353 = None
        addmm_88: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg269_1, view_265, arg270_1);  arg269_1 = view_265 = arg270_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_266: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_88, [1, 1024, 3840]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_22 = torch.ops.aten.split.Tensor(view_266, 1280, 2);  view_266 = None
        getitem_354: "f16[1, 1024, 1280]" = split_22[0]
        getitem_355: "f16[1, 1024, 1280]" = split_22[1]
        getitem_356: "f16[1, 1024, 1280]" = split_22[2];  split_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_269: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_354, [1, 1024, -1, 64]);  getitem_354 = None
        permute_90: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_267: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_355, [1, 1024, -1, 64]);  getitem_355 = None
        permute_88: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_268: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_356, [1, 1024, -1, 64]);  getitem_356 = None
        permute_89: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_46: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_46, full_default_45);  full_default_46 = full_default_45 = None
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_90, permute_88, permute_89, where_22, False, scale = 0.125);  permute_90 = permute_88 = permute_89 = where_22 = None
        getitem_357: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_22[0];  _scaled_dot_product_cudnn_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_91: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_357, [0, 2, 1, 3]);  getitem_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_270: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_91, [1, 1024, -1]);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_271: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_270, [-1, 1280]);  view_270 = None
        addmm_89: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg271_1, view_271, arg272_1);  arg271_1 = view_271 = arg272_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_272: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_89, [1, 1024, 1280]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_182: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_272, add_179);  view_272 = add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_360: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_182, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_360, [2], correction = 0, keepdim = True)
        getitem_366: "f32[1, 1024, 1]" = var_mean_45[0]
        getitem_367: "f32[1, 1024, 1]" = var_mean_45[1];  var_mean_45 = None
        sub_47: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_360, getitem_367);  convert_element_type_360 = getitem_367 = None
        add_183: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_366, 1e-05);  getitem_366 = None
        rsqrt_45: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        mul_178: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_45);  sub_47 = rsqrt_45 = None
        mul_179: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_178, arg273_1);  mul_178 = arg273_1 = None
        add_184: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_179, arg274_1);  mul_179 = arg274_1 = None
        convert_element_type_361: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_184, torch.float16);  add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_273: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_361, [-1, 1280]);  convert_element_type_361 = None
        addmm_90: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg275_1, view_273, arg276_1);  arg275_1 = view_273 = arg276_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_274: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_90, [1, 1024, 5120]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_180: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_274, 0.5)
        pow_23: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_274, 3.0)
        mul_181: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_185: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_274, mul_181);  view_274 = mul_181 = None
        mul_182: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_185, 0.7978845608028654);  add_185 = None
        tanh_22: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_182);  mul_182 = None
        add_186: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
        mul_183: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_180, add_186);  mul_180 = add_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_275: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_183, [-1, 5120]);  mul_183 = None
        addmm_91: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg277_1, view_275, arg278_1);  arg277_1 = view_275 = arg278_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_276: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_91, [1, 1024, 1280]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_187: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_182, view_276);  add_182 = view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_368: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_187, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_368, [2], correction = 0, keepdim = True)
        getitem_368: "f32[1, 1024, 1]" = var_mean_46[0]
        getitem_369: "f32[1, 1024, 1]" = var_mean_46[1];  var_mean_46 = None
        sub_48: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_368, getitem_369);  convert_element_type_368 = getitem_369 = None
        add_188: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_368, 1e-05);  getitem_368 = None
        rsqrt_46: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_184: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_46);  sub_48 = rsqrt_46 = None
        mul_185: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_184, arg279_1);  mul_184 = arg279_1 = None
        add_189: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_185, arg280_1);  mul_185 = arg280_1 = None
        convert_element_type_369: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_189, torch.float16);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_277: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_369, [-1, 1280]);  convert_element_type_369 = None
        addmm_92: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg281_1, view_277, arg282_1);  arg281_1 = view_277 = arg282_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_278: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_92, [1, 1024, 3840]);  addmm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_23 = torch.ops.aten.split.Tensor(view_278, 1280, 2);  view_278 = None
        getitem_370: "f16[1, 1024, 1280]" = split_23[0]
        getitem_371: "f16[1, 1024, 1280]" = split_23[1]
        getitem_372: "f16[1, 1024, 1280]" = split_23[2];  split_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_281: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_370, [1, 1024, -1, 64]);  getitem_370 = None
        permute_94: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_279: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_371, [1, 1024, -1, 64]);  getitem_371 = None
        permute_92: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_280: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_372, [1, 1024, -1, 64]);  getitem_372 = None
        permute_93: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_48: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_48, full_default_47);  full_default_48 = full_default_47 = None
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_94, permute_92, permute_93, where_23, False, scale = 0.125);  permute_94 = permute_92 = permute_93 = where_23 = None
        getitem_373: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_23[0];  _scaled_dot_product_cudnn_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_373, [0, 2, 1, 3]);  getitem_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_282: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_95, [1, 1024, -1]);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_283: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_282, [-1, 1280]);  view_282 = None
        addmm_93: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg283_1, view_283, arg284_1);  arg283_1 = view_283 = arg284_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_284: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_93, [1, 1024, 1280]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_190: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_284, add_187);  view_284 = add_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_376: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_190, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_376, [2], correction = 0, keepdim = True)
        getitem_382: "f32[1, 1024, 1]" = var_mean_47[0]
        getitem_383: "f32[1, 1024, 1]" = var_mean_47[1];  var_mean_47 = None
        sub_49: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_376, getitem_383);  convert_element_type_376 = getitem_383 = None
        add_191: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_382, 1e-05);  getitem_382 = None
        rsqrt_47: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        mul_186: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_47);  sub_49 = rsqrt_47 = None
        mul_187: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_186, arg285_1);  mul_186 = arg285_1 = None
        add_192: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_187, arg286_1);  mul_187 = arg286_1 = None
        convert_element_type_377: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_192, torch.float16);  add_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_285: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_377, [-1, 1280]);  convert_element_type_377 = None
        addmm_94: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg287_1, view_285, arg288_1);  arg287_1 = view_285 = arg288_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_286: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_94, [1, 1024, 5120]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_188: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_286, 0.5)
        pow_24: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_286, 3.0)
        mul_189: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_193: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_286, mul_189);  view_286 = mul_189 = None
        mul_190: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_193, 0.7978845608028654);  add_193 = None
        tanh_23: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
        add_194: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
        mul_191: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_188, add_194);  mul_188 = add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_287: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_191, [-1, 5120]);  mul_191 = None
        addmm_95: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg289_1, view_287, arg290_1);  arg289_1 = view_287 = arg290_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_288: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_95, [1, 1024, 1280]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_195: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_190, view_288);  add_190 = view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_384: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_195, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_384, [2], correction = 0, keepdim = True)
        getitem_384: "f32[1, 1024, 1]" = var_mean_48[0]
        getitem_385: "f32[1, 1024, 1]" = var_mean_48[1];  var_mean_48 = None
        sub_50: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_384, getitem_385);  convert_element_type_384 = getitem_385 = None
        add_196: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_384, 1e-05);  getitem_384 = None
        rsqrt_48: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        mul_192: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_48);  sub_50 = rsqrt_48 = None
        mul_193: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_192, arg291_1);  mul_192 = arg291_1 = None
        add_197: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_193, arg292_1);  mul_193 = arg292_1 = None
        convert_element_type_385: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_197, torch.float16);  add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_289: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_385, [-1, 1280]);  convert_element_type_385 = None
        addmm_96: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg293_1, view_289, arg294_1);  arg293_1 = view_289 = arg294_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_290: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_96, [1, 1024, 3840]);  addmm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_24 = torch.ops.aten.split.Tensor(view_290, 1280, 2);  view_290 = None
        getitem_386: "f16[1, 1024, 1280]" = split_24[0]
        getitem_387: "f16[1, 1024, 1280]" = split_24[1]
        getitem_388: "f16[1, 1024, 1280]" = split_24[2];  split_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_293: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_386, [1, 1024, -1, 64]);  getitem_386 = None
        permute_98: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_291: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_387, [1, 1024, -1, 64]);  getitem_387 = None
        permute_96: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_292: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_388, [1, 1024, -1, 64]);  getitem_388 = None
        permute_97: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_50: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_50, full_default_49);  full_default_50 = full_default_49 = None
        _scaled_dot_product_cudnn_attention_24 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_98, permute_96, permute_97, where_24, False, scale = 0.125);  permute_98 = permute_96 = permute_97 = where_24 = None
        getitem_389: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_24[0];  _scaled_dot_product_cudnn_attention_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_99: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_389, [0, 2, 1, 3]);  getitem_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_294: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_99, [1, 1024, -1]);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_295: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_294, [-1, 1280]);  view_294 = None
        addmm_97: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg295_1, view_295, arg296_1);  arg295_1 = view_295 = arg296_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_296: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_97, [1, 1024, 1280]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_198: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_296, add_195);  view_296 = add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_392: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_198, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_392, [2], correction = 0, keepdim = True)
        getitem_398: "f32[1, 1024, 1]" = var_mean_49[0]
        getitem_399: "f32[1, 1024, 1]" = var_mean_49[1];  var_mean_49 = None
        sub_51: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_392, getitem_399);  convert_element_type_392 = getitem_399 = None
        add_199: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_398, 1e-05);  getitem_398 = None
        rsqrt_49: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        mul_194: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_49);  sub_51 = rsqrt_49 = None
        mul_195: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_194, arg297_1);  mul_194 = arg297_1 = None
        add_200: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_195, arg298_1);  mul_195 = arg298_1 = None
        convert_element_type_393: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_200, torch.float16);  add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_297: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_393, [-1, 1280]);  convert_element_type_393 = None
        addmm_98: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg299_1, view_297, arg300_1);  arg299_1 = view_297 = arg300_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_298: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_98, [1, 1024, 5120]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_196: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_298, 0.5)
        pow_25: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_298, 3.0)
        mul_197: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_201: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_298, mul_197);  view_298 = mul_197 = None
        mul_198: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_201, 0.7978845608028654);  add_201 = None
        tanh_24: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_198);  mul_198 = None
        add_202: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_24, 1.0);  tanh_24 = None
        mul_199: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_196, add_202);  mul_196 = add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_299: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_199, [-1, 5120]);  mul_199 = None
        addmm_99: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg301_1, view_299, arg302_1);  arg301_1 = view_299 = arg302_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_300: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_99, [1, 1024, 1280]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_203: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_198, view_300);  add_198 = view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_400: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_203, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_400, [2], correction = 0, keepdim = True)
        getitem_400: "f32[1, 1024, 1]" = var_mean_50[0]
        getitem_401: "f32[1, 1024, 1]" = var_mean_50[1];  var_mean_50 = None
        sub_52: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_400, getitem_401);  convert_element_type_400 = getitem_401 = None
        add_204: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_400, 1e-05);  getitem_400 = None
        rsqrt_50: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        mul_200: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_50);  sub_52 = rsqrt_50 = None
        mul_201: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_200, arg303_1);  mul_200 = arg303_1 = None
        add_205: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_201, arg304_1);  mul_201 = arg304_1 = None
        convert_element_type_401: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_205, torch.float16);  add_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_301: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_401, [-1, 1280]);  convert_element_type_401 = None
        addmm_100: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg305_1, view_301, arg306_1);  arg305_1 = view_301 = arg306_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_302: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_100, [1, 1024, 3840]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_25 = torch.ops.aten.split.Tensor(view_302, 1280, 2);  view_302 = None
        getitem_402: "f16[1, 1024, 1280]" = split_25[0]
        getitem_403: "f16[1, 1024, 1280]" = split_25[1]
        getitem_404: "f16[1, 1024, 1280]" = split_25[2];  split_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_305: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_402, [1, 1024, -1, 64]);  getitem_402 = None
        permute_102: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_303: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_403, [1, 1024, -1, 64]);  getitem_403 = None
        permute_100: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_304: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_404, [1, 1024, -1, 64]);  getitem_404 = None
        permute_101: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_52: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_52, full_default_51);  full_default_52 = full_default_51 = None
        _scaled_dot_product_cudnn_attention_25 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_102, permute_100, permute_101, where_25, False, scale = 0.125);  permute_102 = permute_100 = permute_101 = where_25 = None
        getitem_405: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_25[0];  _scaled_dot_product_cudnn_attention_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_103: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_405, [0, 2, 1, 3]);  getitem_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_306: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_103, [1, 1024, -1]);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_307: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_306, [-1, 1280]);  view_306 = None
        addmm_101: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg307_1, view_307, arg308_1);  arg307_1 = view_307 = arg308_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_308: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_101, [1, 1024, 1280]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_206: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_308, add_203);  view_308 = add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_408: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_206, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_408, [2], correction = 0, keepdim = True)
        getitem_414: "f32[1, 1024, 1]" = var_mean_51[0]
        getitem_415: "f32[1, 1024, 1]" = var_mean_51[1];  var_mean_51 = None
        sub_53: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_408, getitem_415);  convert_element_type_408 = getitem_415 = None
        add_207: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_414, 1e-05);  getitem_414 = None
        rsqrt_51: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        mul_202: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_51);  sub_53 = rsqrt_51 = None
        mul_203: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_202, arg309_1);  mul_202 = arg309_1 = None
        add_208: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_203, arg310_1);  mul_203 = arg310_1 = None
        convert_element_type_409: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_208, torch.float16);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_309: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_409, [-1, 1280]);  convert_element_type_409 = None
        addmm_102: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg311_1, view_309, arg312_1);  arg311_1 = view_309 = arg312_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_310: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_102, [1, 1024, 5120]);  addmm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_204: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_310, 0.5)
        pow_26: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_310, 3.0)
        mul_205: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
        add_209: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_310, mul_205);  view_310 = mul_205 = None
        mul_206: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_209, 0.7978845608028654);  add_209 = None
        tanh_25: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_206);  mul_206 = None
        add_210: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_25, 1.0);  tanh_25 = None
        mul_207: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_204, add_210);  mul_204 = add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_311: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_207, [-1, 5120]);  mul_207 = None
        addmm_103: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg313_1, view_311, arg314_1);  arg313_1 = view_311 = arg314_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_312: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_103, [1, 1024, 1280]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_211: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_206, view_312);  add_206 = view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_416: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_211, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_416, [2], correction = 0, keepdim = True)
        getitem_416: "f32[1, 1024, 1]" = var_mean_52[0]
        getitem_417: "f32[1, 1024, 1]" = var_mean_52[1];  var_mean_52 = None
        sub_54: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_416, getitem_417);  convert_element_type_416 = getitem_417 = None
        add_212: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_416, 1e-05);  getitem_416 = None
        rsqrt_52: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        mul_208: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_52);  sub_54 = rsqrt_52 = None
        mul_209: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_208, arg315_1);  mul_208 = arg315_1 = None
        add_213: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_209, arg316_1);  mul_209 = arg316_1 = None
        convert_element_type_417: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_213, torch.float16);  add_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_313: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_417, [-1, 1280]);  convert_element_type_417 = None
        addmm_104: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg317_1, view_313, arg318_1);  arg317_1 = view_313 = arg318_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_314: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_104, [1, 1024, 3840]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_26 = torch.ops.aten.split.Tensor(view_314, 1280, 2);  view_314 = None
        getitem_418: "f16[1, 1024, 1280]" = split_26[0]
        getitem_419: "f16[1, 1024, 1280]" = split_26[1]
        getitem_420: "f16[1, 1024, 1280]" = split_26[2];  split_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_317: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_418, [1, 1024, -1, 64]);  getitem_418 = None
        permute_106: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_315: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_419, [1, 1024, -1, 64]);  getitem_419 = None
        permute_104: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_316: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_420, [1, 1024, -1, 64]);  getitem_420 = None
        permute_105: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_54: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_53: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_54, full_default_53);  full_default_54 = full_default_53 = None
        _scaled_dot_product_cudnn_attention_26 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_106, permute_104, permute_105, where_26, False, scale = 0.125);  permute_106 = permute_104 = permute_105 = where_26 = None
        getitem_421: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_26[0];  _scaled_dot_product_cudnn_attention_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_421, [0, 2, 1, 3]);  getitem_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_318: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_107, [1, 1024, -1]);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_319: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_318, [-1, 1280]);  view_318 = None
        addmm_105: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg319_1, view_319, arg320_1);  arg319_1 = view_319 = arg320_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_320: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_105, [1, 1024, 1280]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_214: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_320, add_211);  view_320 = add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_424: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_214, torch.float32)
        var_mean_53 = torch.ops.aten.var_mean.correction(convert_element_type_424, [2], correction = 0, keepdim = True)
        getitem_430: "f32[1, 1024, 1]" = var_mean_53[0]
        getitem_431: "f32[1, 1024, 1]" = var_mean_53[1];  var_mean_53 = None
        sub_55: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_424, getitem_431);  convert_element_type_424 = getitem_431 = None
        add_215: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_430, 1e-05);  getitem_430 = None
        rsqrt_53: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        mul_210: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_53);  sub_55 = rsqrt_53 = None
        mul_211: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_210, arg321_1);  mul_210 = arg321_1 = None
        add_216: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_211, arg322_1);  mul_211 = arg322_1 = None
        convert_element_type_425: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_216, torch.float16);  add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_321: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_425, [-1, 1280]);  convert_element_type_425 = None
        addmm_106: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg323_1, view_321, arg324_1);  arg323_1 = view_321 = arg324_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_322: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_106, [1, 1024, 5120]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_212: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_322, 0.5)
        pow_27: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_322, 3.0)
        mul_213: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_217: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_322, mul_213);  view_322 = mul_213 = None
        mul_214: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_217, 0.7978845608028654);  add_217 = None
        tanh_26: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_214);  mul_214 = None
        add_218: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_26, 1.0);  tanh_26 = None
        mul_215: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_212, add_218);  mul_212 = add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_323: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_215, [-1, 5120]);  mul_215 = None
        addmm_107: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg325_1, view_323, arg326_1);  arg325_1 = view_323 = arg326_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_324: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_107, [1, 1024, 1280]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_219: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_214, view_324);  add_214 = view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_432: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_219, torch.float32)
        var_mean_54 = torch.ops.aten.var_mean.correction(convert_element_type_432, [2], correction = 0, keepdim = True)
        getitem_432: "f32[1, 1024, 1]" = var_mean_54[0]
        getitem_433: "f32[1, 1024, 1]" = var_mean_54[1];  var_mean_54 = None
        sub_56: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_432, getitem_433);  convert_element_type_432 = getitem_433 = None
        add_220: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_432, 1e-05);  getitem_432 = None
        rsqrt_54: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        mul_216: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_54);  sub_56 = rsqrt_54 = None
        mul_217: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_216, arg327_1);  mul_216 = arg327_1 = None
        add_221: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_217, arg328_1);  mul_217 = arg328_1 = None
        convert_element_type_433: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_221, torch.float16);  add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_325: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_433, [-1, 1280]);  convert_element_type_433 = None
        addmm_108: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg329_1, view_325, arg330_1);  arg329_1 = view_325 = arg330_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_326: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_108, [1, 1024, 3840]);  addmm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_27 = torch.ops.aten.split.Tensor(view_326, 1280, 2);  view_326 = None
        getitem_434: "f16[1, 1024, 1280]" = split_27[0]
        getitem_435: "f16[1, 1024, 1280]" = split_27[1]
        getitem_436: "f16[1, 1024, 1280]" = split_27[2];  split_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_329: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_434, [1, 1024, -1, 64]);  getitem_434 = None
        permute_110: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_327: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_435, [1, 1024, -1, 64]);  getitem_435 = None
        permute_108: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_328: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_436, [1, 1024, -1, 64]);  getitem_436 = None
        permute_109: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_56: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_55: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_56, full_default_55);  full_default_56 = full_default_55 = None
        _scaled_dot_product_cudnn_attention_27 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_110, permute_108, permute_109, where_27, False, scale = 0.125);  permute_110 = permute_108 = permute_109 = where_27 = None
        getitem_437: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_27[0];  _scaled_dot_product_cudnn_attention_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_111: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_437, [0, 2, 1, 3]);  getitem_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_330: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_111, [1, 1024, -1]);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_331: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_330, [-1, 1280]);  view_330 = None
        addmm_109: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg331_1, view_331, arg332_1);  arg331_1 = view_331 = arg332_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_332: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_109, [1, 1024, 1280]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_222: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_332, add_219);  view_332 = add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_440: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_222, torch.float32)
        var_mean_55 = torch.ops.aten.var_mean.correction(convert_element_type_440, [2], correction = 0, keepdim = True)
        getitem_446: "f32[1, 1024, 1]" = var_mean_55[0]
        getitem_447: "f32[1, 1024, 1]" = var_mean_55[1];  var_mean_55 = None
        sub_57: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_440, getitem_447);  convert_element_type_440 = getitem_447 = None
        add_223: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_446, 1e-05);  getitem_446 = None
        rsqrt_55: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        mul_218: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_55);  sub_57 = rsqrt_55 = None
        mul_219: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_218, arg333_1);  mul_218 = arg333_1 = None
        add_224: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_219, arg334_1);  mul_219 = arg334_1 = None
        convert_element_type_441: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_224, torch.float16);  add_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_333: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_441, [-1, 1280]);  convert_element_type_441 = None
        addmm_110: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg335_1, view_333, arg336_1);  arg335_1 = view_333 = arg336_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_334: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_110, [1, 1024, 5120]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_220: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_334, 0.5)
        pow_28: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_334, 3.0)
        mul_221: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
        add_225: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_334, mul_221);  view_334 = mul_221 = None
        mul_222: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_225, 0.7978845608028654);  add_225 = None
        tanh_27: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_222);  mul_222 = None
        add_226: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_27, 1.0);  tanh_27 = None
        mul_223: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_220, add_226);  mul_220 = add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_335: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_223, [-1, 5120]);  mul_223 = None
        addmm_111: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg337_1, view_335, arg338_1);  arg337_1 = view_335 = arg338_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_336: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_111, [1, 1024, 1280]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_227: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_222, view_336);  add_222 = view_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_448: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_227, torch.float32)
        var_mean_56 = torch.ops.aten.var_mean.correction(convert_element_type_448, [2], correction = 0, keepdim = True)
        getitem_448: "f32[1, 1024, 1]" = var_mean_56[0]
        getitem_449: "f32[1, 1024, 1]" = var_mean_56[1];  var_mean_56 = None
        sub_58: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_448, getitem_449);  convert_element_type_448 = getitem_449 = None
        add_228: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_448, 1e-05);  getitem_448 = None
        rsqrt_56: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
        mul_224: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_56);  sub_58 = rsqrt_56 = None
        mul_225: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_224, arg339_1);  mul_224 = arg339_1 = None
        add_229: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_225, arg340_1);  mul_225 = arg340_1 = None
        convert_element_type_449: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_229, torch.float16);  add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_337: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_449, [-1, 1280]);  convert_element_type_449 = None
        addmm_112: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg341_1, view_337, arg342_1);  arg341_1 = view_337 = arg342_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_338: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_112, [1, 1024, 3840]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_28 = torch.ops.aten.split.Tensor(view_338, 1280, 2);  view_338 = None
        getitem_450: "f16[1, 1024, 1280]" = split_28[0]
        getitem_451: "f16[1, 1024, 1280]" = split_28[1]
        getitem_452: "f16[1, 1024, 1280]" = split_28[2];  split_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_341: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_450, [1, 1024, -1, 64]);  getitem_450 = None
        permute_114: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_339: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_451, [1, 1024, -1, 64]);  getitem_451 = None
        permute_112: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_340: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_452, [1, 1024, -1, 64]);  getitem_452 = None
        permute_113: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_58: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_58, full_default_57);  full_default_58 = full_default_57 = None
        _scaled_dot_product_cudnn_attention_28 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_114, permute_112, permute_113, where_28, False, scale = 0.125);  permute_114 = permute_112 = permute_113 = where_28 = None
        getitem_453: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_28[0];  _scaled_dot_product_cudnn_attention_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_115: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_453, [0, 2, 1, 3]);  getitem_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_342: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_115, [1, 1024, -1]);  permute_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_343: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_342, [-1, 1280]);  view_342 = None
        addmm_113: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg343_1, view_343, arg344_1);  arg343_1 = view_343 = arg344_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_344: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_113, [1, 1024, 1280]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_230: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_344, add_227);  view_344 = add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_456: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_230, torch.float32)
        var_mean_57 = torch.ops.aten.var_mean.correction(convert_element_type_456, [2], correction = 0, keepdim = True)
        getitem_462: "f32[1, 1024, 1]" = var_mean_57[0]
        getitem_463: "f32[1, 1024, 1]" = var_mean_57[1];  var_mean_57 = None
        sub_59: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_456, getitem_463);  convert_element_type_456 = getitem_463 = None
        add_231: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_462, 1e-05);  getitem_462 = None
        rsqrt_57: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
        mul_226: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_57);  sub_59 = rsqrt_57 = None
        mul_227: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_226, arg345_1);  mul_226 = arg345_1 = None
        add_232: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_227, arg346_1);  mul_227 = arg346_1 = None
        convert_element_type_457: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_232, torch.float16);  add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_345: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_457, [-1, 1280]);  convert_element_type_457 = None
        addmm_114: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg347_1, view_345, arg348_1);  arg347_1 = view_345 = arg348_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_346: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_114, [1, 1024, 5120]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_228: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_346, 0.5)
        pow_29: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_346, 3.0)
        mul_229: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_233: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_346, mul_229);  view_346 = mul_229 = None
        mul_230: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_233, 0.7978845608028654);  add_233 = None
        tanh_28: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_230);  mul_230 = None
        add_234: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_28, 1.0);  tanh_28 = None
        mul_231: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_228, add_234);  mul_228 = add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_347: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_231, [-1, 5120]);  mul_231 = None
        addmm_115: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg349_1, view_347, arg350_1);  arg349_1 = view_347 = arg350_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_348: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_115, [1, 1024, 1280]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_235: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_230, view_348);  add_230 = view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_464: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_235, torch.float32)
        var_mean_58 = torch.ops.aten.var_mean.correction(convert_element_type_464, [2], correction = 0, keepdim = True)
        getitem_464: "f32[1, 1024, 1]" = var_mean_58[0]
        getitem_465: "f32[1, 1024, 1]" = var_mean_58[1];  var_mean_58 = None
        sub_60: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_464, getitem_465);  convert_element_type_464 = getitem_465 = None
        add_236: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_464, 1e-05);  getitem_464 = None
        rsqrt_58: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        mul_232: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_58);  sub_60 = rsqrt_58 = None
        mul_233: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_232, arg351_1);  mul_232 = arg351_1 = None
        add_237: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_233, arg352_1);  mul_233 = arg352_1 = None
        convert_element_type_465: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_237, torch.float16);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_349: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_465, [-1, 1280]);  convert_element_type_465 = None
        addmm_116: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg353_1, view_349, arg354_1);  arg353_1 = view_349 = arg354_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_350: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_116, [1, 1024, 3840]);  addmm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_29 = torch.ops.aten.split.Tensor(view_350, 1280, 2);  view_350 = None
        getitem_466: "f16[1, 1024, 1280]" = split_29[0]
        getitem_467: "f16[1, 1024, 1280]" = split_29[1]
        getitem_468: "f16[1, 1024, 1280]" = split_29[2];  split_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_353: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_466, [1, 1024, -1, 64]);  getitem_466 = None
        permute_118: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_351: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_467, [1, 1024, -1, 64]);  getitem_467 = None
        permute_116: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_352: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_468, [1, 1024, -1, 64]);  getitem_468 = None
        permute_117: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_60: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_59: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_29: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_60, full_default_59);  full_default_60 = full_default_59 = None
        _scaled_dot_product_cudnn_attention_29 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_118, permute_116, permute_117, where_29, False, scale = 0.125);  permute_118 = permute_116 = permute_117 = where_29 = None
        getitem_469: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_29[0];  _scaled_dot_product_cudnn_attention_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_119: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_469, [0, 2, 1, 3]);  getitem_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_354: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_119, [1, 1024, -1]);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_355: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_354, [-1, 1280]);  view_354 = None
        addmm_117: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg355_1, view_355, arg356_1);  arg355_1 = view_355 = arg356_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_356: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_117, [1, 1024, 1280]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_238: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_356, add_235);  view_356 = add_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_472: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_238, torch.float32)
        var_mean_59 = torch.ops.aten.var_mean.correction(convert_element_type_472, [2], correction = 0, keepdim = True)
        getitem_478: "f32[1, 1024, 1]" = var_mean_59[0]
        getitem_479: "f32[1, 1024, 1]" = var_mean_59[1];  var_mean_59 = None
        sub_61: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_472, getitem_479);  convert_element_type_472 = getitem_479 = None
        add_239: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_478, 1e-05);  getitem_478 = None
        rsqrt_59: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        mul_234: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_59);  sub_61 = rsqrt_59 = None
        mul_235: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_234, arg357_1);  mul_234 = arg357_1 = None
        add_240: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_235, arg358_1);  mul_235 = arg358_1 = None
        convert_element_type_473: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_240, torch.float16);  add_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_357: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_473, [-1, 1280]);  convert_element_type_473 = None
        addmm_118: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg359_1, view_357, arg360_1);  arg359_1 = view_357 = arg360_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_358: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_118, [1, 1024, 5120]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_236: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_358, 0.5)
        pow_30: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_358, 3.0)
        mul_237: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_30, 0.044715);  pow_30 = None
        add_241: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_358, mul_237);  view_358 = mul_237 = None
        mul_238: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_241, 0.7978845608028654);  add_241 = None
        tanh_29: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
        add_242: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_29, 1.0);  tanh_29 = None
        mul_239: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_236, add_242);  mul_236 = add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_359: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_239, [-1, 5120]);  mul_239 = None
        addmm_119: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg361_1, view_359, arg362_1);  arg361_1 = view_359 = arg362_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_360: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_119, [1, 1024, 1280]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_243: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_238, view_360);  add_238 = view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_480: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_243, torch.float32)
        var_mean_60 = torch.ops.aten.var_mean.correction(convert_element_type_480, [2], correction = 0, keepdim = True)
        getitem_480: "f32[1, 1024, 1]" = var_mean_60[0]
        getitem_481: "f32[1, 1024, 1]" = var_mean_60[1];  var_mean_60 = None
        sub_62: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_480, getitem_481);  convert_element_type_480 = getitem_481 = None
        add_244: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_480, 1e-05);  getitem_480 = None
        rsqrt_60: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        mul_240: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_60);  sub_62 = rsqrt_60 = None
        mul_241: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_240, arg363_1);  mul_240 = arg363_1 = None
        add_245: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_241, arg364_1);  mul_241 = arg364_1 = None
        convert_element_type_481: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_245, torch.float16);  add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_361: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_481, [-1, 1280]);  convert_element_type_481 = None
        addmm_120: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg365_1, view_361, arg366_1);  arg365_1 = view_361 = arg366_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_362: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_120, [1, 1024, 3840]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_30 = torch.ops.aten.split.Tensor(view_362, 1280, 2);  view_362 = None
        getitem_482: "f16[1, 1024, 1280]" = split_30[0]
        getitem_483: "f16[1, 1024, 1280]" = split_30[1]
        getitem_484: "f16[1, 1024, 1280]" = split_30[2];  split_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_365: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_482, [1, 1024, -1, 64]);  getitem_482 = None
        permute_122: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_363: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_483, [1, 1024, -1, 64]);  getitem_483 = None
        permute_120: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_363, [0, 2, 1, 3]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_364: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_484, [1, 1024, -1, 64]);  getitem_484 = None
        permute_121: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_62: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_30: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_62, full_default_61);  full_default_62 = full_default_61 = None
        _scaled_dot_product_cudnn_attention_30 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_122, permute_120, permute_121, where_30, False, scale = 0.125);  permute_122 = permute_120 = permute_121 = where_30 = None
        getitem_485: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_30[0];  _scaled_dot_product_cudnn_attention_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_123: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_485, [0, 2, 1, 3]);  getitem_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_366: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_123, [1, 1024, -1]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_367: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_366, [-1, 1280]);  view_366 = None
        addmm_121: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg367_1, view_367, arg368_1);  arg367_1 = view_367 = arg368_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_368: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_121, [1, 1024, 1280]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_246: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_368, add_243);  view_368 = add_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_488: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_246, torch.float32)
        var_mean_61 = torch.ops.aten.var_mean.correction(convert_element_type_488, [2], correction = 0, keepdim = True)
        getitem_494: "f32[1, 1024, 1]" = var_mean_61[0]
        getitem_495: "f32[1, 1024, 1]" = var_mean_61[1];  var_mean_61 = None
        sub_63: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_488, getitem_495);  convert_element_type_488 = getitem_495 = None
        add_247: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_494, 1e-05);  getitem_494 = None
        rsqrt_61: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_242: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_61);  sub_63 = rsqrt_61 = None
        mul_243: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_242, arg369_1);  mul_242 = arg369_1 = None
        add_248: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_243, arg370_1);  mul_243 = arg370_1 = None
        convert_element_type_489: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_248, torch.float16);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_369: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_489, [-1, 1280]);  convert_element_type_489 = None
        addmm_122: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg371_1, view_369, arg372_1);  arg371_1 = view_369 = arg372_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_370: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_122, [1, 1024, 5120]);  addmm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_244: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_370, 0.5)
        pow_31: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_370, 3.0)
        mul_245: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_31, 0.044715);  pow_31 = None
        add_249: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_370, mul_245);  view_370 = mul_245 = None
        mul_246: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_249, 0.7978845608028654);  add_249 = None
        tanh_30: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_246);  mul_246 = None
        add_250: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_30, 1.0);  tanh_30 = None
        mul_247: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_244, add_250);  mul_244 = add_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_371: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_247, [-1, 5120]);  mul_247 = None
        addmm_123: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg373_1, view_371, arg374_1);  arg373_1 = view_371 = arg374_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_372: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_123, [1, 1024, 1280]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_251: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_246, view_372);  add_246 = view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_496: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_251, torch.float32)
        var_mean_62 = torch.ops.aten.var_mean.correction(convert_element_type_496, [2], correction = 0, keepdim = True)
        getitem_496: "f32[1, 1024, 1]" = var_mean_62[0]
        getitem_497: "f32[1, 1024, 1]" = var_mean_62[1];  var_mean_62 = None
        sub_64: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_496, getitem_497);  convert_element_type_496 = getitem_497 = None
        add_252: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_496, 1e-05);  getitem_496 = None
        rsqrt_62: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
        mul_248: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_62);  sub_64 = rsqrt_62 = None
        mul_249: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_248, arg375_1);  mul_248 = arg375_1 = None
        add_253: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_249, arg376_1);  mul_249 = arg376_1 = None
        convert_element_type_497: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_253, torch.float16);  add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_373: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_497, [-1, 1280]);  convert_element_type_497 = None
        addmm_124: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg377_1, view_373, arg378_1);  arg377_1 = view_373 = arg378_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_374: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_124, [1, 1024, 3840]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_31 = torch.ops.aten.split.Tensor(view_374, 1280, 2);  view_374 = None
        getitem_498: "f16[1, 1024, 1280]" = split_31[0]
        getitem_499: "f16[1, 1024, 1280]" = split_31[1]
        getitem_500: "f16[1, 1024, 1280]" = split_31[2];  split_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_377: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_498, [1, 1024, -1, 64]);  getitem_498 = None
        permute_126: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_377, [0, 2, 1, 3]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_375: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_499, [1, 1024, -1, 64]);  getitem_499 = None
        permute_124: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_376: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_500, [1, 1024, -1, 64]);  getitem_500 = None
        permute_125: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_64: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_64, full_default_63);  full_default_64 = full_default_63 = None
        _scaled_dot_product_cudnn_attention_31 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_126, permute_124, permute_125, where_31, False, scale = 0.125);  permute_126 = permute_124 = permute_125 = where_31 = None
        getitem_501: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_31[0];  _scaled_dot_product_cudnn_attention_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_127: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_501, [0, 2, 1, 3]);  getitem_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_378: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_127, [1, 1024, -1]);  permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_379: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_378, [-1, 1280]);  view_378 = None
        addmm_125: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg379_1, view_379, arg380_1);  arg379_1 = view_379 = arg380_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_380: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_125, [1, 1024, 1280]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_254: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_380, add_251);  view_380 = add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_504: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_254, torch.float32)
        var_mean_63 = torch.ops.aten.var_mean.correction(convert_element_type_504, [2], correction = 0, keepdim = True)
        getitem_510: "f32[1, 1024, 1]" = var_mean_63[0]
        getitem_511: "f32[1, 1024, 1]" = var_mean_63[1];  var_mean_63 = None
        sub_65: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_504, getitem_511);  convert_element_type_504 = getitem_511 = None
        add_255: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_510, 1e-05);  getitem_510 = None
        rsqrt_63: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        mul_250: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_63);  sub_65 = rsqrt_63 = None
        mul_251: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_250, arg381_1);  mul_250 = arg381_1 = None
        add_256: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_251, arg382_1);  mul_251 = arg382_1 = None
        convert_element_type_505: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_256, torch.float16);  add_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_381: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_505, [-1, 1280]);  convert_element_type_505 = None
        addmm_126: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg383_1, view_381, arg384_1);  arg383_1 = view_381 = arg384_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_382: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_126, [1, 1024, 5120]);  addmm_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_252: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_382, 0.5)
        pow_32: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_382, 3.0)
        mul_253: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_32, 0.044715);  pow_32 = None
        add_257: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_382, mul_253);  view_382 = mul_253 = None
        mul_254: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_257, 0.7978845608028654);  add_257 = None
        tanh_31: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_254);  mul_254 = None
        add_258: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_31, 1.0);  tanh_31 = None
        mul_255: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_252, add_258);  mul_252 = add_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_383: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_255, [-1, 5120]);  mul_255 = None
        addmm_127: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg385_1, view_383, arg386_1);  arg385_1 = view_383 = arg386_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_384: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_127, [1, 1024, 1280]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_259: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_254, view_384);  add_254 = view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_512: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_259, torch.float32)
        var_mean_64 = torch.ops.aten.var_mean.correction(convert_element_type_512, [2], correction = 0, keepdim = True)
        getitem_512: "f32[1, 1024, 1]" = var_mean_64[0]
        getitem_513: "f32[1, 1024, 1]" = var_mean_64[1];  var_mean_64 = None
        sub_66: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_512, getitem_513);  convert_element_type_512 = getitem_513 = None
        add_260: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_512, 1e-05);  getitem_512 = None
        rsqrt_64: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
        mul_256: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_64);  sub_66 = rsqrt_64 = None
        mul_257: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_256, arg387_1);  mul_256 = arg387_1 = None
        add_261: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_257, arg388_1);  mul_257 = arg388_1 = None
        convert_element_type_513: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_261, torch.float16);  add_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_385: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_513, [-1, 1280]);  convert_element_type_513 = None
        addmm_128: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg389_1, view_385, arg390_1);  arg389_1 = view_385 = arg390_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_386: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_128, [1, 1024, 3840]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_32 = torch.ops.aten.split.Tensor(view_386, 1280, 2);  view_386 = None
        getitem_514: "f16[1, 1024, 1280]" = split_32[0]
        getitem_515: "f16[1, 1024, 1280]" = split_32[1]
        getitem_516: "f16[1, 1024, 1280]" = split_32[2];  split_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_389: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_514, [1, 1024, -1, 64]);  getitem_514 = None
        permute_130: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_387: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_515, [1, 1024, -1, 64]);  getitem_515 = None
        permute_128: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_388: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_516, [1, 1024, -1, 64]);  getitem_516 = None
        permute_129: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_66: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_65: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_66, full_default_65);  full_default_66 = full_default_65 = None
        _scaled_dot_product_cudnn_attention_32 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_130, permute_128, permute_129, where_32, False, scale = 0.125);  permute_130 = permute_128 = permute_129 = where_32 = None
        getitem_517: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_32[0];  _scaled_dot_product_cudnn_attention_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_131: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_517, [0, 2, 1, 3]);  getitem_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_390: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_131, [1, 1024, -1]);  permute_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_391: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_390, [-1, 1280]);  view_390 = None
        addmm_129: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg391_1, view_391, arg392_1);  arg391_1 = view_391 = arg392_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_392: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_129, [1, 1024, 1280]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_262: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_392, add_259);  view_392 = add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_520: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_262, torch.float32)
        var_mean_65 = torch.ops.aten.var_mean.correction(convert_element_type_520, [2], correction = 0, keepdim = True)
        getitem_526: "f32[1, 1024, 1]" = var_mean_65[0]
        getitem_527: "f32[1, 1024, 1]" = var_mean_65[1];  var_mean_65 = None
        sub_67: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_520, getitem_527);  convert_element_type_520 = getitem_527 = None
        add_263: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_526, 1e-05);  getitem_526 = None
        rsqrt_65: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
        mul_258: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_65);  sub_67 = rsqrt_65 = None
        mul_259: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_258, arg393_1);  mul_258 = arg393_1 = None
        add_264: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_259, arg394_1);  mul_259 = arg394_1 = None
        convert_element_type_521: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_264, torch.float16);  add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_393: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_521, [-1, 1280]);  convert_element_type_521 = None
        addmm_130: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg395_1, view_393, arg396_1);  arg395_1 = view_393 = arg396_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_394: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_130, [1, 1024, 5120]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_260: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_394, 0.5)
        pow_33: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_394, 3.0)
        mul_261: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_265: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_394, mul_261);  view_394 = mul_261 = None
        mul_262: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_265, 0.7978845608028654);  add_265 = None
        tanh_32: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_262);  mul_262 = None
        add_266: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_32, 1.0);  tanh_32 = None
        mul_263: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_260, add_266);  mul_260 = add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_395: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_263, [-1, 5120]);  mul_263 = None
        addmm_131: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg397_1, view_395, arg398_1);  arg397_1 = view_395 = arg398_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_396: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_131, [1, 1024, 1280]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_267: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_262, view_396);  add_262 = view_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_528: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_267, torch.float32)
        var_mean_66 = torch.ops.aten.var_mean.correction(convert_element_type_528, [2], correction = 0, keepdim = True)
        getitem_528: "f32[1, 1024, 1]" = var_mean_66[0]
        getitem_529: "f32[1, 1024, 1]" = var_mean_66[1];  var_mean_66 = None
        sub_68: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_528, getitem_529);  convert_element_type_528 = getitem_529 = None
        add_268: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_528, 1e-05);  getitem_528 = None
        rsqrt_66: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        mul_264: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_66);  sub_68 = rsqrt_66 = None
        mul_265: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_264, arg399_1);  mul_264 = arg399_1 = None
        add_269: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_265, arg400_1);  mul_265 = arg400_1 = None
        convert_element_type_529: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_269, torch.float16);  add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_397: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_529, [-1, 1280]);  convert_element_type_529 = None
        addmm_132: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg401_1, view_397, arg402_1);  arg401_1 = view_397 = arg402_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_398: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_132, [1, 1024, 3840]);  addmm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_33 = torch.ops.aten.split.Tensor(view_398, 1280, 2);  view_398 = None
        getitem_530: "f16[1, 1024, 1280]" = split_33[0]
        getitem_531: "f16[1, 1024, 1280]" = split_33[1]
        getitem_532: "f16[1, 1024, 1280]" = split_33[2];  split_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_401: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_530, [1, 1024, -1, 64]);  getitem_530 = None
        permute_134: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_399: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_531, [1, 1024, -1, 64]);  getitem_531 = None
        permute_132: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_400: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_532, [1, 1024, -1, 64]);  getitem_532 = None
        permute_133: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_68: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_67: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_33: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_68, full_default_67);  full_default_68 = full_default_67 = None
        _scaled_dot_product_cudnn_attention_33 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_134, permute_132, permute_133, where_33, False, scale = 0.125);  permute_134 = permute_132 = permute_133 = where_33 = None
        getitem_533: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_33[0];  _scaled_dot_product_cudnn_attention_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_135: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_533, [0, 2, 1, 3]);  getitem_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_402: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_135, [1, 1024, -1]);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_403: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_402, [-1, 1280]);  view_402 = None
        addmm_133: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg403_1, view_403, arg404_1);  arg403_1 = view_403 = arg404_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_404: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_133, [1, 1024, 1280]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_270: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_404, add_267);  view_404 = add_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_536: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_270, torch.float32)
        var_mean_67 = torch.ops.aten.var_mean.correction(convert_element_type_536, [2], correction = 0, keepdim = True)
        getitem_542: "f32[1, 1024, 1]" = var_mean_67[0]
        getitem_543: "f32[1, 1024, 1]" = var_mean_67[1];  var_mean_67 = None
        sub_69: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_536, getitem_543);  convert_element_type_536 = getitem_543 = None
        add_271: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_542, 1e-05);  getitem_542 = None
        rsqrt_67: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        mul_266: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_67);  sub_69 = rsqrt_67 = None
        mul_267: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_266, arg405_1);  mul_266 = arg405_1 = None
        add_272: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_267, arg406_1);  mul_267 = arg406_1 = None
        convert_element_type_537: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_272, torch.float16);  add_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_405: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_537, [-1, 1280]);  convert_element_type_537 = None
        addmm_134: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg407_1, view_405, arg408_1);  arg407_1 = view_405 = arg408_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_406: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_134, [1, 1024, 5120]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_268: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_406, 0.5)
        pow_34: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_406, 3.0)
        mul_269: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_34, 0.044715);  pow_34 = None
        add_273: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_406, mul_269);  view_406 = mul_269 = None
        mul_270: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_273, 0.7978845608028654);  add_273 = None
        tanh_33: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_270);  mul_270 = None
        add_274: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_33, 1.0);  tanh_33 = None
        mul_271: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_268, add_274);  mul_268 = add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_407: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_271, [-1, 5120]);  mul_271 = None
        addmm_135: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg409_1, view_407, arg410_1);  arg409_1 = view_407 = arg410_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_408: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_135, [1, 1024, 1280]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_275: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_270, view_408);  add_270 = view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_544: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_275, torch.float32)
        var_mean_68 = torch.ops.aten.var_mean.correction(convert_element_type_544, [2], correction = 0, keepdim = True)
        getitem_544: "f32[1, 1024, 1]" = var_mean_68[0]
        getitem_545: "f32[1, 1024, 1]" = var_mean_68[1];  var_mean_68 = None
        sub_70: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_544, getitem_545);  convert_element_type_544 = getitem_545 = None
        add_276: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_544, 1e-05);  getitem_544 = None
        rsqrt_68: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        mul_272: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_68);  sub_70 = rsqrt_68 = None
        mul_273: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_272, arg411_1);  mul_272 = arg411_1 = None
        add_277: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_273, arg412_1);  mul_273 = arg412_1 = None
        convert_element_type_545: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_277, torch.float16);  add_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_409: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_545, [-1, 1280]);  convert_element_type_545 = None
        addmm_136: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg413_1, view_409, arg414_1);  arg413_1 = view_409 = arg414_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_410: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_136, [1, 1024, 3840]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_34 = torch.ops.aten.split.Tensor(view_410, 1280, 2);  view_410 = None
        getitem_546: "f16[1, 1024, 1280]" = split_34[0]
        getitem_547: "f16[1, 1024, 1280]" = split_34[1]
        getitem_548: "f16[1, 1024, 1280]" = split_34[2];  split_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_413: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_546, [1, 1024, -1, 64]);  getitem_546 = None
        permute_138: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_411: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_547, [1, 1024, -1, 64]);  getitem_547 = None
        permute_136: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_412: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_548, [1, 1024, -1, 64]);  getitem_548 = None
        permute_137: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_70: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_69: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_34: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_70, full_default_69);  full_default_70 = full_default_69 = None
        _scaled_dot_product_cudnn_attention_34 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_138, permute_136, permute_137, where_34, False, scale = 0.125);  permute_138 = permute_136 = permute_137 = where_34 = None
        getitem_549: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_34[0];  _scaled_dot_product_cudnn_attention_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_139: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_549, [0, 2, 1, 3]);  getitem_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_414: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_139, [1, 1024, -1]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_415: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_414, [-1, 1280]);  view_414 = None
        addmm_137: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg415_1, view_415, arg416_1);  arg415_1 = view_415 = arg416_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_416: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_137, [1, 1024, 1280]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_278: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_416, add_275);  view_416 = add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_552: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_278, torch.float32)
        var_mean_69 = torch.ops.aten.var_mean.correction(convert_element_type_552, [2], correction = 0, keepdim = True)
        getitem_558: "f32[1, 1024, 1]" = var_mean_69[0]
        getitem_559: "f32[1, 1024, 1]" = var_mean_69[1];  var_mean_69 = None
        sub_71: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_552, getitem_559);  convert_element_type_552 = getitem_559 = None
        add_279: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_558, 1e-05);  getitem_558 = None
        rsqrt_69: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_279);  add_279 = None
        mul_274: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_69);  sub_71 = rsqrt_69 = None
        mul_275: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_274, arg417_1);  mul_274 = arg417_1 = None
        add_280: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_275, arg418_1);  mul_275 = arg418_1 = None
        convert_element_type_553: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_280, torch.float16);  add_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_417: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_553, [-1, 1280]);  convert_element_type_553 = None
        addmm_138: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg419_1, view_417, arg420_1);  arg419_1 = view_417 = arg420_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_418: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_138, [1, 1024, 5120]);  addmm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_276: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_418, 0.5)
        pow_35: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_418, 3.0)
        mul_277: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_35, 0.044715);  pow_35 = None
        add_281: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_418, mul_277);  view_418 = mul_277 = None
        mul_278: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_281, 0.7978845608028654);  add_281 = None
        tanh_34: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
        add_282: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_34, 1.0);  tanh_34 = None
        mul_279: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_276, add_282);  mul_276 = add_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_419: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_279, [-1, 5120]);  mul_279 = None
        addmm_139: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg421_1, view_419, arg422_1);  arg421_1 = view_419 = arg422_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_420: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_139, [1, 1024, 1280]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_283: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_278, view_420);  add_278 = view_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_560: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_283, torch.float32)
        var_mean_70 = torch.ops.aten.var_mean.correction(convert_element_type_560, [2], correction = 0, keepdim = True)
        getitem_560: "f32[1, 1024, 1]" = var_mean_70[0]
        getitem_561: "f32[1, 1024, 1]" = var_mean_70[1];  var_mean_70 = None
        sub_72: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_560, getitem_561);  convert_element_type_560 = getitem_561 = None
        add_284: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_560, 1e-05);  getitem_560 = None
        rsqrt_70: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_284);  add_284 = None
        mul_280: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_70);  sub_72 = rsqrt_70 = None
        mul_281: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_280, arg423_1);  mul_280 = arg423_1 = None
        add_285: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_281, arg424_1);  mul_281 = arg424_1 = None
        convert_element_type_561: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_285, torch.float16);  add_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_421: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_561, [-1, 1280]);  convert_element_type_561 = None
        addmm_140: "f16[1024, 3840]" = torch.ops.aten.addmm.default(arg425_1, view_421, arg426_1);  arg425_1 = view_421 = arg426_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_422: "f16[1, 1024, 3840]" = torch.ops.aten.reshape.default(addmm_140, [1, 1024, 3840]);  addmm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_35 = torch.ops.aten.split.Tensor(view_422, 1280, 2);  view_422 = None
        getitem_562: "f16[1, 1024, 1280]" = split_35[0]
        getitem_563: "f16[1, 1024, 1280]" = split_35[1]
        getitem_564: "f16[1, 1024, 1280]" = split_35[2];  split_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_425: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_562, [1, 1024, -1, 64]);  getitem_562 = None
        permute_142: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_423: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_563, [1, 1024, -1, 64]);  getitem_563 = None
        permute_140: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_424: "f16[1, 1024, 20, 64]" = torch.ops.aten.reshape.default(getitem_564, [1, 1024, -1, 64]);  getitem_564 = None
        permute_141: "f16[1, 20, 1024, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_72: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_71: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_35: "f16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default_72, full_default_71);  expand = full_default_72 = full_default_71 = None
        _scaled_dot_product_cudnn_attention_35 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_142, permute_140, permute_141, where_35, False, scale = 0.125);  permute_142 = permute_140 = permute_141 = where_35 = None
        getitem_565: "f16[1, 20, 1024, 64]" = _scaled_dot_product_cudnn_attention_35[0];  _scaled_dot_product_cudnn_attention_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_143: "f16[1, 1024, 20, 64]" = torch.ops.aten.permute.default(getitem_565, [0, 2, 1, 3]);  getitem_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_426: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(permute_143, [1, 1024, -1]);  permute_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_427: "f16[1024, 1280]" = torch.ops.aten.reshape.default(view_426, [-1, 1280]);  view_426 = None
        addmm_141: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg427_1, view_427, arg428_1);  arg427_1 = view_427 = arg428_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_428: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_141, [1, 1024, 1280]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_286: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(view_428, add_283);  view_428 = add_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_568: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_286, torch.float32)
        var_mean_71 = torch.ops.aten.var_mean.correction(convert_element_type_568, [2], correction = 0, keepdim = True)
        getitem_574: "f32[1, 1024, 1]" = var_mean_71[0]
        getitem_575: "f32[1, 1024, 1]" = var_mean_71[1];  var_mean_71 = None
        sub_73: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_568, getitem_575);  convert_element_type_568 = getitem_575 = None
        add_287: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_574, 1e-05);  getitem_574 = None
        rsqrt_71: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_287);  add_287 = None
        mul_282: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_71);  sub_73 = rsqrt_71 = None
        mul_283: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_282, arg429_1);  mul_282 = arg429_1 = None
        add_288: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_283, arg430_1);  mul_283 = arg430_1 = None
        convert_element_type_569: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_288, torch.float16);  add_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_429: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_569, [-1, 1280]);  convert_element_type_569 = None
        addmm_142: "f16[1024, 5120]" = torch.ops.aten.addmm.default(arg431_1, view_429, arg432_1);  arg431_1 = view_429 = arg432_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_430: "f16[1, 1024, 5120]" = torch.ops.aten.reshape.default(addmm_142, [1, 1024, 5120]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_284: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(view_430, 0.5)
        pow_36: "f16[1, 1024, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_430, 3.0)
        mul_285: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(pow_36, 0.044715);  pow_36 = None
        add_289: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(view_430, mul_285);  view_430 = mul_285 = None
        mul_286: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(add_289, 0.7978845608028654);  add_289 = None
        tanh_35: "f16[1, 1024, 5120]" = torch.ops.aten.tanh.default(mul_286);  mul_286 = None
        add_290: "f16[1, 1024, 5120]" = torch.ops.aten.add.Tensor(tanh_35, 1.0);  tanh_35 = None
        mul_287: "f16[1, 1024, 5120]" = torch.ops.aten.mul.Tensor(mul_284, add_290);  mul_284 = add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_431: "f16[1024, 5120]" = torch.ops.aten.reshape.default(mul_287, [-1, 5120]);  mul_287 = None
        addmm_143: "f16[1024, 1280]" = torch.ops.aten.addmm.default(arg433_1, view_431, arg434_1);  arg433_1 = view_431 = arg434_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_432: "f16[1, 1024, 1280]" = torch.ops.aten.reshape.default(addmm_143, [1, 1024, 1280]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_291: "f16[1, 1024, 1280]" = torch.ops.aten.add.Tensor(add_286, view_432);  add_286 = view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        convert_element_type_576: "f32[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_291, torch.float32);  add_291 = None
        var_mean_72 = torch.ops.aten.var_mean.correction(convert_element_type_576, [2], correction = 0, keepdim = True)
        getitem_576: "f32[1, 1024, 1]" = var_mean_72[0]
        getitem_577: "f32[1, 1024, 1]" = var_mean_72[1];  var_mean_72 = None
        sub_74: "f32[1, 1024, 1280]" = torch.ops.aten.sub.Tensor(convert_element_type_576, getitem_577);  convert_element_type_576 = getitem_577 = None
        add_292: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_576, 1e-05);  getitem_576 = None
        rsqrt_72: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        mul_288: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_72);  sub_74 = rsqrt_72 = None
        mul_289: "f32[1, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_288, arg435_1);  mul_288 = arg435_1 = None
        add_293: "f32[1, 1024, 1280]" = torch.ops.aten.add.Tensor(mul_289, arg436_1);  mul_289 = arg436_1 = None
        convert_element_type_577: "f16[1, 1024, 1280]" = torch.ops.prims.convert_element_type.default(add_293, torch.float16);  add_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:706 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_434: "f16[1024, 1280]" = torch.ops.aten.reshape.default(convert_element_type_577, [1024, 1280]);  convert_element_type_577 = None
        permute_144: "f16[1280, 50257]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "f16[1280, 50264]" = torch.ops.aten.constant_pad_nd.default(permute_144, [0, 7, 0, 0]);  permute_144 = None
        mm_default: "f16[1024, 50264]" = torch.ops.aten.mm.default(view_434, constant_pad_nd_default);  view_434 = constant_pad_nd_default = None
        slice_tensor: "f16[1024, 50257]" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -7);  mm_default = None
        view_435: "f16[1, 1024, 50257]" = torch.ops.aten.reshape.default(slice_tensor, [1, 1024, 50257]);  slice_tensor = None
        return (view_435,)
