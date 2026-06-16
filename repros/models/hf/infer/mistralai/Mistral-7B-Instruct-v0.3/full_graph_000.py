class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 1000][1000, 1]cuda:0", arg1_1: "bf16[32768, 4096][4096, 1]cuda:0", arg2_1: "bf16[64][1]cuda:0", arg3_1: "bf16[4096][1]cuda:0", arg4_1: "bf16[4096, 4096][4096, 1]cuda:0", arg5_1: "bf16[1024, 4096][4096, 1]cuda:0", arg6_1: "bf16[1024, 4096][4096, 1]cuda:0", arg7_1: "bf16[4096, 4096][4096, 1]cuda:0", arg8_1: "bf16[4096][1]cuda:0", arg9_1: "bf16[14336, 4096][4096, 1]cuda:0", arg10_1: "bf16[14336, 4096][4096, 1]cuda:0", arg11_1: "bf16[4096, 14336][14336, 1]cuda:0", arg12_1: "bf16[4096][1]cuda:0", arg13_1: "bf16[4096, 4096][4096, 1]cuda:0", arg14_1: "bf16[1024, 4096][4096, 1]cuda:0", arg15_1: "bf16[1024, 4096][4096, 1]cuda:0", arg16_1: "bf16[4096, 4096][4096, 1]cuda:0", arg17_1: "bf16[4096][1]cuda:0", arg18_1: "bf16[14336, 4096][4096, 1]cuda:0", arg19_1: "bf16[14336, 4096][4096, 1]cuda:0", arg20_1: "bf16[4096, 14336][14336, 1]cuda:0", arg21_1: "bf16[4096][1]cuda:0", arg22_1: "bf16[4096, 4096][4096, 1]cuda:0", arg23_1: "bf16[1024, 4096][4096, 1]cuda:0", arg24_1: "bf16[1024, 4096][4096, 1]cuda:0", arg25_1: "bf16[4096, 4096][4096, 1]cuda:0", arg26_1: "bf16[4096][1]cuda:0", arg27_1: "bf16[14336, 4096][4096, 1]cuda:0", arg28_1: "bf16[14336, 4096][4096, 1]cuda:0", arg29_1: "bf16[4096, 14336][14336, 1]cuda:0", arg30_1: "bf16[4096][1]cuda:0", arg31_1: "bf16[4096, 4096][4096, 1]cuda:0", arg32_1: "bf16[1024, 4096][4096, 1]cuda:0", arg33_1: "bf16[1024, 4096][4096, 1]cuda:0", arg34_1: "bf16[4096, 4096][4096, 1]cuda:0", arg35_1: "bf16[4096][1]cuda:0", arg36_1: "bf16[14336, 4096][4096, 1]cuda:0", arg37_1: "bf16[14336, 4096][4096, 1]cuda:0", arg38_1: "bf16[4096, 14336][14336, 1]cuda:0", arg39_1: "bf16[4096][1]cuda:0", arg40_1: "bf16[4096, 4096][4096, 1]cuda:0", arg41_1: "bf16[1024, 4096][4096, 1]cuda:0", arg42_1: "bf16[1024, 4096][4096, 1]cuda:0", arg43_1: "bf16[4096, 4096][4096, 1]cuda:0", arg44_1: "bf16[4096][1]cuda:0", arg45_1: "bf16[14336, 4096][4096, 1]cuda:0", arg46_1: "bf16[14336, 4096][4096, 1]cuda:0", arg47_1: "bf16[4096, 14336][14336, 1]cuda:0", arg48_1: "bf16[4096][1]cuda:0", arg49_1: "bf16[4096, 4096][4096, 1]cuda:0", arg50_1: "bf16[1024, 4096][4096, 1]cuda:0", arg51_1: "bf16[1024, 4096][4096, 1]cuda:0", arg52_1: "bf16[4096, 4096][4096, 1]cuda:0", arg53_1: "bf16[4096][1]cuda:0", arg54_1: "bf16[14336, 4096][4096, 1]cuda:0", arg55_1: "bf16[14336, 4096][4096, 1]cuda:0", arg56_1: "bf16[4096, 14336][14336, 1]cuda:0", arg57_1: "bf16[4096][1]cuda:0", arg58_1: "bf16[4096, 4096][4096, 1]cuda:0", arg59_1: "bf16[1024, 4096][4096, 1]cuda:0", arg60_1: "bf16[1024, 4096][4096, 1]cuda:0", arg61_1: "bf16[4096, 4096][4096, 1]cuda:0", arg62_1: "bf16[4096][1]cuda:0", arg63_1: "bf16[14336, 4096][4096, 1]cuda:0", arg64_1: "bf16[14336, 4096][4096, 1]cuda:0", arg65_1: "bf16[4096, 14336][14336, 1]cuda:0", arg66_1: "bf16[4096][1]cuda:0", arg67_1: "bf16[4096, 4096][4096, 1]cuda:0", arg68_1: "bf16[1024, 4096][4096, 1]cuda:0", arg69_1: "bf16[1024, 4096][4096, 1]cuda:0", arg70_1: "bf16[4096, 4096][4096, 1]cuda:0", arg71_1: "bf16[4096][1]cuda:0", arg72_1: "bf16[14336, 4096][4096, 1]cuda:0", arg73_1: "bf16[14336, 4096][4096, 1]cuda:0", arg74_1: "bf16[4096, 14336][14336, 1]cuda:0", arg75_1: "bf16[4096][1]cuda:0", arg76_1: "bf16[4096, 4096][4096, 1]cuda:0", arg77_1: "bf16[1024, 4096][4096, 1]cuda:0", arg78_1: "bf16[1024, 4096][4096, 1]cuda:0", arg79_1: "bf16[4096, 4096][4096, 1]cuda:0", arg80_1: "bf16[4096][1]cuda:0", arg81_1: "bf16[14336, 4096][4096, 1]cuda:0", arg82_1: "bf16[14336, 4096][4096, 1]cuda:0", arg83_1: "bf16[4096, 14336][14336, 1]cuda:0", arg84_1: "bf16[4096][1]cuda:0", arg85_1: "bf16[4096, 4096][4096, 1]cuda:0", arg86_1: "bf16[1024, 4096][4096, 1]cuda:0", arg87_1: "bf16[1024, 4096][4096, 1]cuda:0", arg88_1: "bf16[4096, 4096][4096, 1]cuda:0", arg89_1: "bf16[4096][1]cuda:0", arg90_1: "bf16[14336, 4096][4096, 1]cuda:0", arg91_1: "bf16[14336, 4096][4096, 1]cuda:0", arg92_1: "bf16[4096, 14336][14336, 1]cuda:0", arg93_1: "bf16[4096][1]cuda:0", arg94_1: "bf16[4096, 4096][4096, 1]cuda:0", arg95_1: "bf16[1024, 4096][4096, 1]cuda:0", arg96_1: "bf16[1024, 4096][4096, 1]cuda:0", arg97_1: "bf16[4096, 4096][4096, 1]cuda:0", arg98_1: "bf16[4096][1]cuda:0", arg99_1: "bf16[14336, 4096][4096, 1]cuda:0", arg100_1: "bf16[14336, 4096][4096, 1]cuda:0", arg101_1: "bf16[4096, 14336][14336, 1]cuda:0", arg102_1: "bf16[4096][1]cuda:0", arg103_1: "bf16[4096, 4096][4096, 1]cuda:0", arg104_1: "bf16[1024, 4096][4096, 1]cuda:0", arg105_1: "bf16[1024, 4096][4096, 1]cuda:0", arg106_1: "bf16[4096, 4096][4096, 1]cuda:0", arg107_1: "bf16[4096][1]cuda:0", arg108_1: "bf16[14336, 4096][4096, 1]cuda:0", arg109_1: "bf16[14336, 4096][4096, 1]cuda:0", arg110_1: "bf16[4096, 14336][14336, 1]cuda:0", arg111_1: "bf16[4096][1]cuda:0", arg112_1: "bf16[4096, 4096][4096, 1]cuda:0", arg113_1: "bf16[1024, 4096][4096, 1]cuda:0", arg114_1: "bf16[1024, 4096][4096, 1]cuda:0", arg115_1: "bf16[4096, 4096][4096, 1]cuda:0", arg116_1: "bf16[4096][1]cuda:0", arg117_1: "bf16[14336, 4096][4096, 1]cuda:0", arg118_1: "bf16[14336, 4096][4096, 1]cuda:0", arg119_1: "bf16[4096, 14336][14336, 1]cuda:0", arg120_1: "bf16[4096][1]cuda:0", arg121_1: "bf16[4096, 4096][4096, 1]cuda:0", arg122_1: "bf16[1024, 4096][4096, 1]cuda:0", arg123_1: "bf16[1024, 4096][4096, 1]cuda:0", arg124_1: "bf16[4096, 4096][4096, 1]cuda:0", arg125_1: "bf16[4096][1]cuda:0", arg126_1: "bf16[14336, 4096][4096, 1]cuda:0", arg127_1: "bf16[14336, 4096][4096, 1]cuda:0", arg128_1: "bf16[4096, 14336][14336, 1]cuda:0", arg129_1: "bf16[4096][1]cuda:0", arg130_1: "bf16[4096, 4096][4096, 1]cuda:0", arg131_1: "bf16[1024, 4096][4096, 1]cuda:0", arg132_1: "bf16[1024, 4096][4096, 1]cuda:0", arg133_1: "bf16[4096, 4096][4096, 1]cuda:0", arg134_1: "bf16[4096][1]cuda:0", arg135_1: "bf16[14336, 4096][4096, 1]cuda:0", arg136_1: "bf16[14336, 4096][4096, 1]cuda:0", arg137_1: "bf16[4096, 14336][14336, 1]cuda:0", arg138_1: "bf16[4096][1]cuda:0", arg139_1: "bf16[4096, 4096][4096, 1]cuda:0", arg140_1: "bf16[1024, 4096][4096, 1]cuda:0", arg141_1: "bf16[1024, 4096][4096, 1]cuda:0", arg142_1: "bf16[4096, 4096][4096, 1]cuda:0", arg143_1: "bf16[4096][1]cuda:0", arg144_1: "bf16[14336, 4096][4096, 1]cuda:0", arg145_1: "bf16[14336, 4096][4096, 1]cuda:0", arg146_1: "bf16[4096, 14336][14336, 1]cuda:0", arg147_1: "bf16[4096][1]cuda:0", arg148_1: "bf16[4096, 4096][4096, 1]cuda:0", arg149_1: "bf16[1024, 4096][4096, 1]cuda:0", arg150_1: "bf16[1024, 4096][4096, 1]cuda:0", arg151_1: "bf16[4096, 4096][4096, 1]cuda:0", arg152_1: "bf16[4096][1]cuda:0", arg153_1: "bf16[14336, 4096][4096, 1]cuda:0", arg154_1: "bf16[14336, 4096][4096, 1]cuda:0", arg155_1: "bf16[4096, 14336][14336, 1]cuda:0", arg156_1: "bf16[4096][1]cuda:0", arg157_1: "bf16[4096, 4096][4096, 1]cuda:0", arg158_1: "bf16[1024, 4096][4096, 1]cuda:0", arg159_1: "bf16[1024, 4096][4096, 1]cuda:0", arg160_1: "bf16[4096, 4096][4096, 1]cuda:0", arg161_1: "bf16[4096][1]cuda:0", arg162_1: "bf16[14336, 4096][4096, 1]cuda:0", arg163_1: "bf16[14336, 4096][4096, 1]cuda:0", arg164_1: "bf16[4096, 14336][14336, 1]cuda:0", arg165_1: "bf16[4096][1]cuda:0", arg166_1: "bf16[4096, 4096][4096, 1]cuda:0", arg167_1: "bf16[1024, 4096][4096, 1]cuda:0", arg168_1: "bf16[1024, 4096][4096, 1]cuda:0", arg169_1: "bf16[4096, 4096][4096, 1]cuda:0", arg170_1: "bf16[4096][1]cuda:0", arg171_1: "bf16[14336, 4096][4096, 1]cuda:0", arg172_1: "bf16[14336, 4096][4096, 1]cuda:0", arg173_1: "bf16[4096, 14336][14336, 1]cuda:0", arg174_1: "bf16[4096][1]cuda:0", arg175_1: "bf16[4096, 4096][4096, 1]cuda:0", arg176_1: "bf16[1024, 4096][4096, 1]cuda:0", arg177_1: "bf16[1024, 4096][4096, 1]cuda:0", arg178_1: "bf16[4096, 4096][4096, 1]cuda:0", arg179_1: "bf16[4096][1]cuda:0", arg180_1: "bf16[14336, 4096][4096, 1]cuda:0", arg181_1: "bf16[14336, 4096][4096, 1]cuda:0", arg182_1: "bf16[4096, 14336][14336, 1]cuda:0", arg183_1: "bf16[4096][1]cuda:0", arg184_1: "bf16[4096, 4096][4096, 1]cuda:0", arg185_1: "bf16[1024, 4096][4096, 1]cuda:0", arg186_1: "bf16[1024, 4096][4096, 1]cuda:0", arg187_1: "bf16[4096, 4096][4096, 1]cuda:0", arg188_1: "bf16[4096][1]cuda:0", arg189_1: "bf16[14336, 4096][4096, 1]cuda:0", arg190_1: "bf16[14336, 4096][4096, 1]cuda:0", arg191_1: "bf16[4096, 14336][14336, 1]cuda:0", arg192_1: "bf16[4096][1]cuda:0", arg193_1: "bf16[4096, 4096][4096, 1]cuda:0", arg194_1: "bf16[1024, 4096][4096, 1]cuda:0", arg195_1: "bf16[1024, 4096][4096, 1]cuda:0", arg196_1: "bf16[4096, 4096][4096, 1]cuda:0", arg197_1: "bf16[4096][1]cuda:0", arg198_1: "bf16[14336, 4096][4096, 1]cuda:0", arg199_1: "bf16[14336, 4096][4096, 1]cuda:0", arg200_1: "bf16[4096, 14336][14336, 1]cuda:0", arg201_1: "bf16[4096][1]cuda:0", arg202_1: "bf16[4096, 4096][4096, 1]cuda:0", arg203_1: "bf16[1024, 4096][4096, 1]cuda:0", arg204_1: "bf16[1024, 4096][4096, 1]cuda:0", arg205_1: "bf16[4096, 4096][4096, 1]cuda:0", arg206_1: "bf16[4096][1]cuda:0", arg207_1: "bf16[14336, 4096][4096, 1]cuda:0", arg208_1: "bf16[14336, 4096][4096, 1]cuda:0", arg209_1: "bf16[4096, 14336][14336, 1]cuda:0", arg210_1: "bf16[4096][1]cuda:0", arg211_1: "bf16[4096, 4096][4096, 1]cuda:0", arg212_1: "bf16[1024, 4096][4096, 1]cuda:0", arg213_1: "bf16[1024, 4096][4096, 1]cuda:0", arg214_1: "bf16[4096, 4096][4096, 1]cuda:0", arg215_1: "bf16[4096][1]cuda:0", arg216_1: "bf16[14336, 4096][4096, 1]cuda:0", arg217_1: "bf16[14336, 4096][4096, 1]cuda:0", arg218_1: "bf16[4096, 14336][14336, 1]cuda:0", arg219_1: "bf16[4096][1]cuda:0", arg220_1: "bf16[4096, 4096][4096, 1]cuda:0", arg221_1: "bf16[1024, 4096][4096, 1]cuda:0", arg222_1: "bf16[1024, 4096][4096, 1]cuda:0", arg223_1: "bf16[4096, 4096][4096, 1]cuda:0", arg224_1: "bf16[4096][1]cuda:0", arg225_1: "bf16[14336, 4096][4096, 1]cuda:0", arg226_1: "bf16[14336, 4096][4096, 1]cuda:0", arg227_1: "bf16[4096, 14336][14336, 1]cuda:0", arg228_1: "bf16[4096][1]cuda:0", arg229_1: "bf16[4096, 4096][4096, 1]cuda:0", arg230_1: "bf16[1024, 4096][4096, 1]cuda:0", arg231_1: "bf16[1024, 4096][4096, 1]cuda:0", arg232_1: "bf16[4096, 4096][4096, 1]cuda:0", arg233_1: "bf16[4096][1]cuda:0", arg234_1: "bf16[14336, 4096][4096, 1]cuda:0", arg235_1: "bf16[14336, 4096][4096, 1]cuda:0", arg236_1: "bf16[4096, 14336][14336, 1]cuda:0", arg237_1: "bf16[4096][1]cuda:0", arg238_1: "bf16[4096, 4096][4096, 1]cuda:0", arg239_1: "bf16[1024, 4096][4096, 1]cuda:0", arg240_1: "bf16[1024, 4096][4096, 1]cuda:0", arg241_1: "bf16[4096, 4096][4096, 1]cuda:0", arg242_1: "bf16[4096][1]cuda:0", arg243_1: "bf16[14336, 4096][4096, 1]cuda:0", arg244_1: "bf16[14336, 4096][4096, 1]cuda:0", arg245_1: "bf16[4096, 14336][14336, 1]cuda:0", arg246_1: "bf16[4096][1]cuda:0", arg247_1: "bf16[4096, 4096][4096, 1]cuda:0", arg248_1: "bf16[1024, 4096][4096, 1]cuda:0", arg249_1: "bf16[1024, 4096][4096, 1]cuda:0", arg250_1: "bf16[4096, 4096][4096, 1]cuda:0", arg251_1: "bf16[4096][1]cuda:0", arg252_1: "bf16[14336, 4096][4096, 1]cuda:0", arg253_1: "bf16[14336, 4096][4096, 1]cuda:0", arg254_1: "bf16[4096, 14336][14336, 1]cuda:0", arg255_1: "bf16[4096][1]cuda:0", arg256_1: "bf16[4096, 4096][4096, 1]cuda:0", arg257_1: "bf16[1024, 4096][4096, 1]cuda:0", arg258_1: "bf16[1024, 4096][4096, 1]cuda:0", arg259_1: "bf16[4096, 4096][4096, 1]cuda:0", arg260_1: "bf16[4096][1]cuda:0", arg261_1: "bf16[14336, 4096][4096, 1]cuda:0", arg262_1: "bf16[14336, 4096][4096, 1]cuda:0", arg263_1: "bf16[4096, 14336][14336, 1]cuda:0", arg264_1: "bf16[4096][1]cuda:0", arg265_1: "bf16[4096, 4096][4096, 1]cuda:0", arg266_1: "bf16[1024, 4096][4096, 1]cuda:0", arg267_1: "bf16[1024, 4096][4096, 1]cuda:0", arg268_1: "bf16[4096, 4096][4096, 1]cuda:0", arg269_1: "bf16[4096][1]cuda:0", arg270_1: "bf16[14336, 4096][4096, 1]cuda:0", arg271_1: "bf16[14336, 4096][4096, 1]cuda:0", arg272_1: "bf16[4096, 14336][14336, 1]cuda:0", arg273_1: "bf16[4096][1]cuda:0", arg274_1: "bf16[4096, 4096][4096, 1]cuda:0", arg275_1: "bf16[1024, 4096][4096, 1]cuda:0", arg276_1: "bf16[1024, 4096][4096, 1]cuda:0", arg277_1: "bf16[4096, 4096][4096, 1]cuda:0", arg278_1: "bf16[4096][1]cuda:0", arg279_1: "bf16[14336, 4096][4096, 1]cuda:0", arg280_1: "bf16[14336, 4096][4096, 1]cuda:0", arg281_1: "bf16[4096, 14336][14336, 1]cuda:0", arg282_1: "bf16[4096][1]cuda:0", arg283_1: "bf16[4096, 4096][4096, 1]cuda:0", arg284_1: "bf16[1024, 4096][4096, 1]cuda:0", arg285_1: "bf16[1024, 4096][4096, 1]cuda:0", arg286_1: "bf16[4096, 4096][4096, 1]cuda:0", arg287_1: "bf16[4096][1]cuda:0", arg288_1: "bf16[14336, 4096][4096, 1]cuda:0", arg289_1: "bf16[14336, 4096][4096, 1]cuda:0", arg290_1: "bf16[4096, 14336][14336, 1]cuda:0", arg291_1: "bf16[4096][1]cuda:0", arg292_1: "bf16[32768, 4096][4096, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:362 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_4: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_4, 2)
        mean: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_3: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_3: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_4, rsqrt);  convert_element_type_4 = rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_5: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_4: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg3_1, convert_element_type_5);  arg3_1 = convert_element_type_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_4: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_4, [1000, 4096])
        permute_1: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
        view_5: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [1, 1000, 4096]);  mm = None
        view_6: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [1, 1000, -1, 128]);  view_5 = None
        permute_2: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:314 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_7: "bf16[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_8: "bf16[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        convert_element_type: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_8, torch.float32);  unsqueeze_8 = None
        expand_1: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type, [1, -1, 1]);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:319 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_2: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.expand.default(expand_1, [1, 64, 1]);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:369 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[1000][1]cuda:0" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1000][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:370 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 1000][1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:315 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_9: "i64[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        convert_element_type_1: "f32[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_9, torch.float32);  unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:319 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_3: "f32[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_1, [1, 1, 1000]);  convert_element_type_1 = None
        mul: "f32[1, 64, 1000][64000, 1000, 1]cuda:0" = torch.ops.aten.mul.Tensor(expand_2, expand_3);  expand_2 = expand_3 = None
        permute: "f32[1, 1000, 64][64000, 1, 1000]cuda:0" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:320 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_10: "f32[1, 1000, 1, 64][64000, 1, 64000, 1000]cuda:0" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_4: "f32[1, 1000, 2, 64][64000, 1, 0, 1000]cuda:0" = torch.ops.aten.expand.default(unsqueeze_10, [1, 1000, 2, 64]);  unsqueeze_10 = None
        clone: "f32[1, 1000, 2, 64][128000, 128, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_3: "f32[1, 1000, 128][128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [1, 1000, 128]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:321 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 1000, 128][128000, 128, 1]cuda:0" = torch.ops.aten.cos.default(view_3)
        mul_1: "f32[1, 1000, 128][128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:324 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "bf16[1, 1000, 128][128000, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_11: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_5: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_2, unsqueeze_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_2: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_2, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_2);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_1: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_2, 3, 0, 64);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg, slice_1], -1);  neg = slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:322 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 1000, 128][128000, 128, 1]cuda:0" = torch.ops.aten.sin.default(view_3);  view_3 = None
        mul_2: "f32[1, 1000, 128][128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:324 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_3: "bf16[1, 1000, 128][128000, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_12: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_6: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat, unsqueeze_12);  cat = None
        add_4: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_7: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_4, [1000, 4096])
        permute_3: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_7, permute_3);  view_7 = permute_3 = None
        view_8: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [1, 1000, 1024]);  mm_1 = None
        view_9: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_8, [1, 1000, -1, 128]);  view_8 = None
        permute_4: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_7: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_11);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_4: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_4, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_4);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_3: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_4, 3, 0, 64);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_1: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_1, slice_3], -1);  neg_1 = slice_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_8: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_12);  cat_1 = unsqueeze_12 = None
        add_5: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_13: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_5, 2)
        expand_5: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_13, [1, 8, 4, 1000, 128]);  unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_4: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_13: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [1, 32, 1000, 128]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_10: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_4, [1000, 4096]);  mul_4 = None
        permute_5: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_2: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
        view_11: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [1, 1000, 1024]);  mm_2 = None
        view_12: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_11, [1, 1000, -1, 128]);  view_11 = None
        permute_6: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_14: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_6, 2)
        expand_6: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_14, [1, 8, 4, 1000, 128]);  unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_5: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_14: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [1, 32, 1000, 128]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[1000][1]cuda:0" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[1000][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_4: "i64[1, 1000][1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_5: "i64[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 1, 1000][1000, 1000, 1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[1000][1]cuda:0" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[1000][1]cuda:0" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_1: "i64[1, 1000][1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_2: "i64[1, 1, 1000][1000, 1000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 1);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 1000, 1][1000, 1000, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_6, unsqueeze_3);  unsqueeze_6 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.expand.default(le, [1, -1, 1000, 1000]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_1, full_default);  full_default_1 = full_default = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_4, view_13, view_14, where, False, scale = 0.08838834764831845);  add_4 = view_13 = view_14 = where = None
        getitem: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_7, [1, 1000, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_16: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [1000, 4096]);  view_15 = None
        permute_8: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_3: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
        view_17: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [1, 1000, 4096]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_6: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, view_17);  embedding = view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_14: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_2: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_14, 2)
        mean_1: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_7: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        rsqrt_1: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_9: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_14, rsqrt_1);  convert_element_type_14 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_15: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        mul_10: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg8_1, convert_element_type_15);  arg8_1 = convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_18: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_10, [1000, 4096])
        permute_9: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_4: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_18, permute_9);  view_18 = permute_9 = None
        view_19: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [1, 1000, 14336]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_18: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        neg_2: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_18)
        exp: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_8: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_18, add_8);  convert_element_type_18 = add_8 = None
        convert_element_type_19: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_20: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_10, [1000, 4096]);  mul_10 = None
        permute_10: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_5: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_20, permute_10);  view_20 = permute_10 = None
        view_21: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [1, 1000, 14336]);  mm_5 = None
        mul_11: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_19, view_21);  convert_element_type_19 = view_21 = None
        view_22: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_11, [1000, 14336]);  mul_11 = None
        permute_11: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_6: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
        view_23: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [1, 1000, 4096]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_9: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_6, view_23);  add_6 = view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_24: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_3: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_24, 2)
        mean_2: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_2, 1e-05);  mean_2 = None
        rsqrt_2: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_12: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, rsqrt_2);  convert_element_type_24 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_25: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        mul_13: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg12_1, convert_element_type_25);  arg12_1 = convert_element_type_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_24: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_13, [1000, 4096])
        permute_12: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_7: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_24, permute_12);  view_24 = permute_12 = None
        view_25: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [1, 1000, 4096]);  mm_7 = None
        view_26: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_25, [1, 1000, -1, 128]);  view_25 = None
        permute_13: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_15: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_14: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_13, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_6: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_13, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_3: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_6);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_5: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_13, 3, 0, 64);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_2: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_3, slice_5], -1);  neg_3 = slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_16: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_15: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_16);  cat_2 = None
        add_11: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_27: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_13, [1000, 4096])
        permute_14: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        mm_8: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_27, permute_14);  view_27 = permute_14 = None
        view_28: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [1, 1000, 1024]);  mm_8 = None
        view_29: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_28, [1, 1000, -1, 128]);  view_28 = None
        permute_15: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_16: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_15, unsqueeze_15);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_8: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_15, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_4: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_8);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_7: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_15, 3, 0, 64);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_3: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_4, slice_7], -1);  neg_4 = slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_17: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_16);  cat_3 = unsqueeze_16 = None
        add_12: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_17: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_12, 2)
        expand_7: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_17, [1, 8, 4, 1000, 128]);  unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_8: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_33: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [1, 32, 1000, 128]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_30: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_13, [1000, 4096]);  mul_13 = None
        permute_16: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_9: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_30, permute_16);  view_30 = permute_16 = None
        view_31: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [1, 1000, 1024]);  mm_9 = None
        view_32: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_31, [1, 1000, -1, 128]);  view_31 = None
        permute_17: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_18: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_17, 2)
        expand_8: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_18, [1, 8, 4, 1000, 128]);  unsqueeze_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_9: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_34: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [1, 32, 1000, 128]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_3, full_default_2);  full_default_3 = full_default_2 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_11, view_33, view_34, where_1, False, scale = 0.08838834764831845);  add_11 = view_33 = view_34 = where_1 = None
        getitem_9: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_35: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_18, [1, 1000, -1]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_36: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_35, [1000, 4096]);  view_35 = None
        permute_19: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_10: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_36, permute_19);  view_36 = permute_19 = None
        view_37: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [1, 1000, 4096]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_13: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_9, view_37);  add_9 = view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_34: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_4: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_34, 2)
        mean_3: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_14: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_3, 1e-05);  mean_3 = None
        rsqrt_3: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_18: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, rsqrt_3);  convert_element_type_34 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_35: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        mul_19: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg17_1, convert_element_type_35);  arg17_1 = convert_element_type_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_38: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [1000, 4096])
        permute_20: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_11: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_38, permute_20);  view_38 = permute_20 = None
        view_39: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [1, 1000, 14336]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_38: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        neg_5: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_38)
        exp_1: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_15: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_38, add_15);  convert_element_type_38 = add_15 = None
        convert_element_type_39: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_40: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [1000, 4096]);  mul_19 = None
        permute_21: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_12: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_40, permute_21);  view_40 = permute_21 = None
        view_41: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [1, 1000, 14336]);  mm_12 = None
        mul_20: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, view_41);  convert_element_type_39 = view_41 = None
        view_42: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_20, [1000, 14336]);  mul_20 = None
        permute_22: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_13: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_42, permute_22);  view_42 = permute_22 = None
        view_43: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [1, 1000, 4096]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_16: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_13, view_43);  add_13 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_44: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_5: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_44, 2)
        mean_4: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_4, 1e-05);  mean_4 = None
        rsqrt_4: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_21: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, rsqrt_4);  convert_element_type_44 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_45: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        mul_22: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg21_1, convert_element_type_45);  arg21_1 = convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_44: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_22, [1000, 4096])
        permute_23: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_14: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_44, permute_23);  view_44 = permute_23 = None
        view_45: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [1, 1000, 4096]);  mm_14 = None
        view_46: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [1, 1000, -1, 128]);  view_45 = None
        permute_24: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_19: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_23: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_24, unsqueeze_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_10: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_24, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_6: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_10);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_9: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_24, 3, 0, 64);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_4: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_6, slice_9], -1);  neg_6 = slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_20: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_24: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_20);  cat_4 = None
        add_18: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_47: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_22, [1000, 4096])
        permute_25: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_15: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_47, permute_25);  view_47 = permute_25 = None
        view_48: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [1, 1000, 1024]);  mm_15 = None
        view_49: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_48, [1, 1000, -1, 128]);  view_48 = None
        permute_26: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_25: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_26, unsqueeze_19);  unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_12: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_26, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_7: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_12);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_11: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_26, 3, 0, 64);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_5: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_7, slice_11], -1);  neg_7 = slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_26: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_20);  cat_5 = unsqueeze_20 = None
        add_19: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_21: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_19, 2)
        expand_9: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_21, [1, 8, 4, 1000, 128]);  unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_12: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_53: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [1, 32, 1000, 128]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_50: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_22, [1000, 4096]);  mul_22 = None
        permute_27: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_16: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_50, permute_27);  view_50 = permute_27 = None
        view_51: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [1, 1000, 1024]);  mm_16 = None
        view_52: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_51, [1, 1000, -1, 128]);  view_51 = None
        permute_28: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_22: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_28, 2)
        expand_10: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_22, [1, 8, 4, 1000, 128]);  unsqueeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_13: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_54: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [1, 32, 1000, 128]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_5: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_5, full_default_4);  full_default_5 = full_default_4 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_18, view_53, view_54, where_2, False, scale = 0.08838834764831845);  add_18 = view_53 = view_54 = where_2 = None
        getitem_18: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_55: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_29, [1, 1000, -1]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_56: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_55, [1000, 4096]);  view_55 = None
        permute_30: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_17: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_56, permute_30);  view_56 = permute_30 = None
        view_57: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [1, 1000, 4096]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_20: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_16, view_57);  add_16 = view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_54: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_6: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_54, 2)
        mean_5: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_21: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_5, 1e-05);  mean_5 = None
        rsqrt_5: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_27: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_54, rsqrt_5);  convert_element_type_54 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_55: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        mul_28: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg26_1, convert_element_type_55);  arg26_1 = convert_element_type_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_58: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_28, [1000, 4096])
        permute_31: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_18: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_58, permute_31);  view_58 = permute_31 = None
        view_59: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [1, 1000, 14336]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_58: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        neg_8: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_58)
        exp_2: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_22: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_58, add_22);  convert_element_type_58 = add_22 = None
        convert_element_type_59: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_60: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_28, [1000, 4096]);  mul_28 = None
        permute_32: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_19: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_60, permute_32);  view_60 = permute_32 = None
        view_61: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [1, 1000, 14336]);  mm_19 = None
        mul_29: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_59, view_61);  convert_element_type_59 = view_61 = None
        view_62: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_29, [1000, 14336]);  mul_29 = None
        permute_33: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_20: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_62, permute_33);  view_62 = permute_33 = None
        view_63: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [1, 1000, 4096]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_23: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_20, view_63);  add_20 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_64: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_7: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_64, 2)
        mean_6: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_24: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_6, 1e-05);  mean_6 = None
        rsqrt_6: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_30: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_64, rsqrt_6);  convert_element_type_64 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_65: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        mul_31: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg30_1, convert_element_type_65);  arg30_1 = convert_element_type_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_64: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_31, [1000, 4096])
        permute_34: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_21: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_64, permute_34);  view_64 = permute_34 = None
        view_65: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [1, 1000, 4096]);  mm_21 = None
        view_66: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_65, [1, 1000, -1, 128]);  view_65 = None
        permute_35: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_23: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_32: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_35, unsqueeze_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_14: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_35, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_9: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_14);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_13: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_35, 3, 0, 64);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_6: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_9, slice_13], -1);  neg_9 = slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_24: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_33: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_24);  cat_6 = None
        add_25: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_67: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_31, [1000, 4096])
        permute_36: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        mm_22: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_67, permute_36);  view_67 = permute_36 = None
        view_68: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [1, 1000, 1024]);  mm_22 = None
        view_69: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_68, [1, 1000, -1, 128]);  view_68 = None
        permute_37: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_34: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_37, unsqueeze_23);  unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_16: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_37, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_10: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_16);  slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_15: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_37, 3, 0, 64);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_7: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_10, slice_15], -1);  neg_10 = slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_35: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_24);  cat_7 = unsqueeze_24 = None
        add_26: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_25: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_26, 2)
        expand_11: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_25, [1, 8, 4, 1000, 128]);  unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_16: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_73: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [1, 32, 1000, 128]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_70: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_31, [1000, 4096]);  mul_31 = None
        permute_38: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_23: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_70, permute_38);  view_70 = permute_38 = None
        view_71: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [1, 1000, 1024]);  mm_23 = None
        view_72: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_71, [1, 1000, -1, 128]);  view_71 = None
        permute_39: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_26: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_39, 2)
        expand_12: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_26, [1, 8, 4, 1000, 128]);  unsqueeze_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_17: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_74: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [1, 32, 1000, 128]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_7: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_25, view_73, view_74, where_3, False, scale = 0.08838834764831845);  add_25 = view_73 = view_74 = where_3 = None
        getitem_27: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_75: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_40, [1, 1000, -1]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_76: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_75, [1000, 4096]);  view_75 = None
        permute_41: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_24: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_76, permute_41);  view_76 = permute_41 = None
        view_77: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [1, 1000, 4096]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_27: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_23, view_77);  add_23 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_74: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_8: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_74, 2)
        mean_7: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_28: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_7, 1e-05);  mean_7 = None
        rsqrt_7: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_36: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_74, rsqrt_7);  convert_element_type_74 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_75: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None
        mul_37: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg35_1, convert_element_type_75);  arg35_1 = convert_element_type_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_78: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_37, [1000, 4096])
        permute_42: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_25: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_78, permute_42);  view_78 = permute_42 = None
        view_79: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [1, 1000, 14336]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_78: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        neg_11: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_78)
        exp_3: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_29: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_78, add_29);  convert_element_type_78 = add_29 = None
        convert_element_type_79: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_80: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_37, [1000, 4096]);  mul_37 = None
        permute_43: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_26: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_80, permute_43);  view_80 = permute_43 = None
        view_81: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [1, 1000, 14336]);  mm_26 = None
        mul_38: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_79, view_81);  convert_element_type_79 = view_81 = None
        view_82: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_38, [1000, 14336]);  mul_38 = None
        permute_44: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_27: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_82, permute_44);  view_82 = permute_44 = None
        view_83: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [1, 1000, 4096]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_30: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_27, view_83);  add_27 = view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_84: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_9: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_84, 2)
        mean_8: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_31: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_8, 1e-05);  mean_8 = None
        rsqrt_8: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_39: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_84, rsqrt_8);  convert_element_type_84 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_85: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        mul_40: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg39_1, convert_element_type_85);  arg39_1 = convert_element_type_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_84: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_40, [1000, 4096])
        permute_45: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_28: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_84, permute_45);  view_84 = permute_45 = None
        view_85: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [1, 1000, 4096]);  mm_28 = None
        view_86: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_85, [1, 1000, -1, 128]);  view_85 = None
        permute_46: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_27: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_41: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_46, unsqueeze_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_18: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_46, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_12: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_18);  slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_17: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_46, 3, 0, 64);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_8: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_12, slice_17], -1);  neg_12 = slice_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_28: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_42: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_28);  cat_8 = None
        add_32: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_87: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_40, [1000, 4096])
        permute_47: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_29: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_87, permute_47);  view_87 = permute_47 = None
        view_88: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [1, 1000, 1024]);  mm_29 = None
        view_89: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_88, [1, 1000, -1, 128]);  view_88 = None
        permute_48: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_43: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_48, unsqueeze_27);  unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_20: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_48, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_13: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_20);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_19: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_48, 3, 0, 64);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_9: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_13, slice_19], -1);  neg_13 = slice_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_44: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_28);  cat_9 = unsqueeze_28 = None
        add_33: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_29: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_33, 2)
        expand_13: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_29, [1, 8, 4, 1000, 128]);  unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_20: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_93: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [1, 32, 1000, 128]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_90: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_40, [1000, 4096]);  mul_40 = None
        permute_49: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_30: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_90, permute_49);  view_90 = permute_49 = None
        view_91: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [1, 1000, 1024]);  mm_30 = None
        view_92: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_91, [1, 1000, -1, 128]);  view_91 = None
        permute_50: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_30: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_50, 2)
        expand_14: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_30, [1, 8, 4, 1000, 128]);  unsqueeze_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_21: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_94: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [1, 32, 1000, 128]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_9, full_default_8);  full_default_9 = full_default_8 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_32, view_93, view_94, where_4, False, scale = 0.08838834764831845);  add_32 = view_93 = view_94 = where_4 = None
        getitem_36: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_95: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_51, [1, 1000, -1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_96: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_95, [1000, 4096]);  view_95 = None
        permute_52: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        mm_31: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_96, permute_52);  view_96 = permute_52 = None
        view_97: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [1, 1000, 4096]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_34: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_30, view_97);  add_30 = view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_94: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_10: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_94, 2)
        mean_9: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_9, 1e-05);  mean_9 = None
        rsqrt_9: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_45: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_94, rsqrt_9);  convert_element_type_94 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_95: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        mul_46: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg44_1, convert_element_type_95);  arg44_1 = convert_element_type_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_98: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_46, [1000, 4096])
        permute_53: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_32: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_98, permute_53);  view_98 = permute_53 = None
        view_99: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [1, 1000, 14336]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_98: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        neg_14: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_98)
        exp_4: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_36: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_98, add_36);  convert_element_type_98 = add_36 = None
        convert_element_type_99: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_100: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_46, [1000, 4096]);  mul_46 = None
        permute_54: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_33: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_100, permute_54);  view_100 = permute_54 = None
        view_101: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [1, 1000, 14336]);  mm_33 = None
        mul_47: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_99, view_101);  convert_element_type_99 = view_101 = None
        view_102: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_47, [1000, 14336]);  mul_47 = None
        permute_55: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_34: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_102, permute_55);  view_102 = permute_55 = None
        view_103: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [1, 1000, 4096]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_37: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_34, view_103);  add_34 = view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_104: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_11: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_104, 2)
        mean_10: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_38: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_10, 1e-05);  mean_10 = None
        rsqrt_10: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_48: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_104, rsqrt_10);  convert_element_type_104 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_105: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        mul_49: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg48_1, convert_element_type_105);  arg48_1 = convert_element_type_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_104: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_49, [1000, 4096])
        permute_56: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_35: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_104, permute_56);  view_104 = permute_56 = None
        view_105: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [1, 1000, 4096]);  mm_35 = None
        view_106: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_105, [1, 1000, -1, 128]);  view_105 = None
        permute_57: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_31: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_50: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_57, unsqueeze_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_22: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_57, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_15: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_22);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_21: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_57, 3, 0, 64);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_10: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_15, slice_21], -1);  neg_15 = slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_32: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_51: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_32);  cat_10 = None
        add_39: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_107: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_49, [1000, 4096])
        permute_58: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_36: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_107, permute_58);  view_107 = permute_58 = None
        view_108: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [1, 1000, 1024]);  mm_36 = None
        view_109: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_108, [1, 1000, -1, 128]);  view_108 = None
        permute_59: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_52: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_59, unsqueeze_31);  unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_24: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_59, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_16: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_24);  slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_23: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_59, 3, 0, 64);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_11: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_16, slice_23], -1);  neg_16 = slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_53: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_11, unsqueeze_32);  cat_11 = unsqueeze_32 = None
        add_40: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_33: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_40, 2)
        expand_15: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_33, [1, 8, 4, 1000, 128]);  unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_24: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_113: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [1, 32, 1000, 128]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_110: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_49, [1000, 4096]);  mul_49 = None
        permute_60: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_37: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_110, permute_60);  view_110 = permute_60 = None
        view_111: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [1, 1000, 1024]);  mm_37 = None
        view_112: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [1, 1000, -1, 128]);  view_111 = None
        permute_61: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_34: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_61, 2)
        expand_16: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_34, [1, 8, 4, 1000, 128]);  unsqueeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_25: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_114: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [1, 32, 1000, 128]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_11: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_11, full_default_10);  full_default_11 = full_default_10 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_39, view_113, view_114, where_5, False, scale = 0.08838834764831845);  add_39 = view_113 = view_114 = where_5 = None
        getitem_45: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_115: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_62, [1, 1000, -1]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_116: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_115, [1000, 4096]);  view_115 = None
        permute_63: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_38: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_116, permute_63);  view_116 = permute_63 = None
        view_117: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [1, 1000, 4096]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_41: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_37, view_117);  add_37 = view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_114: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_12: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_114, 2)
        mean_11: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_11, 1e-05);  mean_11 = None
        rsqrt_11: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_54: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_114, rsqrt_11);  convert_element_type_114 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_115: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        mul_55: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg53_1, convert_element_type_115);  arg53_1 = convert_element_type_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_118: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_55, [1000, 4096])
        permute_64: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_39: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_118, permute_64);  view_118 = permute_64 = None
        view_119: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [1, 1000, 14336]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_118: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        neg_17: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_118)
        exp_5: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_43: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_118, add_43);  convert_element_type_118 = add_43 = None
        convert_element_type_119: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_120: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_55, [1000, 4096]);  mul_55 = None
        permute_65: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_40: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_120, permute_65);  view_120 = permute_65 = None
        view_121: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [1, 1000, 14336]);  mm_40 = None
        mul_56: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, view_121);  convert_element_type_119 = view_121 = None
        view_122: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_56, [1000, 14336]);  mul_56 = None
        permute_66: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_41: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_122, permute_66);  view_122 = permute_66 = None
        view_123: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [1, 1000, 4096]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_44: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_41, view_123);  add_41 = view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_124: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_13: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_124, 2)
        mean_12: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_45: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_12, 1e-05);  mean_12 = None
        rsqrt_12: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_57: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_124, rsqrt_12);  convert_element_type_124 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_125: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        mul_58: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg57_1, convert_element_type_125);  arg57_1 = convert_element_type_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_124: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_58, [1000, 4096])
        permute_67: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_42: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_124, permute_67);  view_124 = permute_67 = None
        view_125: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [1, 1000, 4096]);  mm_42 = None
        view_126: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [1, 1000, -1, 128]);  view_125 = None
        permute_68: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_35: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_59: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_68, unsqueeze_35)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_26: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_68, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_18: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_26);  slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_25: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_68, 3, 0, 64);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_12: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_18, slice_25], -1);  neg_18 = slice_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_36: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_60: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_12, unsqueeze_36);  cat_12 = None
        add_46: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_127: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_58, [1000, 4096])
        permute_69: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_43: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_127, permute_69);  view_127 = permute_69 = None
        view_128: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_43, [1, 1000, 1024]);  mm_43 = None
        view_129: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_128, [1, 1000, -1, 128]);  view_128 = None
        permute_70: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_61: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_70, unsqueeze_35);  unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_28: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_70, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_19: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_28);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_27: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_70, 3, 0, 64);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_13: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_19, slice_27], -1);  neg_19 = slice_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_62: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_36);  cat_13 = unsqueeze_36 = None
        add_47: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_37: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_47, 2)
        expand_17: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_37, [1, 8, 4, 1000, 128]);  unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_28: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_133: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [1, 32, 1000, 128]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_130: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_58, [1000, 4096]);  mul_58 = None
        permute_71: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_44: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_130, permute_71);  view_130 = permute_71 = None
        view_131: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [1, 1000, 1024]);  mm_44 = None
        view_132: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_131, [1, 1000, -1, 128]);  view_131 = None
        permute_72: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_38: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_72, 2)
        expand_18: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_38, [1, 8, 4, 1000, 128]);  unsqueeze_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_29: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_134: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [1, 32, 1000, 128]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_13: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_46, view_133, view_134, where_6, False, scale = 0.08838834764831845);  add_46 = view_133 = view_134 = where_6 = None
        getitem_54: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_135: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_73, [1, 1000, -1]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_136: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_135, [1000, 4096]);  view_135 = None
        permute_74: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_45: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_136, permute_74);  view_136 = permute_74 = None
        view_137: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_45, [1, 1000, 4096]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_48: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_44, view_137);  add_44 = view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_134: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_14: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_134, 2)
        mean_13: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_13, 1e-05);  mean_13 = None
        rsqrt_13: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_63: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_134, rsqrt_13);  convert_element_type_134 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_135: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None
        mul_64: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg62_1, convert_element_type_135);  arg62_1 = convert_element_type_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_138: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_64, [1000, 4096])
        permute_75: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_46: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_138, permute_75);  view_138 = permute_75 = None
        view_139: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [1, 1000, 14336]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_138: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None
        neg_20: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_138)
        exp_6: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_50: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_138, add_50);  convert_element_type_138 = add_50 = None
        convert_element_type_139: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_140: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_64, [1000, 4096]);  mul_64 = None
        permute_76: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_47: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_140, permute_76);  view_140 = permute_76 = None
        view_141: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [1, 1000, 14336]);  mm_47 = None
        mul_65: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, view_141);  convert_element_type_139 = view_141 = None
        view_142: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_65, [1000, 14336]);  mul_65 = None
        permute_77: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_48: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_142, permute_77);  view_142 = permute_77 = None
        view_143: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [1, 1000, 4096]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_51: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_48, view_143);  add_48 = view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_144: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_15: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_144, 2)
        mean_14: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_52: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_14, 1e-05);  mean_14 = None
        rsqrt_14: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_66: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_144, rsqrt_14);  convert_element_type_144 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_145: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_66, torch.bfloat16);  mul_66 = None
        mul_67: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg66_1, convert_element_type_145);  arg66_1 = convert_element_type_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_144: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_67, [1000, 4096])
        permute_78: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_49: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_144, permute_78);  view_144 = permute_78 = None
        view_145: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_49, [1, 1000, 4096]);  mm_49 = None
        view_146: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_145, [1, 1000, -1, 128]);  view_145 = None
        permute_79: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_39: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_68: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_79, unsqueeze_39)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_30: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_79, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_21: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_30);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_29: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_79, 3, 0, 64);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_14: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_21, slice_29], -1);  neg_21 = slice_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_40: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_69: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_14, unsqueeze_40);  cat_14 = None
        add_53: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_147: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_67, [1000, 4096])
        permute_80: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_50: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_147, permute_80);  view_147 = permute_80 = None
        view_148: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [1, 1000, 1024]);  mm_50 = None
        view_149: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_148, [1, 1000, -1, 128]);  view_148 = None
        permute_81: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_70: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_81, unsqueeze_39);  unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_32: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_81, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_22: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_32);  slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_31: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_81, 3, 0, 64);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_15: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_22, slice_31], -1);  neg_22 = slice_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_71: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_15, unsqueeze_40);  cat_15 = unsqueeze_40 = None
        add_54: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_70, mul_71);  mul_70 = mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_41: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_54, 2)
        expand_19: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_41, [1, 8, 4, 1000, 128]);  unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_32: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_153: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [1, 32, 1000, 128]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_150: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_67, [1000, 4096]);  mul_67 = None
        permute_82: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_51: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_150, permute_82);  view_150 = permute_82 = None
        view_151: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_51, [1, 1000, 1024]);  mm_51 = None
        view_152: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_151, [1, 1000, -1, 128]);  view_151 = None
        permute_83: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_42: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_83, 2)
        expand_20: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_42, [1, 8, 4, 1000, 128]);  unsqueeze_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_33: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_154: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [1, 32, 1000, 128]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_15: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_15, full_default_14);  full_default_15 = full_default_14 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_53, view_153, view_154, where_7, False, scale = 0.08838834764831845);  add_53 = view_153 = view_154 = where_7 = None
        getitem_63: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_155: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_84, [1, 1000, -1]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_156: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [1000, 4096]);  view_155 = None
        permute_85: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_52: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_156, permute_85);  view_156 = permute_85 = None
        view_157: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [1, 1000, 4096]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_55: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_51, view_157);  add_51 = view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_154: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_16: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_154, 2)
        mean_15: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_56: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_15, 1e-05);  mean_15 = None
        rsqrt_15: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_72: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_154, rsqrt_15);  convert_element_type_154 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_155: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        mul_73: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg71_1, convert_element_type_155);  arg71_1 = convert_element_type_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_158: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_73, [1000, 4096])
        permute_86: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_53: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_158, permute_86);  view_158 = permute_86 = None
        view_159: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_53, [1, 1000, 14336]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_158: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        neg_23: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_158)
        exp_7: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_57: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_158, add_57);  convert_element_type_158 = add_57 = None
        convert_element_type_159: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_160: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_73, [1000, 4096]);  mul_73 = None
        permute_87: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_54: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_160, permute_87);  view_160 = permute_87 = None
        view_161: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [1, 1000, 14336]);  mm_54 = None
        mul_74: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_159, view_161);  convert_element_type_159 = view_161 = None
        view_162: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_74, [1000, 14336]);  mul_74 = None
        permute_88: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_55: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_162, permute_88);  view_162 = permute_88 = None
        view_163: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [1, 1000, 4096]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_58: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_55, view_163);  add_55 = view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_164: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_58, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_17: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_164, 2)
        mean_16: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_59: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_16, 1e-05);  mean_16 = None
        rsqrt_16: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_75: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, rsqrt_16);  convert_element_type_164 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_165: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_75, torch.bfloat16);  mul_75 = None
        mul_76: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg75_1, convert_element_type_165);  arg75_1 = convert_element_type_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_164: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_76, [1000, 4096])
        permute_89: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_56: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_164, permute_89);  view_164 = permute_89 = None
        view_165: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [1, 1000, 4096]);  mm_56 = None
        view_166: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_165, [1, 1000, -1, 128]);  view_165 = None
        permute_90: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_43: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_77: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_90, unsqueeze_43)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_34: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_90, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_24: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_34);  slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_33: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_90, 3, 0, 64);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_16: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_24, slice_33], -1);  neg_24 = slice_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_44: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_78: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_16, unsqueeze_44);  cat_16 = None
        add_60: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_167: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_76, [1000, 4096])
        permute_91: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        mm_57: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_167, permute_91);  view_167 = permute_91 = None
        view_168: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [1, 1000, 1024]);  mm_57 = None
        view_169: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_168, [1, 1000, -1, 128]);  view_168 = None
        permute_92: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_79: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_92, unsqueeze_43);  unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_36: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_92, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_25: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_36);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_35: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_92, 3, 0, 64);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_17: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_25, slice_35], -1);  neg_25 = slice_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_80: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_17, unsqueeze_44);  cat_17 = unsqueeze_44 = None
        add_61: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, mul_80);  mul_79 = mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_45: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_61, 2)
        expand_21: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_45, [1, 8, 4, 1000, 128]);  unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_36: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_173: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [1, 32, 1000, 128]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_170: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_76, [1000, 4096]);  mul_76 = None
        permute_93: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_58: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_170, permute_93);  view_170 = permute_93 = None
        view_171: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [1, 1000, 1024]);  mm_58 = None
        view_172: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_171, [1, 1000, -1, 128]);  view_171 = None
        permute_94: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_46: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_94, 2)
        expand_22: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_46, [1, 8, 4, 1000, 128]);  unsqueeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_37: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_174: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [1, 32, 1000, 128]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_17: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_17, full_default_16);  full_default_17 = full_default_16 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_60, view_173, view_174, where_8, False, scale = 0.08838834764831845);  add_60 = view_173 = view_174 = where_8 = None
        getitem_72: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_175: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_95, [1, 1000, -1]);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_176: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_175, [1000, 4096]);  view_175 = None
        permute_96: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_59: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_96);  view_176 = permute_96 = None
        view_177: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_59, [1, 1000, 4096]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_62: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_58, view_177);  add_58 = view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_174: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_18: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_174, 2)
        mean_17: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_17, 1e-05);  mean_17 = None
        rsqrt_17: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_81: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_174, rsqrt_17);  convert_element_type_174 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_175: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None
        mul_82: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg80_1, convert_element_type_175);  arg80_1 = convert_element_type_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_178: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_82, [1000, 4096])
        permute_97: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_60: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_178, permute_97);  view_178 = permute_97 = None
        view_179: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [1, 1000, 14336]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_178: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_179, torch.float32);  view_179 = None
        neg_26: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_178)
        exp_8: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_64: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_178, add_64);  convert_element_type_178 = add_64 = None
        convert_element_type_179: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_180: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_82, [1000, 4096]);  mul_82 = None
        permute_98: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_61: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_180, permute_98);  view_180 = permute_98 = None
        view_181: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_61, [1, 1000, 14336]);  mm_61 = None
        mul_83: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_179, view_181);  convert_element_type_179 = view_181 = None
        view_182: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_83, [1000, 14336]);  mul_83 = None
        permute_99: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_62: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_182, permute_99);  view_182 = permute_99 = None
        view_183: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [1, 1000, 4096]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_65: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_62, view_183);  add_62 = view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_184: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_19: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_184, 2)
        mean_18: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_66: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_18, 1e-05);  mean_18 = None
        rsqrt_18: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_84: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_184, rsqrt_18);  convert_element_type_184 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_185: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None
        mul_85: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg84_1, convert_element_type_185);  arg84_1 = convert_element_type_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_184: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_85, [1000, 4096])
        permute_100: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_63: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_184, permute_100);  view_184 = permute_100 = None
        view_185: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_63, [1, 1000, 4096]);  mm_63 = None
        view_186: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_185, [1, 1000, -1, 128]);  view_185 = None
        permute_101: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_47: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_86: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_101, unsqueeze_47)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_38: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_101, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_27: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_38);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_37: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_101, 3, 0, 64);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_18: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_27, slice_37], -1);  neg_27 = slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_48: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_87: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_18, unsqueeze_48);  cat_18 = None
        add_67: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_86, mul_87);  mul_86 = mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_187: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_85, [1000, 4096])
        permute_102: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_64: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_187, permute_102);  view_187 = permute_102 = None
        view_188: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [1, 1000, 1024]);  mm_64 = None
        view_189: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_188, [1, 1000, -1, 128]);  view_188 = None
        permute_103: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_88: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_103, unsqueeze_47);  unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_40: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_103, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_28: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_40);  slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_39: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_103, 3, 0, 64);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_19: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_28, slice_39], -1);  neg_28 = slice_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_89: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_19, unsqueeze_48);  cat_19 = unsqueeze_48 = None
        add_68: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_49: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_68, 2)
        expand_23: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_49, [1, 8, 4, 1000, 128]);  unsqueeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_40: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_193: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [1, 32, 1000, 128]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_190: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_85, [1000, 4096]);  mul_85 = None
        permute_104: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_65: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_190, permute_104);  view_190 = permute_104 = None
        view_191: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_65, [1, 1000, 1024]);  mm_65 = None
        view_192: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [1, 1000, -1, 128]);  view_191 = None
        permute_105: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_50: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_105, 2)
        expand_24: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_50, [1, 8, 4, 1000, 128]);  unsqueeze_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_41: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_194: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [1, 32, 1000, 128]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_19: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_67, view_193, view_194, where_9, False, scale = 0.08838834764831845);  add_67 = view_193 = view_194 = where_9 = None
        getitem_81: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3]);  getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_195: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_106, [1, 1000, -1]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_196: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_195, [1000, 4096]);  view_195 = None
        permute_107: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        mm_66: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_196, permute_107);  view_196 = permute_107 = None
        view_197: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [1, 1000, 4096]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_69: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_65, view_197);  add_65 = view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_194: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_20: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_194, 2)
        mean_19: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_70: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_19, 1e-05);  mean_19 = None
        rsqrt_19: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_90: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_194, rsqrt_19);  convert_element_type_194 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_195: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None
        mul_91: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg89_1, convert_element_type_195);  arg89_1 = convert_element_type_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_198: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_91, [1000, 4096])
        permute_108: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_67: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_198, permute_108);  view_198 = permute_108 = None
        view_199: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_67, [1, 1000, 14336]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_198: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_199, torch.float32);  view_199 = None
        neg_29: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_198)
        exp_9: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_71: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_198, add_71);  convert_element_type_198 = add_71 = None
        convert_element_type_199: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_200: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_91, [1000, 4096]);  mul_91 = None
        permute_109: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        mm_68: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_200, permute_109);  view_200 = permute_109 = None
        view_201: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [1, 1000, 14336]);  mm_68 = None
        mul_92: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_199, view_201);  convert_element_type_199 = view_201 = None
        view_202: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_92, [1000, 14336]);  mul_92 = None
        permute_110: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        mm_69: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_202, permute_110);  view_202 = permute_110 = None
        view_203: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_69, [1, 1000, 4096]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_72: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_69, view_203);  add_69 = view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_204: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_21: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_204, 2)
        mean_20: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_73: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_20, 1e-05);  mean_20 = None
        rsqrt_20: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_93: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_204, rsqrt_20);  convert_element_type_204 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_205: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        mul_94: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg93_1, convert_element_type_205);  arg93_1 = convert_element_type_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_204: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_94, [1000, 4096])
        permute_111: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_70: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_204, permute_111);  view_204 = permute_111 = None
        view_205: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [1, 1000, 4096]);  mm_70 = None
        view_206: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_205, [1, 1000, -1, 128]);  view_205 = None
        permute_112: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_51: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_95: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_112, unsqueeze_51)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_42: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_112, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_30: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_42);  slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_41: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_112, 3, 0, 64);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_20: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_30, slice_41], -1);  neg_30 = slice_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_52: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_96: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_20, unsqueeze_52);  cat_20 = None
        add_74: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_207: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_94, [1000, 4096])
        permute_113: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_71: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_207, permute_113);  view_207 = permute_113 = None
        view_208: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [1, 1000, 1024]);  mm_71 = None
        view_209: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_208, [1, 1000, -1, 128]);  view_208 = None
        permute_114: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_97: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_114, unsqueeze_51);  unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_44: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_114, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_31: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_44);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_43: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_114, 3, 0, 64);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_21: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_31, slice_43], -1);  neg_31 = slice_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_98: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_21, unsqueeze_52);  cat_21 = unsqueeze_52 = None
        add_75: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_53: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_75, 2)
        expand_25: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_53, [1, 8, 4, 1000, 128]);  unsqueeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_44: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_213: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [1, 32, 1000, 128]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_210: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_94, [1000, 4096]);  mul_94 = None
        permute_115: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_72: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_210, permute_115);  view_210 = permute_115 = None
        view_211: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [1, 1000, 1024]);  mm_72 = None
        view_212: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_211, [1, 1000, -1, 128]);  view_211 = None
        permute_116: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_54: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_116, 2)
        expand_26: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_54, [1, 8, 4, 1000, 128]);  unsqueeze_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_45: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_214: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [1, 32, 1000, 128]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_21: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_21, full_default_20);  full_default_21 = full_default_20 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_74, view_213, view_214, where_10, False, scale = 0.08838834764831845);  add_74 = view_213 = view_214 = where_10 = None
        getitem_90: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_215: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_117, [1, 1000, -1]);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_216: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_215, [1000, 4096]);  view_215 = None
        permute_118: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        mm_73: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_216, permute_118);  view_216 = permute_118 = None
        view_217: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_73, [1, 1000, 4096]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_76: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_72, view_217);  add_72 = view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_214: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_22: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_214, 2)
        mean_21: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_77: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_21, 1e-05);  mean_21 = None
        rsqrt_21: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_99: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_214, rsqrt_21);  convert_element_type_214 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_215: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None
        mul_100: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg98_1, convert_element_type_215);  arg98_1 = convert_element_type_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_218: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_100, [1000, 4096])
        permute_119: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_74: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_218, permute_119);  view_218 = permute_119 = None
        view_219: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [1, 1000, 14336]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_218: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_219, torch.float32);  view_219 = None
        neg_32: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_218)
        exp_10: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_78: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_218, add_78);  convert_element_type_218 = add_78 = None
        convert_element_type_219: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_220: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_100, [1000, 4096]);  mul_100 = None
        permute_120: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_75: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_220, permute_120);  view_220 = permute_120 = None
        view_221: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_75, [1, 1000, 14336]);  mm_75 = None
        mul_101: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_219, view_221);  convert_element_type_219 = view_221 = None
        view_222: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_101, [1000, 14336]);  mul_101 = None
        permute_121: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_76: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_222, permute_121);  view_222 = permute_121 = None
        view_223: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [1, 1000, 4096]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_79: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_76, view_223);  add_76 = view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_224: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_23: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_224, 2)
        mean_22: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_80: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_22, 1e-05);  mean_22 = None
        rsqrt_22: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_102: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_224, rsqrt_22);  convert_element_type_224 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_225: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        mul_103: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg102_1, convert_element_type_225);  arg102_1 = convert_element_type_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_224: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_103, [1000, 4096])
        permute_122: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_77: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_224, permute_122);  view_224 = permute_122 = None
        view_225: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [1, 1000, 4096]);  mm_77 = None
        view_226: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_225, [1, 1000, -1, 128]);  view_225 = None
        permute_123: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_55: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_104: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_123, unsqueeze_55)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_46: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_123, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_33: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_46);  slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_45: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_123, 3, 0, 64);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_22: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_33, slice_45], -1);  neg_33 = slice_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_56: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_105: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_22, unsqueeze_56);  cat_22 = None
        add_81: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_227: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_103, [1000, 4096])
        permute_124: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_78: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_227, permute_124);  view_227 = permute_124 = None
        view_228: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [1, 1000, 1024]);  mm_78 = None
        view_229: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_228, [1, 1000, -1, 128]);  view_228 = None
        permute_125: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_106: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_125, unsqueeze_55);  unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_48: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_125, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_34: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_48);  slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_47: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_125, 3, 0, 64);  permute_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_23: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_34, slice_47], -1);  neg_34 = slice_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_107: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_23, unsqueeze_56);  cat_23 = unsqueeze_56 = None
        add_82: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_57: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_82, 2)
        expand_27: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_57, [1, 8, 4, 1000, 128]);  unsqueeze_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_48: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_233: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [1, 32, 1000, 128]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_230: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_103, [1000, 4096]);  mul_103 = None
        permute_126: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_79: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_230, permute_126);  view_230 = permute_126 = None
        view_231: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [1, 1000, 1024]);  mm_79 = None
        view_232: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_231, [1, 1000, -1, 128]);  view_231 = None
        permute_127: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_58: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_127, 2)
        expand_28: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_58, [1, 8, 4, 1000, 128]);  unsqueeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_49: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_234: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [1, 32, 1000, 128]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_23: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_23, full_default_22);  full_default_23 = full_default_22 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_81, view_233, view_234, where_11, False, scale = 0.08838834764831845);  add_81 = view_233 = view_234 = where_11 = None
        getitem_99: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_128, [1, 1000, -1]);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_236: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [1000, 4096]);  view_235 = None
        permute_129: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        mm_80: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_236, permute_129);  view_236 = permute_129 = None
        view_237: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [1, 1000, 4096]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_83: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_79, view_237);  add_79 = view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_234: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_24: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_234, 2)
        mean_23: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_84: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_23, 1e-05);  mean_23 = None
        rsqrt_23: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_108: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_234, rsqrt_23);  convert_element_type_234 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_235: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_108, torch.bfloat16);  mul_108 = None
        mul_109: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg107_1, convert_element_type_235);  arg107_1 = convert_element_type_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_238: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_109, [1000, 4096])
        permute_130: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_81: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_238, permute_130);  view_238 = permute_130 = None
        view_239: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_81, [1, 1000, 14336]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_238: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        neg_35: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_238)
        exp_11: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_35);  neg_35 = None
        add_85: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_238, add_85);  convert_element_type_238 = add_85 = None
        convert_element_type_239: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_240: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_109, [1000, 4096]);  mul_109 = None
        permute_131: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_82: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_240, permute_131);  view_240 = permute_131 = None
        view_241: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [1, 1000, 14336]);  mm_82 = None
        mul_110: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_239, view_241);  convert_element_type_239 = view_241 = None
        view_242: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_110, [1000, 14336]);  mul_110 = None
        permute_132: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        mm_83: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_132);  view_242 = permute_132 = None
        view_243: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_83, [1, 1000, 4096]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_86: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_83, view_243);  add_83 = view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_244: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_86, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_25: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_244, 2)
        mean_24: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_87: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_24, 1e-05);  mean_24 = None
        rsqrt_24: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_111: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_244, rsqrt_24);  convert_element_type_244 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_245: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_111, torch.bfloat16);  mul_111 = None
        mul_112: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg111_1, convert_element_type_245);  arg111_1 = convert_element_type_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_244: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_112, [1000, 4096])
        permute_133: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_84: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_244, permute_133);  view_244 = permute_133 = None
        view_245: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [1, 1000, 4096]);  mm_84 = None
        view_246: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_245, [1, 1000, -1, 128]);  view_245 = None
        permute_134: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_59: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_113: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_134, unsqueeze_59)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_50: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_134, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_36: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_50);  slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_49: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_134, 3, 0, 64);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_24: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_36, slice_49], -1);  neg_36 = slice_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_60: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_114: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_24, unsqueeze_60);  cat_24 = None
        add_88: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_247: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_112, [1000, 4096])
        permute_135: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_85: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_247, permute_135);  view_247 = permute_135 = None
        view_248: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_85, [1, 1000, 1024]);  mm_85 = None
        view_249: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_248, [1, 1000, -1, 128]);  view_248 = None
        permute_136: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_115: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_136, unsqueeze_59);  unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_52: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_136, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_37: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_52);  slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_51: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_136, 3, 0, 64);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_25: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_37, slice_51], -1);  neg_37 = slice_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_116: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_25, unsqueeze_60);  cat_25 = unsqueeze_60 = None
        add_89: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_61: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_89, 2)
        expand_29: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_61, [1, 8, 4, 1000, 128]);  unsqueeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_52: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_253: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [1, 32, 1000, 128]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_250: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_112, [1000, 4096]);  mul_112 = None
        permute_137: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_86: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_250, permute_137);  view_250 = permute_137 = None
        view_251: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [1, 1000, 1024]);  mm_86 = None
        view_252: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [1, 1000, -1, 128]);  view_251 = None
        permute_138: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_62: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_138, 2)
        expand_30: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_62, [1, 8, 4, 1000, 128]);  unsqueeze_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_53: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_254: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [1, 32, 1000, 128]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_25: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_25, full_default_24);  full_default_25 = full_default_24 = None
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_88, view_253, view_254, where_12, False, scale = 0.08838834764831845);  add_88 = view_253 = view_254 = where_12 = None
        getitem_108: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_12[0];  _scaled_dot_product_cudnn_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_139: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_255: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_139, [1, 1000, -1]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_256: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_255, [1000, 4096]);  view_255 = None
        permute_140: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_87: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_256, permute_140);  view_256 = permute_140 = None
        view_257: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [1, 1000, 4096]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_90: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_86, view_257);  add_86 = view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_254: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_26: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_254, 2)
        mean_25: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_91: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_25, 1e-05);  mean_25 = None
        rsqrt_25: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        mul_117: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_254, rsqrt_25);  convert_element_type_254 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_255: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        mul_118: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg116_1, convert_element_type_255);  arg116_1 = convert_element_type_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_258: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_118, [1000, 4096])
        permute_141: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_88: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_258, permute_141);  view_258 = permute_141 = None
        view_259: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [1, 1000, 14336]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_258: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_259, torch.float32);  view_259 = None
        neg_38: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_258)
        exp_12: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_38);  neg_38 = None
        add_92: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_258, add_92);  convert_element_type_258 = add_92 = None
        convert_element_type_259: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_260: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_118, [1000, 4096]);  mul_118 = None
        permute_142: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_89: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_260, permute_142);  view_260 = permute_142 = None
        view_261: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [1, 1000, 14336]);  mm_89 = None
        mul_119: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_259, view_261);  convert_element_type_259 = view_261 = None
        view_262: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_119, [1000, 14336]);  mul_119 = None
        permute_143: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_90: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_262, permute_143);  view_262 = permute_143 = None
        view_263: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [1, 1000, 4096]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_93: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_90, view_263);  add_90 = view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_264: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_27: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_264, 2)
        mean_26: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_94: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_26, 1e-05);  mean_26 = None
        rsqrt_26: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_120: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_264, rsqrt_26);  convert_element_type_264 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_265: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        mul_121: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg120_1, convert_element_type_265);  arg120_1 = convert_element_type_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_264: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_121, [1000, 4096])
        permute_144: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_91: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_264, permute_144);  view_264 = permute_144 = None
        view_265: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [1, 1000, 4096]);  mm_91 = None
        view_266: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_265, [1, 1000, -1, 128]);  view_265 = None
        permute_145: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_63: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_122: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_145, unsqueeze_63)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_54: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_145, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_39: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_54);  slice_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_53: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_145, 3, 0, 64);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_26: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_39, slice_53], -1);  neg_39 = slice_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_64: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_123: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_26, unsqueeze_64);  cat_26 = None
        add_95: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_267: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_121, [1000, 4096])
        permute_146: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_92: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_267, permute_146);  view_267 = permute_146 = None
        view_268: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [1, 1000, 1024]);  mm_92 = None
        view_269: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_268, [1, 1000, -1, 128]);  view_268 = None
        permute_147: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_124: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_147, unsqueeze_63);  unsqueeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_56: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_147, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_40: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_56);  slice_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_55: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_147, 3, 0, 64);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_27: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_40, slice_55], -1);  neg_40 = slice_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_125: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_27, unsqueeze_64);  cat_27 = unsqueeze_64 = None
        add_96: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_65: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_96, 2)
        expand_31: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_65, [1, 8, 4, 1000, 128]);  unsqueeze_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_56: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_273: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_56, [1, 32, 1000, 128]);  clone_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_270: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_121, [1000, 4096]);  mul_121 = None
        permute_148: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_93: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_270, permute_148);  view_270 = permute_148 = None
        view_271: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_93, [1, 1000, 1024]);  mm_93 = None
        view_272: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_271, [1, 1000, -1, 128]);  view_271 = None
        permute_149: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_66: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_149, 2)
        expand_32: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_66, [1, 8, 4, 1000, 128]);  unsqueeze_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_57: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_274: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [1, 32, 1000, 128]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_27: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_27, full_default_26);  full_default_27 = full_default_26 = None
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_95, view_273, view_274, where_13, False, scale = 0.08838834764831845);  add_95 = view_273 = view_274 = where_13 = None
        getitem_117: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_13[0];  _scaled_dot_product_cudnn_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_150: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_275: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_150, [1, 1000, -1]);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_276: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_275, [1000, 4096]);  view_275 = None
        permute_151: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        mm_94: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_276, permute_151);  view_276 = permute_151 = None
        view_277: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [1, 1000, 4096]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_97: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_93, view_277);  add_93 = view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_274: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_28: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_274, 2)
        mean_27: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_98: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_27, 1e-05);  mean_27 = None
        rsqrt_27: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_126: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_274, rsqrt_27);  convert_element_type_274 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_275: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        mul_127: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg125_1, convert_element_type_275);  arg125_1 = convert_element_type_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_278: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_127, [1000, 4096])
        permute_152: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_95: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_278, permute_152);  view_278 = permute_152 = None
        view_279: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_95, [1, 1000, 14336]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_278: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        neg_41: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_278)
        exp_13: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_41);  neg_41 = None
        add_99: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_278, add_99);  convert_element_type_278 = add_99 = None
        convert_element_type_279: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_280: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_127, [1000, 4096]);  mul_127 = None
        permute_153: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_96: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_280, permute_153);  view_280 = permute_153 = None
        view_281: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [1, 1000, 14336]);  mm_96 = None
        mul_128: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_279, view_281);  convert_element_type_279 = view_281 = None
        view_282: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_128, [1000, 14336]);  mul_128 = None
        permute_154: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        mm_97: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_282, permute_154);  view_282 = permute_154 = None
        view_283: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_97, [1, 1000, 4096]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_100: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_97, view_283);  add_97 = view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_284: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_29: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_284, 2)
        mean_28: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_101: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_28, 1e-05);  mean_28 = None
        rsqrt_28: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        mul_129: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, rsqrt_28);  convert_element_type_284 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_285: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        mul_130: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg129_1, convert_element_type_285);  arg129_1 = convert_element_type_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_284: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_130, [1000, 4096])
        permute_155: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        mm_98: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_284, permute_155);  view_284 = permute_155 = None
        view_285: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [1, 1000, 4096]);  mm_98 = None
        view_286: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_285, [1, 1000, -1, 128]);  view_285 = None
        permute_156: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_67: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_131: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_156, unsqueeze_67)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_58: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_156, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_42: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_58);  slice_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_57: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_156, 3, 0, 64);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_28: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_42, slice_57], -1);  neg_42 = slice_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_68: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_132: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_28, unsqueeze_68);  cat_28 = None
        add_102: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_287: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_130, [1000, 4096])
        permute_157: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_99: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_287, permute_157);  view_287 = permute_157 = None
        view_288: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_99, [1, 1000, 1024]);  mm_99 = None
        view_289: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_288, [1, 1000, -1, 128]);  view_288 = None
        permute_158: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_133: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_158, unsqueeze_67);  unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_60: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_158, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_43: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_60);  slice_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_59: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_158, 3, 0, 64);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_29: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_43, slice_59], -1);  neg_43 = slice_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_134: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_29, unsqueeze_68);  cat_29 = unsqueeze_68 = None
        add_103: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_69: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_103, 2)
        expand_33: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_69, [1, 8, 4, 1000, 128]);  unsqueeze_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_60: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_293: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [1, 32, 1000, 128]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_290: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_130, [1000, 4096]);  mul_130 = None
        permute_159: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        mm_100: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_290, permute_159);  view_290 = permute_159 = None
        view_291: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [1, 1000, 1024]);  mm_100 = None
        view_292: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_291, [1, 1000, -1, 128]);  view_291 = None
        permute_160: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_70: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_160, 2)
        expand_34: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_70, [1, 8, 4, 1000, 128]);  unsqueeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_61: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_294: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [1, 32, 1000, 128]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_29: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_29, full_default_28);  full_default_29 = full_default_28 = None
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_102, view_293, view_294, where_14, False, scale = 0.08838834764831845);  add_102 = view_293 = view_294 = where_14 = None
        getitem_126: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_14[0];  _scaled_dot_product_cudnn_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_161: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_295: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_161, [1, 1000, -1]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_296: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_295, [1000, 4096]);  view_295 = None
        permute_162: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_101: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_296, permute_162);  view_296 = permute_162 = None
        view_297: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_101, [1, 1000, 4096]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_104: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_100, view_297);  add_100 = view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_294: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_30: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_294, 2)
        mean_29: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_105: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_29, 1e-05);  mean_29 = None
        rsqrt_29: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        mul_135: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_294, rsqrt_29);  convert_element_type_294 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_295: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_135, torch.bfloat16);  mul_135 = None
        mul_136: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg134_1, convert_element_type_295);  arg134_1 = convert_element_type_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_298: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_136, [1000, 4096])
        permute_163: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_102: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_298, permute_163);  view_298 = permute_163 = None
        view_299: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [1, 1000, 14336]);  mm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_298: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        neg_44: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_298)
        exp_14: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_44);  neg_44 = None
        add_106: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_298, add_106);  convert_element_type_298 = add_106 = None
        convert_element_type_299: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_300: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_136, [1000, 4096]);  mul_136 = None
        permute_164: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        mm_103: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_300, permute_164);  view_300 = permute_164 = None
        view_301: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_103, [1, 1000, 14336]);  mm_103 = None
        mul_137: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_299, view_301);  convert_element_type_299 = view_301 = None
        view_302: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_137, [1000, 14336]);  mul_137 = None
        permute_165: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_104: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_302, permute_165);  view_302 = permute_165 = None
        view_303: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [1, 1000, 4096]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_107: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_104, view_303);  add_104 = view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_304: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_31: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_304, 2)
        mean_30: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_108: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_30, 1e-05);  mean_30 = None
        rsqrt_30: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_138: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_304, rsqrt_30);  convert_element_type_304 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_305: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_138, torch.bfloat16);  mul_138 = None
        mul_139: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg138_1, convert_element_type_305);  arg138_1 = convert_element_type_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_304: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_139, [1000, 4096])
        permute_166: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        mm_105: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_304, permute_166);  view_304 = permute_166 = None
        view_305: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_105, [1, 1000, 4096]);  mm_105 = None
        view_306: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_305, [1, 1000, -1, 128]);  view_305 = None
        permute_167: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_71: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_140: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_167, unsqueeze_71)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_62: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_167, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_45: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_62);  slice_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_61: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_167, 3, 0, 64);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_30: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_45, slice_61], -1);  neg_45 = slice_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_72: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_141: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_30, unsqueeze_72);  cat_30 = None
        add_109: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_307: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_139, [1000, 4096])
        permute_168: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        mm_106: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_307, permute_168);  view_307 = permute_168 = None
        view_308: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [1, 1000, 1024]);  mm_106 = None
        view_309: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_308, [1, 1000, -1, 128]);  view_308 = None
        permute_169: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_142: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_169, unsqueeze_71);  unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_64: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_169, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_46: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_64);  slice_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_63: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_169, 3, 0, 64);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_31: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_46, slice_63], -1);  neg_46 = slice_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_143: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_31, unsqueeze_72);  cat_31 = unsqueeze_72 = None
        add_110: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_73: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_110, 2)
        expand_35: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_73, [1, 8, 4, 1000, 128]);  unsqueeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_64: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_313: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [1, 32, 1000, 128]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_310: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_139, [1000, 4096]);  mul_139 = None
        permute_170: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        mm_107: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_310, permute_170);  view_310 = permute_170 = None
        view_311: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_107, [1, 1000, 1024]);  mm_107 = None
        view_312: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_311, [1, 1000, -1, 128]);  view_311 = None
        permute_171: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_74: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_171, 2)
        expand_36: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_74, [1, 8, 4, 1000, 128]);  unsqueeze_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_65: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_314: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [1, 32, 1000, 128]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_31: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_31, full_default_30);  full_default_31 = full_default_30 = None
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_109, view_313, view_314, where_15, False, scale = 0.08838834764831845);  add_109 = view_313 = view_314 = where_15 = None
        getitem_135: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_15[0];  _scaled_dot_product_cudnn_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_172: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_315: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_172, [1, 1000, -1]);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_316: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_315, [1000, 4096]);  view_315 = None
        permute_173: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        mm_108: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_316, permute_173);  view_316 = permute_173 = None
        view_317: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [1, 1000, 4096]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_111: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_107, view_317);  add_107 = view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_314: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_32: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_314, 2)
        mean_31: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_112: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_31, 1e-05);  mean_31 = None
        rsqrt_31: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_144: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_314, rsqrt_31);  convert_element_type_314 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_315: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None
        mul_145: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg143_1, convert_element_type_315);  arg143_1 = convert_element_type_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_318: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_145, [1000, 4096])
        permute_174: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        mm_109: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_318, permute_174);  view_318 = permute_174 = None
        view_319: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_109, [1, 1000, 14336]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_318: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_319, torch.float32);  view_319 = None
        neg_47: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_318)
        exp_15: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_47);  neg_47 = None
        add_113: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_318, add_113);  convert_element_type_318 = add_113 = None
        convert_element_type_319: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_320: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_145, [1000, 4096]);  mul_145 = None
        permute_175: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        mm_110: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_320, permute_175);  view_320 = permute_175 = None
        view_321: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [1, 1000, 14336]);  mm_110 = None
        mul_146: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_319, view_321);  convert_element_type_319 = view_321 = None
        view_322: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_146, [1000, 14336]);  mul_146 = None
        permute_176: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        mm_111: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_322, permute_176);  view_322 = permute_176 = None
        view_323: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_111, [1, 1000, 4096]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_114: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_111, view_323);  add_111 = view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_324: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_33: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_324, 2)
        mean_32: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_115: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_32, 1e-05);  mean_32 = None
        rsqrt_32: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_147: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_324, rsqrt_32);  convert_element_type_324 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_325: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_147, torch.bfloat16);  mul_147 = None
        mul_148: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg147_1, convert_element_type_325);  arg147_1 = convert_element_type_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_324: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_148, [1000, 4096])
        permute_177: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        mm_112: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_324, permute_177);  view_324 = permute_177 = None
        view_325: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [1, 1000, 4096]);  mm_112 = None
        view_326: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_325, [1, 1000, -1, 128]);  view_325 = None
        permute_178: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_75: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_149: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_178, unsqueeze_75)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_66: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_178, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_48: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_66);  slice_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_65: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_178, 3, 0, 64);  permute_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_32: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_48, slice_65], -1);  neg_48 = slice_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_76: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_150: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_32, unsqueeze_76);  cat_32 = None
        add_116: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_327: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_148, [1000, 4096])
        permute_179: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_113: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_327, permute_179);  view_327 = permute_179 = None
        view_328: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_113, [1, 1000, 1024]);  mm_113 = None
        view_329: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_328, [1, 1000, -1, 128]);  view_328 = None
        permute_180: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_151: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_180, unsqueeze_75);  unsqueeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_68: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_180, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_49: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_68);  slice_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_67: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_180, 3, 0, 64);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_33: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_49, slice_67], -1);  neg_49 = slice_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_152: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_33, unsqueeze_76);  cat_33 = unsqueeze_76 = None
        add_117: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_77: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_117, 2)
        expand_37: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_77, [1, 8, 4, 1000, 128]);  unsqueeze_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_68: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_333: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [1, 32, 1000, 128]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_330: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_148, [1000, 4096]);  mul_148 = None
        permute_181: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        mm_114: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_330, permute_181);  view_330 = permute_181 = None
        view_331: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [1, 1000, 1024]);  mm_114 = None
        view_332: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_331, [1, 1000, -1, 128]);  view_331 = None
        permute_182: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_78: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_182, 2)
        expand_38: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_78, [1, 8, 4, 1000, 128]);  unsqueeze_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_69: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_334: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [1, 32, 1000, 128]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_33: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_33, full_default_32);  full_default_33 = full_default_32 = None
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_116, view_333, view_334, where_16, False, scale = 0.08838834764831845);  add_116 = view_333 = view_334 = where_16 = None
        getitem_144: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_16[0];  _scaled_dot_product_cudnn_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_183: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_335: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_183, [1, 1000, -1]);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_336: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_335, [1000, 4096]);  view_335 = None
        permute_184: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_115: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_336, permute_184);  view_336 = permute_184 = None
        view_337: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_115, [1, 1000, 4096]);  mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_118: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_114, view_337);  add_114 = view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_334: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_118, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_34: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_334, 2)
        mean_33: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_119: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_33, 1e-05);  mean_33 = None
        rsqrt_33: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_153: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_334, rsqrt_33);  convert_element_type_334 = rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_335: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_153, torch.bfloat16);  mul_153 = None
        mul_154: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg152_1, convert_element_type_335);  arg152_1 = convert_element_type_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_338: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_154, [1000, 4096])
        permute_185: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        mm_116: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_338, permute_185);  view_338 = permute_185 = None
        view_339: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [1, 1000, 14336]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_338: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        neg_50: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_338)
        exp_16: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_50);  neg_50 = None
        add_120: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_338, add_120);  convert_element_type_338 = add_120 = None
        convert_element_type_339: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_340: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_154, [1000, 4096]);  mul_154 = None
        permute_186: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        mm_117: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_340, permute_186);  view_340 = permute_186 = None
        view_341: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_117, [1, 1000, 14336]);  mm_117 = None
        mul_155: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_339, view_341);  convert_element_type_339 = view_341 = None
        view_342: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_155, [1000, 14336]);  mul_155 = None
        permute_187: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        mm_118: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_342, permute_187);  view_342 = permute_187 = None
        view_343: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [1, 1000, 4096]);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_121: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_118, view_343);  add_118 = view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_344: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_35: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_344, 2)
        mean_34: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_122: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_34, 1e-05);  mean_34 = None
        rsqrt_34: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_156: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_344, rsqrt_34);  convert_element_type_344 = rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_345: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_156, torch.bfloat16);  mul_156 = None
        mul_157: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg156_1, convert_element_type_345);  arg156_1 = convert_element_type_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_344: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_157, [1000, 4096])
        permute_188: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        mm_119: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_344, permute_188);  view_344 = permute_188 = None
        view_345: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_119, [1, 1000, 4096]);  mm_119 = None
        view_346: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_345, [1, 1000, -1, 128]);  view_345 = None
        permute_189: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_79: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_158: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_189, unsqueeze_79)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_70: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_189, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_51: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_70);  slice_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_69: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_189, 3, 0, 64);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_34: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_51, slice_69], -1);  neg_51 = slice_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_80: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_159: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_34, unsqueeze_80);  cat_34 = None
        add_123: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_347: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_157, [1000, 4096])
        permute_190: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        mm_120: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_347, permute_190);  view_347 = permute_190 = None
        view_348: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [1, 1000, 1024]);  mm_120 = None
        view_349: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_348, [1, 1000, -1, 128]);  view_348 = None
        permute_191: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_160: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_191, unsqueeze_79);  unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_72: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_191, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_52: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_72);  slice_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_71: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_191, 3, 0, 64);  permute_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_35: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_52, slice_71], -1);  neg_52 = slice_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_161: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_35, unsqueeze_80);  cat_35 = unsqueeze_80 = None
        add_124: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_81: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_124, 2)
        expand_39: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_81, [1, 8, 4, 1000, 128]);  unsqueeze_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_72: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_353: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [1, 32, 1000, 128]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_350: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_157, [1000, 4096]);  mul_157 = None
        permute_192: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_121: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_350, permute_192);  view_350 = permute_192 = None
        view_351: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_121, [1, 1000, 1024]);  mm_121 = None
        view_352: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_351, [1, 1000, -1, 128]);  view_351 = None
        permute_193: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_82: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_193, 2)
        expand_40: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_82, [1, 8, 4, 1000, 128]);  unsqueeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_73: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_354: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [1, 32, 1000, 128]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_35: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_34: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_35, full_default_34);  full_default_35 = full_default_34 = None
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_123, view_353, view_354, where_17, False, scale = 0.08838834764831845);  add_123 = view_353 = view_354 = where_17 = None
        getitem_153: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_17[0];  _scaled_dot_product_cudnn_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_194: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_355: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_194, [1, 1000, -1]);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_356: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_355, [1000, 4096]);  view_355 = None
        permute_195: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_122: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_356, permute_195);  view_356 = permute_195 = None
        view_357: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [1, 1000, 4096]);  mm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_125: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, view_357);  add_121 = view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_354: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_36: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_354, 2)
        mean_35: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_126: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_35, 1e-05);  mean_35 = None
        rsqrt_35: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_162: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_354, rsqrt_35);  convert_element_type_354 = rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_355: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16);  mul_162 = None
        mul_163: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg161_1, convert_element_type_355);  arg161_1 = convert_element_type_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_358: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_163, [1000, 4096])
        permute_196: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        mm_123: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_358, permute_196);  view_358 = permute_196 = None
        view_359: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_123, [1, 1000, 14336]);  mm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_358: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_359, torch.float32);  view_359 = None
        neg_53: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_358)
        exp_17: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_53);  neg_53 = None
        add_127: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        div_17: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_358, add_127);  convert_element_type_358 = add_127 = None
        convert_element_type_359: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_360: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_163, [1000, 4096]);  mul_163 = None
        permute_197: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_124: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_360, permute_197);  view_360 = permute_197 = None
        view_361: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [1, 1000, 14336]);  mm_124 = None
        mul_164: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_359, view_361);  convert_element_type_359 = view_361 = None
        view_362: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_164, [1000, 14336]);  mul_164 = None
        permute_198: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        mm_125: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_362, permute_198);  view_362 = permute_198 = None
        view_363: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_125, [1, 1000, 4096]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_128: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, view_363);  add_125 = view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_364: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_128, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_37: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_364, 2)
        mean_36: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_37, [-1], True);  pow_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_129: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_36, 1e-05);  mean_36 = None
        rsqrt_36: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_165: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_364, rsqrt_36);  convert_element_type_364 = rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_365: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None
        mul_166: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg165_1, convert_element_type_365);  arg165_1 = convert_element_type_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_364: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_166, [1000, 4096])
        permute_199: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        mm_126: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_364, permute_199);  view_364 = permute_199 = None
        view_365: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [1, 1000, 4096]);  mm_126 = None
        view_366: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_365, [1, 1000, -1, 128]);  view_365 = None
        permute_200: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_83: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_167: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_200, unsqueeze_83)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_74: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_200, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_54: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_74);  slice_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_73: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_200, 3, 0, 64);  permute_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_36: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_54, slice_73], -1);  neg_54 = slice_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_84: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_168: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_36, unsqueeze_84);  cat_36 = None
        add_130: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_367: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_166, [1000, 4096])
        permute_201: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        mm_127: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_367, permute_201);  view_367 = permute_201 = None
        view_368: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_127, [1, 1000, 1024]);  mm_127 = None
        view_369: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_368, [1, 1000, -1, 128]);  view_368 = None
        permute_202: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_169: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_202, unsqueeze_83);  unsqueeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_76: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_202, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_55: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_76);  slice_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_75: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_202, 3, 0, 64);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_37: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_55, slice_75], -1);  neg_55 = slice_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_170: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_37, unsqueeze_84);  cat_37 = unsqueeze_84 = None
        add_131: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_85: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_131, 2)
        expand_41: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_85, [1, 8, 4, 1000, 128]);  unsqueeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_76: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_373: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [1, 32, 1000, 128]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_370: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_166, [1000, 4096]);  mul_166 = None
        permute_203: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        mm_128: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_370, permute_203);  view_370 = permute_203 = None
        view_371: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [1, 1000, 1024]);  mm_128 = None
        view_372: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_371, [1, 1000, -1, 128]);  view_371 = None
        permute_204: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_86: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_204, 2)
        expand_42: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_86, [1, 8, 4, 1000, 128]);  unsqueeze_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_77: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_374: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [1, 32, 1000, 128]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_37: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_37, full_default_36);  full_default_37 = full_default_36 = None
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_130, view_373, view_374, where_18, False, scale = 0.08838834764831845);  add_130 = view_373 = view_374 = where_18 = None
        getitem_162: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_18[0];  _scaled_dot_product_cudnn_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_205: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_375: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_205, [1, 1000, -1]);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_376: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_375, [1000, 4096]);  view_375 = None
        permute_206: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_129: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_376, permute_206);  view_376 = permute_206 = None
        view_377: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_129, [1, 1000, 4096]);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_132: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_128, view_377);  add_128 = view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_374: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_132, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_38: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_374, 2)
        mean_37: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_133: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_37, 1e-05);  mean_37 = None
        rsqrt_37: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_171: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_374, rsqrt_37);  convert_element_type_374 = rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_375: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_171, torch.bfloat16);  mul_171 = None
        mul_172: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg170_1, convert_element_type_375);  arg170_1 = convert_element_type_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_378: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_172, [1000, 4096])
        permute_207: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        mm_130: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_378, permute_207);  view_378 = permute_207 = None
        view_379: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [1, 1000, 14336]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_378: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_379, torch.float32);  view_379 = None
        neg_56: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_378)
        exp_18: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_56);  neg_56 = None
        add_134: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        div_18: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_378, add_134);  convert_element_type_378 = add_134 = None
        convert_element_type_379: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_380: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_172, [1000, 4096]);  mul_172 = None
        permute_208: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        mm_131: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_380, permute_208);  view_380 = permute_208 = None
        view_381: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_131, [1, 1000, 14336]);  mm_131 = None
        mul_173: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_379, view_381);  convert_element_type_379 = view_381 = None
        view_382: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_173, [1000, 14336]);  mul_173 = None
        permute_209: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_132: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_382, permute_209);  view_382 = permute_209 = None
        view_383: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [1, 1000, 4096]);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_135: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_132, view_383);  add_132 = view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_384: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_39: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_384, 2)
        mean_38: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_136: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_38, 1e-05);  mean_38 = None
        rsqrt_38: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_174: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_384, rsqrt_38);  convert_element_type_384 = rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_385: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        mul_175: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg174_1, convert_element_type_385);  arg174_1 = convert_element_type_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_384: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_175, [1000, 4096])
        permute_210: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        mm_133: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_384, permute_210);  view_384 = permute_210 = None
        view_385: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_133, [1, 1000, 4096]);  mm_133 = None
        view_386: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_385, [1, 1000, -1, 128]);  view_385 = None
        permute_211: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_87: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_176: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_211, unsqueeze_87)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_78: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_211, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_57: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_78);  slice_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_77: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_211, 3, 0, 64);  permute_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_38: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_57, slice_77], -1);  neg_57 = slice_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_88: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_177: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_38, unsqueeze_88);  cat_38 = None
        add_137: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_387: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_175, [1000, 4096])
        permute_212: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        mm_134: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_387, permute_212);  view_387 = permute_212 = None
        view_388: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [1, 1000, 1024]);  mm_134 = None
        view_389: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_388, [1, 1000, -1, 128]);  view_388 = None
        permute_213: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_178: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_213, unsqueeze_87);  unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_80: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_213, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_58: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_80);  slice_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_79: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_213, 3, 0, 64);  permute_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_39: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_58, slice_79], -1);  neg_58 = slice_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_179: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_39, unsqueeze_88);  cat_39 = unsqueeze_88 = None
        add_138: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_89: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_138, 2)
        expand_43: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_89, [1, 8, 4, 1000, 128]);  unsqueeze_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_80: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_393: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [1, 32, 1000, 128]);  clone_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_390: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_175, [1000, 4096]);  mul_175 = None
        permute_214: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        mm_135: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_390, permute_214);  view_390 = permute_214 = None
        view_391: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_135, [1, 1000, 1024]);  mm_135 = None
        view_392: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_391, [1, 1000, -1, 128]);  view_391 = None
        permute_215: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_90: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_215, 2)
        expand_44: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_90, [1, 8, 4, 1000, 128]);  unsqueeze_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_81: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_394: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [1, 32, 1000, 128]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_39: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_39, full_default_38);  full_default_39 = full_default_38 = None
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_137, view_393, view_394, where_19, False, scale = 0.08838834764831845);  add_137 = view_393 = view_394 = where_19 = None
        getitem_171: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_19[0];  _scaled_dot_product_cudnn_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_216: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_395: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_216, [1, 1000, -1]);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_396: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_395, [1000, 4096]);  view_395 = None
        permute_217: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        mm_136: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_396, permute_217);  view_396 = permute_217 = None
        view_397: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [1, 1000, 4096]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_139: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_135, view_397);  add_135 = view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_394: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_40: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_394, 2)
        mean_39: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_140: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_39, 1e-05);  mean_39 = None
        rsqrt_39: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_180: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_394, rsqrt_39);  convert_element_type_394 = rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_395: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_180, torch.bfloat16);  mul_180 = None
        mul_181: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg179_1, convert_element_type_395);  arg179_1 = convert_element_type_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_398: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_181, [1000, 4096])
        permute_218: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        mm_137: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_398, permute_218);  view_398 = permute_218 = None
        view_399: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [1, 1000, 14336]);  mm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_398: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_399, torch.float32);  view_399 = None
        neg_59: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_398)
        exp_19: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_59);  neg_59 = None
        add_141: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_398, add_141);  convert_element_type_398 = add_141 = None
        convert_element_type_399: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_400: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_181, [1000, 4096]);  mul_181 = None
        permute_219: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        mm_138: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_400, permute_219);  view_400 = permute_219 = None
        view_401: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [1, 1000, 14336]);  mm_138 = None
        mul_182: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_399, view_401);  convert_element_type_399 = view_401 = None
        view_402: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_182, [1000, 14336]);  mul_182 = None
        permute_220: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        mm_139: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_402, permute_220);  view_402 = permute_220 = None
        view_403: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_139, [1, 1000, 4096]);  mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_142: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_139, view_403);  add_139 = view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_404: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_142, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_41: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_404, 2)
        mean_40: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_41, [-1], True);  pow_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_143: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_40, 1e-05);  mean_40 = None
        rsqrt_40: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        mul_183: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_404, rsqrt_40);  convert_element_type_404 = rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_405: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_183, torch.bfloat16);  mul_183 = None
        mul_184: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg183_1, convert_element_type_405);  arg183_1 = convert_element_type_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_404: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_184, [1000, 4096])
        permute_221: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        mm_140: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_404, permute_221);  view_404 = permute_221 = None
        view_405: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [1, 1000, 4096]);  mm_140 = None
        view_406: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_405, [1, 1000, -1, 128]);  view_405 = None
        permute_222: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_91: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_185: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_222, unsqueeze_91)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_82: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_222, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_60: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_82);  slice_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_81: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_222, 3, 0, 64);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_40: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_60, slice_81], -1);  neg_60 = slice_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_92: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_186: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_40, unsqueeze_92);  cat_40 = None
        add_144: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_407: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_184, [1000, 4096])
        permute_223: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_141: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_407, permute_223);  view_407 = permute_223 = None
        view_408: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_141, [1, 1000, 1024]);  mm_141 = None
        view_409: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_408, [1, 1000, -1, 128]);  view_408 = None
        permute_224: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_187: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_224, unsqueeze_91);  unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_84: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_224, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_61: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_84);  slice_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_83: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_224, 3, 0, 64);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_41: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_61, slice_83], -1);  neg_61 = slice_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_188: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_41, unsqueeze_92);  cat_41 = unsqueeze_92 = None
        add_145: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_93: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_145, 2)
        expand_45: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_93, [1, 8, 4, 1000, 128]);  unsqueeze_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_84: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_413: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [1, 32, 1000, 128]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_410: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_184, [1000, 4096]);  mul_184 = None
        permute_225: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        mm_142: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_410, permute_225);  view_410 = permute_225 = None
        view_411: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [1, 1000, 1024]);  mm_142 = None
        view_412: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_411, [1, 1000, -1, 128]);  view_411 = None
        permute_226: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_94: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_226, 2)
        expand_46: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_94, [1, 8, 4, 1000, 128]);  unsqueeze_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_85: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_414: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [1, 32, 1000, 128]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_41: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_41, full_default_40);  full_default_41 = full_default_40 = None
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_144, view_413, view_414, where_20, False, scale = 0.08838834764831845);  add_144 = view_413 = view_414 = where_20 = None
        getitem_180: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_20[0];  _scaled_dot_product_cudnn_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_227: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_415: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_227, [1, 1000, -1]);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_416: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_415, [1000, 4096]);  view_415 = None
        permute_228: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        mm_143: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_416, permute_228);  view_416 = permute_228 = None
        view_417: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_143, [1, 1000, 4096]);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_146: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_142, view_417);  add_142 = view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_414: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_146, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_42: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_414, 2)
        mean_41: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_147: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_41, 1e-05);  mean_41 = None
        rsqrt_41: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        mul_189: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_414, rsqrt_41);  convert_element_type_414 = rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_415: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        mul_190: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg188_1, convert_element_type_415);  arg188_1 = convert_element_type_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_418: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_190, [1000, 4096])
        permute_229: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        mm_144: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_418, permute_229);  view_418 = permute_229 = None
        view_419: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [1, 1000, 14336]);  mm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_418: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None
        neg_62: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_418)
        exp_20: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_62);  neg_62 = None
        add_148: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        div_20: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_418, add_148);  convert_element_type_418 = add_148 = None
        convert_element_type_419: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_420: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_190, [1000, 4096]);  mul_190 = None
        permute_230: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        mm_145: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_420, permute_230);  view_420 = permute_230 = None
        view_421: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_145, [1, 1000, 14336]);  mm_145 = None
        mul_191: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_419, view_421);  convert_element_type_419 = view_421 = None
        view_422: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_191, [1000, 14336]);  mul_191 = None
        permute_231: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        mm_146: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_422, permute_231);  view_422 = permute_231 = None
        view_423: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [1, 1000, 4096]);  mm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_149: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_146, view_423);  add_146 = view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_424: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_43: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_424, 2)
        mean_42: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_150: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_42, 1e-05);  mean_42 = None
        rsqrt_42: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        mul_192: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_424, rsqrt_42);  convert_element_type_424 = rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_425: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_192, torch.bfloat16);  mul_192 = None
        mul_193: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg192_1, convert_element_type_425);  arg192_1 = convert_element_type_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_424: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_193, [1000, 4096])
        permute_232: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        mm_147: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_424, permute_232);  view_424 = permute_232 = None
        view_425: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_147, [1, 1000, 4096]);  mm_147 = None
        view_426: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_425, [1, 1000, -1, 128]);  view_425 = None
        permute_233: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_95: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_194: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_233, unsqueeze_95)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_86: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_233, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_63: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_86);  slice_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_85: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_233, 3, 0, 64);  permute_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_42: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_63, slice_85], -1);  neg_63 = slice_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_96: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_195: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_42, unsqueeze_96);  cat_42 = None
        add_151: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_427: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_193, [1000, 4096])
        permute_234: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        mm_148: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_427, permute_234);  view_427 = permute_234 = None
        view_428: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_148, [1, 1000, 1024]);  mm_148 = None
        view_429: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_428, [1, 1000, -1, 128]);  view_428 = None
        permute_235: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_429, [0, 2, 1, 3]);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_196: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_235, unsqueeze_95);  unsqueeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_88: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_235, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_64: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_88);  slice_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_87: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_235, 3, 0, 64);  permute_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_43: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_64, slice_87], -1);  neg_64 = slice_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_197: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_43, unsqueeze_96);  cat_43 = unsqueeze_96 = None
        add_152: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_97: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_152, 2)
        expand_47: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_97, [1, 8, 4, 1000, 128]);  unsqueeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_88: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_433: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [1, 32, 1000, 128]);  clone_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_430: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_193, [1000, 4096]);  mul_193 = None
        permute_236: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        mm_149: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_430, permute_236);  view_430 = permute_236 = None
        view_431: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_149, [1, 1000, 1024]);  mm_149 = None
        view_432: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_431, [1, 1000, -1, 128]);  view_431 = None
        permute_237: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_98: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_237, 2)
        expand_48: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_98, [1, 8, 4, 1000, 128]);  unsqueeze_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_89: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_434: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [1, 32, 1000, 128]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_43: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_43, full_default_42);  full_default_43 = full_default_42 = None
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_151, view_433, view_434, where_21, False, scale = 0.08838834764831845);  add_151 = view_433 = view_434 = where_21 = None
        getitem_189: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_21[0];  _scaled_dot_product_cudnn_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_238: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_435: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_238, [1, 1000, -1]);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_436: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_435, [1000, 4096]);  view_435 = None
        permute_239: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        mm_150: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_436, permute_239);  view_436 = permute_239 = None
        view_437: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_150, [1, 1000, 4096]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_153: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_149, view_437);  add_149 = view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_434: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_153, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_44: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_434, 2)
        mean_43: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_154: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_43, 1e-05);  mean_43 = None
        rsqrt_43: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        mul_198: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_434, rsqrt_43);  convert_element_type_434 = rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_435: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_198, torch.bfloat16);  mul_198 = None
        mul_199: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg197_1, convert_element_type_435);  arg197_1 = convert_element_type_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_438: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_199, [1000, 4096])
        permute_240: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        mm_151: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_438, permute_240);  view_438 = permute_240 = None
        view_439: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_151, [1, 1000, 14336]);  mm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_438: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_439, torch.float32);  view_439 = None
        neg_65: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_438)
        exp_21: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_65);  neg_65 = None
        add_155: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        div_21: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_438, add_155);  convert_element_type_438 = add_155 = None
        convert_element_type_439: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_440: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_199, [1000, 4096]);  mul_199 = None
        permute_241: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        mm_152: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_440, permute_241);  view_440 = permute_241 = None
        view_441: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_152, [1, 1000, 14336]);  mm_152 = None
        mul_200: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_439, view_441);  convert_element_type_439 = view_441 = None
        view_442: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_200, [1000, 14336]);  mul_200 = None
        permute_242: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        mm_153: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_442, permute_242);  view_442 = permute_242 = None
        view_443: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_153, [1, 1000, 4096]);  mm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_156: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_153, view_443);  add_153 = view_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_444: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_156, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_45: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_444, 2)
        mean_44: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_45, [-1], True);  pow_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_157: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_44, 1e-05);  mean_44 = None
        rsqrt_44: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        mul_201: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, rsqrt_44);  convert_element_type_444 = rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_445: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16);  mul_201 = None
        mul_202: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg201_1, convert_element_type_445);  arg201_1 = convert_element_type_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_444: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_202, [1000, 4096])
        permute_243: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        mm_154: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_444, permute_243);  view_444 = permute_243 = None
        view_445: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_154, [1, 1000, 4096]);  mm_154 = None
        view_446: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_445, [1, 1000, -1, 128]);  view_445 = None
        permute_244: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_99: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_203: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_244, unsqueeze_99)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_90: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_244, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_66: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_90);  slice_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_89: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_244, 3, 0, 64);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_44: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_66, slice_89], -1);  neg_66 = slice_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_100: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_204: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_44, unsqueeze_100);  cat_44 = None
        add_158: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_447: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_202, [1000, 4096])
        permute_245: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        mm_155: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_447, permute_245);  view_447 = permute_245 = None
        view_448: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_155, [1, 1000, 1024]);  mm_155 = None
        view_449: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_448, [1, 1000, -1, 128]);  view_448 = None
        permute_246: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_449, [0, 2, 1, 3]);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_205: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_246, unsqueeze_99);  unsqueeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_92: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_246, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_67: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_92);  slice_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_91: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_246, 3, 0, 64);  permute_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_45: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_67, slice_91], -1);  neg_67 = slice_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_206: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_45, unsqueeze_100);  cat_45 = unsqueeze_100 = None
        add_159: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_101: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_159, 2)
        expand_49: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_101, [1, 8, 4, 1000, 128]);  unsqueeze_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_92: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_453: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [1, 32, 1000, 128]);  clone_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_450: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_202, [1000, 4096]);  mul_202 = None
        permute_247: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        mm_156: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_450, permute_247);  view_450 = permute_247 = None
        view_451: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_156, [1, 1000, 1024]);  mm_156 = None
        view_452: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_451, [1, 1000, -1, 128]);  view_451 = None
        permute_248: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_102: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_248, 2)
        expand_50: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_102, [1, 8, 4, 1000, 128]);  unsqueeze_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_93: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_454: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [1, 32, 1000, 128]);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_45: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_45, full_default_44);  full_default_45 = full_default_44 = None
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_158, view_453, view_454, where_22, False, scale = 0.08838834764831845);  add_158 = view_453 = view_454 = where_22 = None
        getitem_198: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_22[0];  _scaled_dot_product_cudnn_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_249: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3]);  getitem_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_455: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_249, [1, 1000, -1]);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_456: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_455, [1000, 4096]);  view_455 = None
        permute_250: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        mm_157: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_456, permute_250);  view_456 = permute_250 = None
        view_457: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_157, [1, 1000, 4096]);  mm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_160: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_156, view_457);  add_156 = view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_454: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_160, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_46: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_454, 2)
        mean_45: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_161: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_45, 1e-05);  mean_45 = None
        rsqrt_45: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        mul_207: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_454, rsqrt_45);  convert_element_type_454 = rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_455: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_207, torch.bfloat16);  mul_207 = None
        mul_208: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg206_1, convert_element_type_455);  arg206_1 = convert_element_type_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_458: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_208, [1000, 4096])
        permute_251: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        mm_158: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_458, permute_251);  view_458 = permute_251 = None
        view_459: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_158, [1, 1000, 14336]);  mm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_458: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        neg_68: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_458)
        exp_22: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_68);  neg_68 = None
        add_162: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_458, add_162);  convert_element_type_458 = add_162 = None
        convert_element_type_459: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_460: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_208, [1000, 4096]);  mul_208 = None
        permute_252: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        mm_159: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_460, permute_252);  view_460 = permute_252 = None
        view_461: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_159, [1, 1000, 14336]);  mm_159 = None
        mul_209: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_459, view_461);  convert_element_type_459 = view_461 = None
        view_462: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_209, [1000, 14336]);  mul_209 = None
        permute_253: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        mm_160: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_462, permute_253);  view_462 = permute_253 = None
        view_463: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_160, [1, 1000, 4096]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_163: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_160, view_463);  add_160 = view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_464: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_163, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_47: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_464, 2)
        mean_46: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_164: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_46, 1e-05);  mean_46 = None
        rsqrt_46: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        mul_210: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_464, rsqrt_46);  convert_element_type_464 = rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_465: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_210, torch.bfloat16);  mul_210 = None
        mul_211: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg210_1, convert_element_type_465);  arg210_1 = convert_element_type_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_464: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_211, [1000, 4096])
        permute_254: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        mm_161: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_464, permute_254);  view_464 = permute_254 = None
        view_465: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_161, [1, 1000, 4096]);  mm_161 = None
        view_466: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_465, [1, 1000, -1, 128]);  view_465 = None
        permute_255: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_103: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_212: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_255, unsqueeze_103)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_94: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_255, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_69: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_94);  slice_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_93: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_255, 3, 0, 64);  permute_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_46: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_69, slice_93], -1);  neg_69 = slice_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_104: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_213: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_46, unsqueeze_104);  cat_46 = None
        add_165: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_467: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_211, [1000, 4096])
        permute_256: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        mm_162: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_467, permute_256);  view_467 = permute_256 = None
        view_468: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_162, [1, 1000, 1024]);  mm_162 = None
        view_469: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_468, [1, 1000, -1, 128]);  view_468 = None
        permute_257: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_214: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_257, unsqueeze_103);  unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_96: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_257, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_70: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_96);  slice_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_95: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_257, 3, 0, 64);  permute_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_47: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_70, slice_95], -1);  neg_70 = slice_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_215: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_47, unsqueeze_104);  cat_47 = unsqueeze_104 = None
        add_166: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_105: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_166, 2)
        expand_51: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_105, [1, 8, 4, 1000, 128]);  unsqueeze_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_96: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_473: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [1, 32, 1000, 128]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_470: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_211, [1000, 4096]);  mul_211 = None
        permute_258: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        mm_163: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_470, permute_258);  view_470 = permute_258 = None
        view_471: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_163, [1, 1000, 1024]);  mm_163 = None
        view_472: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_471, [1, 1000, -1, 128]);  view_471 = None
        permute_259: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_106: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_259, 2)
        expand_52: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_106, [1, 8, 4, 1000, 128]);  unsqueeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_97: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_474: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [1, 32, 1000, 128]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_47: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_47, full_default_46);  full_default_47 = full_default_46 = None
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_165, view_473, view_474, where_23, False, scale = 0.08838834764831845);  add_165 = view_473 = view_474 = where_23 = None
        getitem_207: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_23[0];  _scaled_dot_product_cudnn_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_260: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_475: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_260, [1, 1000, -1]);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_476: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_475, [1000, 4096]);  view_475 = None
        permute_261: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        mm_164: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_476, permute_261);  view_476 = permute_261 = None
        view_477: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_164, [1, 1000, 4096]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_167: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_163, view_477);  add_163 = view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_474: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_167, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_48: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_474, 2)
        mean_47: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_168: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_47, 1e-05);  mean_47 = None
        rsqrt_47: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_216: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_474, rsqrt_47);  convert_element_type_474 = rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_475: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_216, torch.bfloat16);  mul_216 = None
        mul_217: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg215_1, convert_element_type_475);  arg215_1 = convert_element_type_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_478: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_217, [1000, 4096])
        permute_262: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        mm_165: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_478, permute_262);  view_478 = permute_262 = None
        view_479: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_165, [1, 1000, 14336]);  mm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_478: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_479, torch.float32);  view_479 = None
        neg_71: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_478)
        exp_23: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_71);  neg_71 = None
        add_169: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        div_23: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_478, add_169);  convert_element_type_478 = add_169 = None
        convert_element_type_479: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_480: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_217, [1000, 4096]);  mul_217 = None
        permute_263: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        mm_166: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_480, permute_263);  view_480 = permute_263 = None
        view_481: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_166, [1, 1000, 14336]);  mm_166 = None
        mul_218: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_479, view_481);  convert_element_type_479 = view_481 = None
        view_482: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_218, [1000, 14336]);  mul_218 = None
        permute_264: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        mm_167: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_482, permute_264);  view_482 = permute_264 = None
        view_483: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_167, [1, 1000, 4096]);  mm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_170: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_167, view_483);  add_167 = view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_484: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_170, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_49: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_484, 2)
        mean_48: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_49, [-1], True);  pow_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_171: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_48, 1e-05);  mean_48 = None
        rsqrt_48: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        mul_219: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_484, rsqrt_48);  convert_element_type_484 = rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_485: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_219, torch.bfloat16);  mul_219 = None
        mul_220: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg219_1, convert_element_type_485);  arg219_1 = convert_element_type_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_484: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_220, [1000, 4096])
        permute_265: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        mm_168: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_484, permute_265);  view_484 = permute_265 = None
        view_485: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_168, [1, 1000, 4096]);  mm_168 = None
        view_486: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_485, [1, 1000, -1, 128]);  view_485 = None
        permute_266: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_107: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_221: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_266, unsqueeze_107)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_98: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_266, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_72: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_98);  slice_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_97: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_266, 3, 0, 64);  permute_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_48: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_72, slice_97], -1);  neg_72 = slice_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_108: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_222: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_48, unsqueeze_108);  cat_48 = None
        add_172: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_487: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_220, [1000, 4096])
        permute_267: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        mm_169: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_487, permute_267);  view_487 = permute_267 = None
        view_488: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_169, [1, 1000, 1024]);  mm_169 = None
        view_489: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_488, [1, 1000, -1, 128]);  view_488 = None
        permute_268: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_223: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_268, unsqueeze_107);  unsqueeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_100: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_268, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_73: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_100);  slice_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_99: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_268, 3, 0, 64);  permute_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_49: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_73, slice_99], -1);  neg_73 = slice_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_224: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_49, unsqueeze_108);  cat_49 = unsqueeze_108 = None
        add_173: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_109: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_173, 2)
        expand_53: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_109, [1, 8, 4, 1000, 128]);  unsqueeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_100: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_493: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [1, 32, 1000, 128]);  clone_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_490: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_220, [1000, 4096]);  mul_220 = None
        permute_269: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        mm_170: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_490, permute_269);  view_490 = permute_269 = None
        view_491: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_170, [1, 1000, 1024]);  mm_170 = None
        view_492: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_491, [1, 1000, -1, 128]);  view_491 = None
        permute_270: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_110: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_270, 2)
        expand_54: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_110, [1, 8, 4, 1000, 128]);  unsqueeze_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_101: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_494: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [1, 32, 1000, 128]);  clone_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_49: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_49, full_default_48);  full_default_49 = full_default_48 = None
        _scaled_dot_product_cudnn_attention_24 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_172, view_493, view_494, where_24, False, scale = 0.08838834764831845);  add_172 = view_493 = view_494 = where_24 = None
        getitem_216: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_24[0];  _scaled_dot_product_cudnn_attention_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_271: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_495: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_271, [1, 1000, -1]);  permute_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_496: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_495, [1000, 4096]);  view_495 = None
        permute_272: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        mm_171: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_496, permute_272);  view_496 = permute_272 = None
        view_497: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_171, [1, 1000, 4096]);  mm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_174: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_170, view_497);  add_170 = view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_494: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_174, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_50: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_494, 2)
        mean_49: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_175: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_49, 1e-05);  mean_49 = None
        rsqrt_49: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        mul_225: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_494, rsqrt_49);  convert_element_type_494 = rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_495: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_225, torch.bfloat16);  mul_225 = None
        mul_226: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg224_1, convert_element_type_495);  arg224_1 = convert_element_type_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_498: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_226, [1000, 4096])
        permute_273: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        mm_172: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_498, permute_273);  view_498 = permute_273 = None
        view_499: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_172, [1, 1000, 14336]);  mm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_498: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_499, torch.float32);  view_499 = None
        neg_74: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_498)
        exp_24: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_74);  neg_74 = None
        add_176: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        div_24: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_498, add_176);  convert_element_type_498 = add_176 = None
        convert_element_type_499: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_500: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_226, [1000, 4096]);  mul_226 = None
        permute_274: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        mm_173: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_500, permute_274);  view_500 = permute_274 = None
        view_501: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_173, [1, 1000, 14336]);  mm_173 = None
        mul_227: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_499, view_501);  convert_element_type_499 = view_501 = None
        view_502: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_227, [1000, 14336]);  mul_227 = None
        permute_275: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        mm_174: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_502, permute_275);  view_502 = permute_275 = None
        view_503: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_174, [1, 1000, 4096]);  mm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_177: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_174, view_503);  add_174 = view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_504: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_177, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_51: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_504, 2)
        mean_50: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_178: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_50, 1e-05);  mean_50 = None
        rsqrt_50: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_228: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_504, rsqrt_50);  convert_element_type_504 = rsqrt_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_505: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_228, torch.bfloat16);  mul_228 = None
        mul_229: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg228_1, convert_element_type_505);  arg228_1 = convert_element_type_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_504: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_229, [1000, 4096])
        permute_276: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        mm_175: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_504, permute_276);  view_504 = permute_276 = None
        view_505: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_175, [1, 1000, 4096]);  mm_175 = None
        view_506: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_505, [1, 1000, -1, 128]);  view_505 = None
        permute_277: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_506, [0, 2, 1, 3]);  view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_111: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_230: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_277, unsqueeze_111)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_102: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_277, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_75: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_102);  slice_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_101: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_277, 3, 0, 64);  permute_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_50: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_75, slice_101], -1);  neg_75 = slice_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_112: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_231: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_50, unsqueeze_112);  cat_50 = None
        add_179: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_507: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_229, [1000, 4096])
        permute_278: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
        mm_176: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_507, permute_278);  view_507 = permute_278 = None
        view_508: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_176, [1, 1000, 1024]);  mm_176 = None
        view_509: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_508, [1, 1000, -1, 128]);  view_508 = None
        permute_279: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_509, [0, 2, 1, 3]);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_232: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_279, unsqueeze_111);  unsqueeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_104: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_279, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_76: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_104);  slice_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_103: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_279, 3, 0, 64);  permute_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_51: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_76, slice_103], -1);  neg_76 = slice_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_233: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_51, unsqueeze_112);  cat_51 = unsqueeze_112 = None
        add_180: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_113: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_180, 2)
        expand_55: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_113, [1, 8, 4, 1000, 128]);  unsqueeze_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_104: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_513: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_104, [1, 32, 1000, 128]);  clone_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_510: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_229, [1000, 4096]);  mul_229 = None
        permute_280: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        mm_177: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_510, permute_280);  view_510 = permute_280 = None
        view_511: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_177, [1, 1000, 1024]);  mm_177 = None
        view_512: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_511, [1, 1000, -1, 128]);  view_511 = None
        permute_281: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_114: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_281, 2)
        expand_56: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_114, [1, 8, 4, 1000, 128]);  unsqueeze_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_105: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_514: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [1, 32, 1000, 128]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_51: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_50: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_51, full_default_50);  full_default_51 = full_default_50 = None
        _scaled_dot_product_cudnn_attention_25 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_179, view_513, view_514, where_25, False, scale = 0.08838834764831845);  add_179 = view_513 = view_514 = where_25 = None
        getitem_225: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_25[0];  _scaled_dot_product_cudnn_attention_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_282: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3]);  getitem_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_515: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_282, [1, 1000, -1]);  permute_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_516: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_515, [1000, 4096]);  view_515 = None
        permute_283: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        mm_178: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_516, permute_283);  view_516 = permute_283 = None
        view_517: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_178, [1, 1000, 4096]);  mm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_181: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_177, view_517);  add_177 = view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_514: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_181, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_52: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_514, 2)
        mean_51: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_182: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_51, 1e-05);  mean_51 = None
        rsqrt_51: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        mul_234: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_514, rsqrt_51);  convert_element_type_514 = rsqrt_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_515: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_234, torch.bfloat16);  mul_234 = None
        mul_235: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg233_1, convert_element_type_515);  arg233_1 = convert_element_type_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_518: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_235, [1000, 4096])
        permute_284: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        mm_179: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_518, permute_284);  view_518 = permute_284 = None
        view_519: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_179, [1, 1000, 14336]);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_518: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_519, torch.float32);  view_519 = None
        neg_77: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_518)
        exp_25: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_77);  neg_77 = None
        add_183: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_518, add_183);  convert_element_type_518 = add_183 = None
        convert_element_type_519: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_520: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_235, [1000, 4096]);  mul_235 = None
        permute_285: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        mm_180: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_520, permute_285);  view_520 = permute_285 = None
        view_521: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_180, [1, 1000, 14336]);  mm_180 = None
        mul_236: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_519, view_521);  convert_element_type_519 = view_521 = None
        view_522: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_236, [1000, 14336]);  mul_236 = None
        permute_286: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        mm_181: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_522, permute_286);  view_522 = permute_286 = None
        view_523: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_181, [1, 1000, 4096]);  mm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_184: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_181, view_523);  add_181 = view_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_524: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_53: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_524, 2)
        mean_52: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_53, [-1], True);  pow_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_185: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_52, 1e-05);  mean_52 = None
        rsqrt_52: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        mul_237: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_524, rsqrt_52);  convert_element_type_524 = rsqrt_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_525: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_237, torch.bfloat16);  mul_237 = None
        mul_238: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg237_1, convert_element_type_525);  arg237_1 = convert_element_type_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_524: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_238, [1000, 4096])
        permute_287: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        mm_182: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_524, permute_287);  view_524 = permute_287 = None
        view_525: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_182, [1, 1000, 4096]);  mm_182 = None
        view_526: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_525, [1, 1000, -1, 128]);  view_525 = None
        permute_288: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_115: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_239: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_288, unsqueeze_115)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_106: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_288, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_78: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_106);  slice_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_105: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_288, 3, 0, 64);  permute_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_52: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_78, slice_105], -1);  neg_78 = slice_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_116: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_240: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_52, unsqueeze_116);  cat_52 = None
        add_186: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_527: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_238, [1000, 4096])
        permute_289: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        mm_183: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_527, permute_289);  view_527 = permute_289 = None
        view_528: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_183, [1, 1000, 1024]);  mm_183 = None
        view_529: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_528, [1, 1000, -1, 128]);  view_528 = None
        permute_290: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_529, [0, 2, 1, 3]);  view_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_241: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_290, unsqueeze_115);  unsqueeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_108: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_290, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_79: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_108);  slice_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_107: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_290, 3, 0, 64);  permute_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_53: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_79, slice_107], -1);  neg_79 = slice_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_242: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_53, unsqueeze_116);  cat_53 = unsqueeze_116 = None
        add_187: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_117: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_187, 2)
        expand_57: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_117, [1, 8, 4, 1000, 128]);  unsqueeze_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_108: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_533: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [1, 32, 1000, 128]);  clone_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_530: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_238, [1000, 4096]);  mul_238 = None
        permute_291: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        mm_184: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_530, permute_291);  view_530 = permute_291 = None
        view_531: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_184, [1, 1000, 1024]);  mm_184 = None
        view_532: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_531, [1, 1000, -1, 128]);  view_531 = None
        permute_292: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_118: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_292, 2)
        expand_58: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_118, [1, 8, 4, 1000, 128]);  unsqueeze_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_109: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_534: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [1, 32, 1000, 128]);  clone_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_53: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_53, full_default_52);  full_default_53 = full_default_52 = None
        _scaled_dot_product_cudnn_attention_26 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_186, view_533, view_534, where_26, False, scale = 0.08838834764831845);  add_186 = view_533 = view_534 = where_26 = None
        getitem_234: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_26[0];  _scaled_dot_product_cudnn_attention_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_293: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_234, [0, 2, 1, 3]);  getitem_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_535: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_293, [1, 1000, -1]);  permute_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_536: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_535, [1000, 4096]);  view_535 = None
        permute_294: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        mm_185: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_536, permute_294);  view_536 = permute_294 = None
        view_537: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_185, [1, 1000, 4096]);  mm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_188: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_184, view_537);  add_184 = view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_534: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_188, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_54: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_534, 2)
        mean_53: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_189: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_53, 1e-05);  mean_53 = None
        rsqrt_53: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
        mul_243: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_534, rsqrt_53);  convert_element_type_534 = rsqrt_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_535: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_243, torch.bfloat16);  mul_243 = None
        mul_244: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg242_1, convert_element_type_535);  arg242_1 = convert_element_type_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_538: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_244, [1000, 4096])
        permute_295: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        mm_186: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_538, permute_295);  view_538 = permute_295 = None
        view_539: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_186, [1, 1000, 14336]);  mm_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_538: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_539, torch.float32);  view_539 = None
        neg_80: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_538)
        exp_26: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_80);  neg_80 = None
        add_190: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        div_26: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_538, add_190);  convert_element_type_538 = add_190 = None
        convert_element_type_539: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_540: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_244, [1000, 4096]);  mul_244 = None
        permute_296: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        mm_187: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_540, permute_296);  view_540 = permute_296 = None
        view_541: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_187, [1, 1000, 14336]);  mm_187 = None
        mul_245: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_539, view_541);  convert_element_type_539 = view_541 = None
        view_542: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_245, [1000, 14336]);  mul_245 = None
        permute_297: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        mm_188: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_542, permute_297);  view_542 = permute_297 = None
        view_543: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_188, [1, 1000, 4096]);  mm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_191: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_188, view_543);  add_188 = view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_544: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_55: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_544, 2)
        mean_54: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_192: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_54, 1e-05);  mean_54 = None
        rsqrt_54: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_246: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_544, rsqrt_54);  convert_element_type_544 = rsqrt_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_545: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None
        mul_247: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg246_1, convert_element_type_545);  arg246_1 = convert_element_type_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_544: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_247, [1000, 4096])
        permute_298: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        mm_189: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_544, permute_298);  view_544 = permute_298 = None
        view_545: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_189, [1, 1000, 4096]);  mm_189 = None
        view_546: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_545, [1, 1000, -1, 128]);  view_545 = None
        permute_299: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_119: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_248: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_299, unsqueeze_119)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_110: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_299, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_81: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_110);  slice_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_109: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_299, 3, 0, 64);  permute_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_54: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_81, slice_109], -1);  neg_81 = slice_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_120: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_249: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_54, unsqueeze_120);  cat_54 = None
        add_193: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_547: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_247, [1000, 4096])
        permute_300: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        mm_190: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_547, permute_300);  view_547 = permute_300 = None
        view_548: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_190, [1, 1000, 1024]);  mm_190 = None
        view_549: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_548, [1, 1000, -1, 128]);  view_548 = None
        permute_301: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_549, [0, 2, 1, 3]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_250: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_301, unsqueeze_119);  unsqueeze_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_112: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_301, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_82: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_112);  slice_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_111: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_301, 3, 0, 64);  permute_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_55: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_82, slice_111], -1);  neg_82 = slice_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_251: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_55, unsqueeze_120);  cat_55 = unsqueeze_120 = None
        add_194: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_121: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_194, 2)
        expand_59: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_121, [1, 8, 4, 1000, 128]);  unsqueeze_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_112: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_553: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_112, [1, 32, 1000, 128]);  clone_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_550: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_247, [1000, 4096]);  mul_247 = None
        permute_302: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        mm_191: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_550, permute_302);  view_550 = permute_302 = None
        view_551: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_191, [1, 1000, 1024]);  mm_191 = None
        view_552: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_551, [1, 1000, -1, 128]);  view_551 = None
        permute_303: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_122: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_303, 2)
        expand_60: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_122, [1, 8, 4, 1000, 128]);  unsqueeze_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_113: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_554: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [1, 32, 1000, 128]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_55: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_55, full_default_54);  full_default_55 = full_default_54 = None
        _scaled_dot_product_cudnn_attention_27 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_193, view_553, view_554, where_27, False, scale = 0.08838834764831845);  add_193 = view_553 = view_554 = where_27 = None
        getitem_243: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_27[0];  _scaled_dot_product_cudnn_attention_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_304: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_243, [0, 2, 1, 3]);  getitem_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_555: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_304, [1, 1000, -1]);  permute_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_556: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_555, [1000, 4096]);  view_555 = None
        permute_305: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        mm_192: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_556, permute_305);  view_556 = permute_305 = None
        view_557: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_192, [1, 1000, 4096]);  mm_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_195: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_191, view_557);  add_191 = view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_554: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_195, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_56: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_554, 2)
        mean_55: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_196: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_55, 1e-05);  mean_55 = None
        rsqrt_55: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        mul_252: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_554, rsqrt_55);  convert_element_type_554 = rsqrt_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_555: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_252, torch.bfloat16);  mul_252 = None
        mul_253: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg251_1, convert_element_type_555);  arg251_1 = convert_element_type_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_558: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_253, [1000, 4096])
        permute_306: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        mm_193: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_558, permute_306);  view_558 = permute_306 = None
        view_559: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_193, [1, 1000, 14336]);  mm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_558: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_559, torch.float32);  view_559 = None
        neg_83: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_558)
        exp_27: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_83);  neg_83 = None
        add_197: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        div_27: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_558, add_197);  convert_element_type_558 = add_197 = None
        convert_element_type_559: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_560: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_253, [1000, 4096]);  mul_253 = None
        permute_307: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        mm_194: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_560, permute_307);  view_560 = permute_307 = None
        view_561: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_194, [1, 1000, 14336]);  mm_194 = None
        mul_254: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_559, view_561);  convert_element_type_559 = view_561 = None
        view_562: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_254, [1000, 14336]);  mul_254 = None
        permute_308: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        mm_195: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_562, permute_308);  view_562 = permute_308 = None
        view_563: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_195, [1, 1000, 4096]);  mm_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_198: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_195, view_563);  add_195 = view_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_564: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_198, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_57: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_564, 2)
        mean_56: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_57, [-1], True);  pow_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_199: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_56, 1e-05);  mean_56 = None
        rsqrt_56: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        mul_255: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_564, rsqrt_56);  convert_element_type_564 = rsqrt_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_565: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_255, torch.bfloat16);  mul_255 = None
        mul_256: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg255_1, convert_element_type_565);  arg255_1 = convert_element_type_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_564: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_256, [1000, 4096])
        permute_309: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        mm_196: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_564, permute_309);  view_564 = permute_309 = None
        view_565: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_196, [1, 1000, 4096]);  mm_196 = None
        view_566: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_565, [1, 1000, -1, 128]);  view_565 = None
        permute_310: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_123: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_257: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_310, unsqueeze_123)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_114: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_310, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_84: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_114);  slice_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_113: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_310, 3, 0, 64);  permute_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_56: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_84, slice_113], -1);  neg_84 = slice_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_124: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_258: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_56, unsqueeze_124);  cat_56 = None
        add_200: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_567: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_256, [1000, 4096])
        permute_311: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        mm_197: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_567, permute_311);  view_567 = permute_311 = None
        view_568: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_197, [1, 1000, 1024]);  mm_197 = None
        view_569: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_568, [1, 1000, -1, 128]);  view_568 = None
        permute_312: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_259: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_312, unsqueeze_123);  unsqueeze_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_116: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_312, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_85: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_116);  slice_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_115: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_312, 3, 0, 64);  permute_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_57: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_85, slice_115], -1);  neg_85 = slice_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_260: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_57, unsqueeze_124);  cat_57 = unsqueeze_124 = None
        add_201: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_125: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_201, 2)
        expand_61: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_125, [1, 8, 4, 1000, 128]);  unsqueeze_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_116: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_573: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [1, 32, 1000, 128]);  clone_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_570: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_256, [1000, 4096]);  mul_256 = None
        permute_313: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        mm_198: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_570, permute_313);  view_570 = permute_313 = None
        view_571: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_198, [1, 1000, 1024]);  mm_198 = None
        view_572: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_571, [1, 1000, -1, 128]);  view_571 = None
        permute_314: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_126: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_314, 2)
        expand_62: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_126, [1, 8, 4, 1000, 128]);  unsqueeze_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_117: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_574: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [1, 32, 1000, 128]);  clone_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_57: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_57, full_default_56);  full_default_57 = full_default_56 = None
        _scaled_dot_product_cudnn_attention_28 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_200, view_573, view_574, where_28, False, scale = 0.08838834764831845);  add_200 = view_573 = view_574 = where_28 = None
        getitem_252: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_28[0];  _scaled_dot_product_cudnn_attention_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_315: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_252, [0, 2, 1, 3]);  getitem_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_575: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_315, [1, 1000, -1]);  permute_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_576: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_575, [1000, 4096]);  view_575 = None
        permute_316: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        mm_199: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_576, permute_316);  view_576 = permute_316 = None
        view_577: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_199, [1, 1000, 4096]);  mm_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_202: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_198, view_577);  add_198 = view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_574: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_202, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_58: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_574, 2)
        mean_57: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_203: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_57, 1e-05);  mean_57 = None
        rsqrt_57: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        mul_261: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_574, rsqrt_57);  convert_element_type_574 = rsqrt_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_575: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_261, torch.bfloat16);  mul_261 = None
        mul_262: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg260_1, convert_element_type_575);  arg260_1 = convert_element_type_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_578: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_262, [1000, 4096])
        permute_317: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        mm_200: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_578, permute_317);  view_578 = permute_317 = None
        view_579: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_200, [1, 1000, 14336]);  mm_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_578: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_579, torch.float32);  view_579 = None
        neg_86: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_578)
        exp_28: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_86);  neg_86 = None
        add_204: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        div_28: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_578, add_204);  convert_element_type_578 = add_204 = None
        convert_element_type_579: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_580: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_262, [1000, 4096]);  mul_262 = None
        permute_318: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        mm_201: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_580, permute_318);  view_580 = permute_318 = None
        view_581: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_201, [1, 1000, 14336]);  mm_201 = None
        mul_263: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_579, view_581);  convert_element_type_579 = view_581 = None
        view_582: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_263, [1000, 14336]);  mul_263 = None
        permute_319: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        mm_202: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_582, permute_319);  view_582 = permute_319 = None
        view_583: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_202, [1, 1000, 4096]);  mm_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_205: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_202, view_583);  add_202 = view_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_584: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_205, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_59: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_584, 2)
        mean_58: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_59, [-1], True);  pow_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_206: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_58, 1e-05);  mean_58 = None
        rsqrt_58: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        mul_264: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_584, rsqrt_58);  convert_element_type_584 = rsqrt_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_585: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_264, torch.bfloat16);  mul_264 = None
        mul_265: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg264_1, convert_element_type_585);  arg264_1 = convert_element_type_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_584: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_265, [1000, 4096])
        permute_320: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        mm_203: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_584, permute_320);  view_584 = permute_320 = None
        view_585: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_203, [1, 1000, 4096]);  mm_203 = None
        view_586: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_585, [1, 1000, -1, 128]);  view_585 = None
        permute_321: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_586, [0, 2, 1, 3]);  view_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_127: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_266: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_321, unsqueeze_127)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_118: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_321, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_87: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_118);  slice_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_117: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_321, 3, 0, 64);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_58: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_87, slice_117], -1);  neg_87 = slice_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_128: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_267: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_58, unsqueeze_128);  cat_58 = None
        add_207: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_587: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_265, [1000, 4096])
        permute_322: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        mm_204: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_587, permute_322);  view_587 = permute_322 = None
        view_588: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_204, [1, 1000, 1024]);  mm_204 = None
        view_589: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_588, [1, 1000, -1, 128]);  view_588 = None
        permute_323: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_589, [0, 2, 1, 3]);  view_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_268: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_323, unsqueeze_127);  unsqueeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_120: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_323, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_88: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_120);  slice_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_119: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_323, 3, 0, 64);  permute_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_59: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_88, slice_119], -1);  neg_88 = slice_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_269: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_59, unsqueeze_128);  cat_59 = unsqueeze_128 = None
        add_208: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_129: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_208, 2)
        expand_63: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_129, [1, 8, 4, 1000, 128]);  unsqueeze_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_120: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_593: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_120, [1, 32, 1000, 128]);  clone_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_590: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_265, [1000, 4096]);  mul_265 = None
        permute_324: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        mm_205: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_590, permute_324);  view_590 = permute_324 = None
        view_591: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_205, [1, 1000, 1024]);  mm_205 = None
        view_592: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_591, [1, 1000, -1, 128]);  view_591 = None
        permute_325: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_130: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_325, 2)
        expand_64: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_130, [1, 8, 4, 1000, 128]);  unsqueeze_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_121: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_594: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [1, 32, 1000, 128]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_59: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_29: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_59, full_default_58);  full_default_59 = full_default_58 = None
        _scaled_dot_product_cudnn_attention_29 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_207, view_593, view_594, where_29, False, scale = 0.08838834764831845);  add_207 = view_593 = view_594 = where_29 = None
        getitem_261: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_29[0];  _scaled_dot_product_cudnn_attention_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_326: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_261, [0, 2, 1, 3]);  getitem_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_595: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_326, [1, 1000, -1]);  permute_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_596: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_595, [1000, 4096]);  view_595 = None
        permute_327: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        mm_206: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_596, permute_327);  view_596 = permute_327 = None
        view_597: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_206, [1, 1000, 4096]);  mm_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_209: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_205, view_597);  add_205 = view_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_594: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_209, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_60: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_594, 2)
        mean_59: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_60, [-1], True);  pow_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_210: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_59, 1e-05);  mean_59 = None
        rsqrt_59: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        mul_270: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_594, rsqrt_59);  convert_element_type_594 = rsqrt_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_595: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_270, torch.bfloat16);  mul_270 = None
        mul_271: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg269_1, convert_element_type_595);  arg269_1 = convert_element_type_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_598: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_271, [1000, 4096])
        permute_328: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        mm_207: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_598, permute_328);  view_598 = permute_328 = None
        view_599: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_207, [1, 1000, 14336]);  mm_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_598: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_599, torch.float32);  view_599 = None
        neg_89: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_598)
        exp_29: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_89);  neg_89 = None
        add_211: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        div_29: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_598, add_211);  convert_element_type_598 = add_211 = None
        convert_element_type_599: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_29, torch.bfloat16);  div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_600: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_271, [1000, 4096]);  mul_271 = None
        permute_329: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        mm_208: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_600, permute_329);  view_600 = permute_329 = None
        view_601: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_208, [1, 1000, 14336]);  mm_208 = None
        mul_272: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_599, view_601);  convert_element_type_599 = view_601 = None
        view_602: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_272, [1000, 14336]);  mul_272 = None
        permute_330: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        mm_209: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_602, permute_330);  view_602 = permute_330 = None
        view_603: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_209, [1, 1000, 4096]);  mm_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_212: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_209, view_603);  add_209 = view_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_604: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_212, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_61: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_604, 2)
        mean_60: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_61, [-1], True);  pow_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_213: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_60, 1e-05);  mean_60 = None
        rsqrt_60: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        mul_273: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_604, rsqrt_60);  convert_element_type_604 = rsqrt_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_605: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_273, torch.bfloat16);  mul_273 = None
        mul_274: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg273_1, convert_element_type_605);  arg273_1 = convert_element_type_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_604: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_274, [1000, 4096])
        permute_331: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        mm_210: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_604, permute_331);  view_604 = permute_331 = None
        view_605: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_210, [1, 1000, 4096]);  mm_210 = None
        view_606: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_605, [1, 1000, -1, 128]);  view_605 = None
        permute_332: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_131: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_275: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_332, unsqueeze_131)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_122: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_332, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_90: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_122);  slice_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_121: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_332, 3, 0, 64);  permute_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_60: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_90, slice_121], -1);  neg_90 = slice_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_132: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_276: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_60, unsqueeze_132);  cat_60 = None
        add_214: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_607: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_274, [1000, 4096])
        permute_333: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        mm_211: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_607, permute_333);  view_607 = permute_333 = None
        view_608: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_211, [1, 1000, 1024]);  mm_211 = None
        view_609: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_608, [1, 1000, -1, 128]);  view_608 = None
        permute_334: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_609, [0, 2, 1, 3]);  view_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_277: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_334, unsqueeze_131);  unsqueeze_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_124: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_334, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_91: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_124);  slice_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_123: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_334, 3, 0, 64);  permute_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_61: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_91, slice_123], -1);  neg_91 = slice_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_278: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_61, unsqueeze_132);  cat_61 = unsqueeze_132 = None
        add_215: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_133: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_215, 2)
        expand_65: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_133, [1, 8, 4, 1000, 128]);  unsqueeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_124: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_613: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [1, 32, 1000, 128]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_610: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_274, [1000, 4096]);  mul_274 = None
        permute_335: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
        mm_212: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_610, permute_335);  view_610 = permute_335 = None
        view_611: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_212, [1, 1000, 1024]);  mm_212 = None
        view_612: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_611, [1, 1000, -1, 128]);  view_611 = None
        permute_336: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_134: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_336, 2)
        expand_66: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_134, [1, 8, 4, 1000, 128]);  unsqueeze_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_125: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_614: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_125, [1, 32, 1000, 128]);  clone_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_61: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_30: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_61, full_default_60);  full_default_61 = full_default_60 = None
        _scaled_dot_product_cudnn_attention_30 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_214, view_613, view_614, where_30, False, scale = 0.08838834764831845);  add_214 = view_613 = view_614 = where_30 = None
        getitem_270: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_30[0];  _scaled_dot_product_cudnn_attention_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_337: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_270, [0, 2, 1, 3]);  getitem_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_615: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_337, [1, 1000, -1]);  permute_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_616: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_615, [1000, 4096]);  view_615 = None
        permute_338: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        mm_213: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_616, permute_338);  view_616 = permute_338 = None
        view_617: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_213, [1, 1000, 4096]);  mm_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_216: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_212, view_617);  add_212 = view_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_614: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_216, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_62: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_614, 2)
        mean_61: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_62, [-1], True);  pow_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_217: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_61, 1e-05);  mean_61 = None
        rsqrt_61: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        mul_279: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_614, rsqrt_61);  convert_element_type_614 = rsqrt_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_615: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_279, torch.bfloat16);  mul_279 = None
        mul_280: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg278_1, convert_element_type_615);  arg278_1 = convert_element_type_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_618: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_280, [1000, 4096])
        permute_339: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        mm_214: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_618, permute_339);  view_618 = permute_339 = None
        view_619: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_214, [1, 1000, 14336]);  mm_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_618: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_619, torch.float32);  view_619 = None
        neg_92: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_618)
        exp_30: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_92);  neg_92 = None
        add_218: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        div_30: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_618, add_218);  convert_element_type_618 = add_218 = None
        convert_element_type_619: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_30, torch.bfloat16);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_620: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_280, [1000, 4096]);  mul_280 = None
        permute_340: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        mm_215: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_620, permute_340);  view_620 = permute_340 = None
        view_621: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_215, [1, 1000, 14336]);  mm_215 = None
        mul_281: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_619, view_621);  convert_element_type_619 = view_621 = None
        view_622: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_281, [1000, 14336]);  mul_281 = None
        permute_341: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        mm_216: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_622, permute_341);  view_622 = permute_341 = None
        view_623: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_216, [1, 1000, 4096]);  mm_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_219: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_216, view_623);  add_216 = view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_624: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_63: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_624, 2)
        mean_62: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_63, [-1], True);  pow_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_220: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_62, 1e-05);  mean_62 = None
        rsqrt_62: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        mul_282: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_624, rsqrt_62);  convert_element_type_624 = rsqrt_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_625: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_282, torch.bfloat16);  mul_282 = None
        mul_283: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg282_1, convert_element_type_625);  arg282_1 = convert_element_type_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_624: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_283, [1000, 4096])
        permute_342: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        mm_217: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_624, permute_342);  view_624 = permute_342 = None
        view_625: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_217, [1, 1000, 4096]);  mm_217 = None
        view_626: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_625, [1, 1000, -1, 128]);  view_625 = None
        permute_343: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_626, [0, 2, 1, 3]);  view_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_135: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_284: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_343, unsqueeze_135)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_126: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_343, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_93: "bf16[1, 32, 1000, 64][2048000, 64, 2048, 1]cuda:0" = torch.ops.aten.neg.default(slice_126);  slice_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_125: "bf16[1, 32, 1000, 64][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_343, 3, 0, 64);  permute_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_62: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_93, slice_125], -1);  neg_93 = slice_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_136: "bf16[1, 1, 1000, 128][128000, 128000, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_285: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_62, unsqueeze_136);  cat_62 = None
        add_221: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_627: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_283, [1000, 4096])
        permute_344: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        mm_218: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_627, permute_344);  view_627 = permute_344 = None
        view_628: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_218, [1, 1000, 1024]);  mm_218 = None
        view_629: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_628, [1, 1000, -1, 128]);  view_628 = None
        permute_345: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_629, [0, 2, 1, 3]);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_286: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_345, unsqueeze_135);  unsqueeze_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_128: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_345, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_94: "bf16[1, 8, 1000, 64][512000, 64, 512, 1]cuda:0" = torch.ops.aten.neg.default(slice_128);  slice_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_127: "bf16[1, 8, 1000, 64][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_345, 3, 0, 64);  permute_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_63: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.cat.default([neg_94, slice_127], -1);  neg_94 = slice_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_287: "bf16[1, 8, 1000, 128][1024000, 128000, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_63, unsqueeze_136);  cat_63 = unsqueeze_136 = None
        add_222: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_137: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_222, 2)
        expand_67: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_137, [1, 8, 4, 1000, 128]);  unsqueeze_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_128: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_633: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [1, 32, 1000, 128]);  clone_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_630: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_283, [1000, 4096]);  mul_283 = None
        permute_346: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        mm_219: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_630, permute_346);  view_630 = permute_346 = None
        view_631: "bf16[1, 1000, 1024][1024000, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_219, [1, 1000, 1024]);  mm_219 = None
        view_632: "bf16[1, 1000, 8, 128][1024000, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_631, [1, 1000, -1, 128]);  view_631 = None
        permute_347: "bf16[1, 8, 1000, 128][1024000, 128, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_632, [0, 2, 1, 3]);  view_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_138: "bf16[1, 8, 1, 1000, 128][1024000, 128, 1024000, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_347, 2)
        expand_68: "bf16[1, 8, 4, 1000, 128][1024000, 128, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_138, [1, 8, 4, 1000, 128]);  unsqueeze_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_129: "bf16[1, 8, 4, 1000, 128][4096000, 512000, 128000, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_634: "bf16[1, 32, 1000, 128][4096000, 128000, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [1, 32, 1000, 128]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_63: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_62: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31: "bf16[1, 1, 1000, 1000][1000000, 1000000, 1000, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_63, full_default_62);  expand = full_default_63 = full_default_62 = None
        _scaled_dot_product_cudnn_attention_31 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_221, view_633, view_634, where_31, False, scale = 0.08838834764831845);  add_221 = view_633 = view_634 = where_31 = None
        getitem_279: "bf16[1, 32, 1000, 128][4096000, 128, 4096, 1]cuda:0" = _scaled_dot_product_cudnn_attention_31[0];  _scaled_dot_product_cudnn_attention_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_348: "bf16[1, 1000, 32, 128][4096000, 4096, 128, 1]cuda:0" = torch.ops.aten.permute.default(getitem_279, [0, 2, 1, 3]);  getitem_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_635: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_348, [1, 1000, -1]);  permute_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_636: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_635, [1000, 4096]);  view_635 = None
        permute_349: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        mm_220: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_636, permute_349);  view_636 = permute_349 = None
        view_637: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_220, [1, 1000, 4096]);  mm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_223: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_219, view_637);  add_219 = view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_634: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_223, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_64: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_634, 2)
        mean_63: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_64, [-1], True);  pow_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_224: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_63, 1e-05);  mean_63 = None
        rsqrt_63: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        mul_288: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_634, rsqrt_63);  convert_element_type_634 = rsqrt_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_635: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_288, torch.bfloat16);  mul_288 = None
        mul_289: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg287_1, convert_element_type_635);  arg287_1 = convert_element_type_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_638: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_289, [1000, 4096])
        permute_350: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        mm_221: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_638, permute_350);  view_638 = permute_350 = None
        view_639: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_221, [1, 1000, 14336]);  mm_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_638: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_639, torch.float32);  view_639 = None
        neg_95: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_638)
        exp_31: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.exp.default(neg_95);  neg_95 = None
        add_225: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        div_31: "f32[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_638, add_225);  convert_element_type_638 = add_225 = None
        convert_element_type_639: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16);  div_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_640: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_289, [1000, 4096]);  mul_289 = None
        permute_351: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        mm_222: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.mm.default(view_640, permute_351);  view_640 = permute_351 = None
        view_641: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_222, [1, 1000, 14336]);  mm_222 = None
        mul_290: "bf16[1, 1000, 14336][14336000, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_639, view_641);  convert_element_type_639 = view_641 = None
        view_642: "bf16[1000, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_290, [1000, 14336]);  mul_290 = None
        permute_352: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        mm_223: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_642, permute_352);  view_642 = permute_352 = None
        view_643: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_223, [1, 1000, 4096]);  mm_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_226: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_223, view_643);  add_223 = view_643 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_644: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_226, torch.float32);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_65: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_644, 2)
        mean_64: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_65, [-1], True);  pow_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_227: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_64, 1e-05);  mean_64 = None
        rsqrt_64: "f32[1, 1000, 1][1000, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
        mul_291: "f32[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_644, rsqrt_64);  convert_element_type_644 = rsqrt_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_645: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_291, torch.bfloat16);  mul_291 = None
        mul_292: "bf16[1, 1000, 4096][4096000, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg291_1, convert_element_type_645);  arg291_1 = convert_element_type_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:460 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_644: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_292, [1000, 4096]);  mul_292 = None
        permute_353: "bf16[4096, 32768][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
        mm_224: "bf16[1000, 32768][32768, 1]cuda:0" = torch.ops.aten.mm.default(view_644, permute_353);  view_644 = permute_353 = None
        view_645: "bf16[1, 1000, 32768][32768000, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_224, [1, 1000, 32768]);  mm_224 = None
        return (permute_6, add_5, permute_17, add_12, permute_28, add_19, permute_39, add_26, permute_50, add_33, permute_61, add_40, permute_72, add_47, permute_83, add_54, permute_94, add_61, permute_105, add_68, permute_116, add_75, permute_127, add_82, permute_138, add_89, permute_149, add_96, permute_160, add_103, permute_171, add_110, permute_182, add_117, permute_193, add_124, permute_204, add_131, permute_215, add_138, permute_226, add_145, permute_237, add_152, permute_248, add_159, permute_259, add_166, permute_270, add_173, permute_281, add_180, permute_292, add_187, permute_303, add_194, permute_314, add_201, permute_325, add_208, permute_336, add_215, permute_347, add_222, view_645)
