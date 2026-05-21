class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 512]", arg1_1: "bf16[151936, 1024]", arg2_1: "f32[64]", arg3_1: "bf16[1024]", arg4_1: "bf16[2048, 1024]", arg5_1: "bf16[128]", arg6_1: "bf16[1024, 1024]", arg7_1: "bf16[128]", arg8_1: "bf16[1024, 1024]", arg9_1: "bf16[1024, 2048]", arg10_1: "bf16[1024]", arg11_1: "bf16[3072, 1024]", arg12_1: "bf16[3072, 1024]", arg13_1: "bf16[1024, 3072]", arg14_1: "bf16[1024]", arg15_1: "bf16[2048, 1024]", arg16_1: "bf16[128]", arg17_1: "bf16[1024, 1024]", arg18_1: "bf16[128]", arg19_1: "bf16[1024, 1024]", arg20_1: "bf16[1024, 2048]", arg21_1: "bf16[1024]", arg22_1: "bf16[3072, 1024]", arg23_1: "bf16[3072, 1024]", arg24_1: "bf16[1024, 3072]", arg25_1: "bf16[1024]", arg26_1: "bf16[2048, 1024]", arg27_1: "bf16[128]", arg28_1: "bf16[1024, 1024]", arg29_1: "bf16[128]", arg30_1: "bf16[1024, 1024]", arg31_1: "bf16[1024, 2048]", arg32_1: "bf16[1024]", arg33_1: "bf16[3072, 1024]", arg34_1: "bf16[3072, 1024]", arg35_1: "bf16[1024, 3072]", arg36_1: "bf16[1024]", arg37_1: "bf16[2048, 1024]", arg38_1: "bf16[128]", arg39_1: "bf16[1024, 1024]", arg40_1: "bf16[128]", arg41_1: "bf16[1024, 1024]", arg42_1: "bf16[1024, 2048]", arg43_1: "bf16[1024]", arg44_1: "bf16[3072, 1024]", arg45_1: "bf16[3072, 1024]", arg46_1: "bf16[1024, 3072]", arg47_1: "bf16[1024]", arg48_1: "bf16[2048, 1024]", arg49_1: "bf16[128]", arg50_1: "bf16[1024, 1024]", arg51_1: "bf16[128]", arg52_1: "bf16[1024, 1024]", arg53_1: "bf16[1024, 2048]", arg54_1: "bf16[1024]", arg55_1: "bf16[3072, 1024]", arg56_1: "bf16[3072, 1024]", arg57_1: "bf16[1024, 3072]", arg58_1: "bf16[1024]", arg59_1: "bf16[2048, 1024]", arg60_1: "bf16[128]", arg61_1: "bf16[1024, 1024]", arg62_1: "bf16[128]", arg63_1: "bf16[1024, 1024]", arg64_1: "bf16[1024, 2048]", arg65_1: "bf16[1024]", arg66_1: "bf16[3072, 1024]", arg67_1: "bf16[3072, 1024]", arg68_1: "bf16[1024, 3072]", arg69_1: "bf16[1024]", arg70_1: "bf16[2048, 1024]", arg71_1: "bf16[128]", arg72_1: "bf16[1024, 1024]", arg73_1: "bf16[128]", arg74_1: "bf16[1024, 1024]", arg75_1: "bf16[1024, 2048]", arg76_1: "bf16[1024]", arg77_1: "bf16[3072, 1024]", arg78_1: "bf16[3072, 1024]", arg79_1: "bf16[1024, 3072]", arg80_1: "bf16[1024]", arg81_1: "bf16[2048, 1024]", arg82_1: "bf16[128]", arg83_1: "bf16[1024, 1024]", arg84_1: "bf16[128]", arg85_1: "bf16[1024, 1024]", arg86_1: "bf16[1024, 2048]", arg87_1: "bf16[1024]", arg88_1: "bf16[3072, 1024]", arg89_1: "bf16[3072, 1024]", arg90_1: "bf16[1024, 3072]", arg91_1: "bf16[1024]", arg92_1: "bf16[2048, 1024]", arg93_1: "bf16[128]", arg94_1: "bf16[1024, 1024]", arg95_1: "bf16[128]", arg96_1: "bf16[1024, 1024]", arg97_1: "bf16[1024, 2048]", arg98_1: "bf16[1024]", arg99_1: "bf16[3072, 1024]", arg100_1: "bf16[3072, 1024]", arg101_1: "bf16[1024, 3072]", arg102_1: "bf16[1024]", arg103_1: "bf16[2048, 1024]", arg104_1: "bf16[128]", arg105_1: "bf16[1024, 1024]", arg106_1: "bf16[128]", arg107_1: "bf16[1024, 1024]", arg108_1: "bf16[1024, 2048]", arg109_1: "bf16[1024]", arg110_1: "bf16[3072, 1024]", arg111_1: "bf16[3072, 1024]", arg112_1: "bf16[1024, 3072]", arg113_1: "bf16[1024]", arg114_1: "bf16[2048, 1024]", arg115_1: "bf16[128]", arg116_1: "bf16[1024, 1024]", arg117_1: "bf16[128]", arg118_1: "bf16[1024, 1024]", arg119_1: "bf16[1024, 2048]", arg120_1: "bf16[1024]", arg121_1: "bf16[3072, 1024]", arg122_1: "bf16[3072, 1024]", arg123_1: "bf16[1024, 3072]", arg124_1: "bf16[1024]", arg125_1: "bf16[2048, 1024]", arg126_1: "bf16[128]", arg127_1: "bf16[1024, 1024]", arg128_1: "bf16[128]", arg129_1: "bf16[1024, 1024]", arg130_1: "bf16[1024, 2048]", arg131_1: "bf16[1024]", arg132_1: "bf16[3072, 1024]", arg133_1: "bf16[3072, 1024]", arg134_1: "bf16[1024, 3072]", arg135_1: "bf16[1024]", arg136_1: "bf16[2048, 1024]", arg137_1: "bf16[128]", arg138_1: "bf16[1024, 1024]", arg139_1: "bf16[128]", arg140_1: "bf16[1024, 1024]", arg141_1: "bf16[1024, 2048]", arg142_1: "bf16[1024]", arg143_1: "bf16[3072, 1024]", arg144_1: "bf16[3072, 1024]", arg145_1: "bf16[1024, 3072]", arg146_1: "bf16[1024]", arg147_1: "bf16[2048, 1024]", arg148_1: "bf16[128]", arg149_1: "bf16[1024, 1024]", arg150_1: "bf16[128]", arg151_1: "bf16[1024, 1024]", arg152_1: "bf16[1024, 2048]", arg153_1: "bf16[1024]", arg154_1: "bf16[3072, 1024]", arg155_1: "bf16[3072, 1024]", arg156_1: "bf16[1024, 3072]", arg157_1: "bf16[1024]", arg158_1: "bf16[2048, 1024]", arg159_1: "bf16[128]", arg160_1: "bf16[1024, 1024]", arg161_1: "bf16[128]", arg162_1: "bf16[1024, 1024]", arg163_1: "bf16[1024, 2048]", arg164_1: "bf16[1024]", arg165_1: "bf16[3072, 1024]", arg166_1: "bf16[3072, 1024]", arg167_1: "bf16[1024, 3072]", arg168_1: "bf16[1024]", arg169_1: "bf16[2048, 1024]", arg170_1: "bf16[128]", arg171_1: "bf16[1024, 1024]", arg172_1: "bf16[128]", arg173_1: "bf16[1024, 1024]", arg174_1: "bf16[1024, 2048]", arg175_1: "bf16[1024]", arg176_1: "bf16[3072, 1024]", arg177_1: "bf16[3072, 1024]", arg178_1: "bf16[1024, 3072]", arg179_1: "bf16[1024]", arg180_1: "bf16[2048, 1024]", arg181_1: "bf16[128]", arg182_1: "bf16[1024, 1024]", arg183_1: "bf16[128]", arg184_1: "bf16[1024, 1024]", arg185_1: "bf16[1024, 2048]", arg186_1: "bf16[1024]", arg187_1: "bf16[3072, 1024]", arg188_1: "bf16[3072, 1024]", arg189_1: "bf16[1024, 3072]", arg190_1: "bf16[1024]", arg191_1: "bf16[2048, 1024]", arg192_1: "bf16[128]", arg193_1: "bf16[1024, 1024]", arg194_1: "bf16[128]", arg195_1: "bf16[1024, 1024]", arg196_1: "bf16[1024, 2048]", arg197_1: "bf16[1024]", arg198_1: "bf16[3072, 1024]", arg199_1: "bf16[3072, 1024]", arg200_1: "bf16[1024, 3072]", arg201_1: "bf16[1024]", arg202_1: "bf16[2048, 1024]", arg203_1: "bf16[128]", arg204_1: "bf16[1024, 1024]", arg205_1: "bf16[128]", arg206_1: "bf16[1024, 1024]", arg207_1: "bf16[1024, 2048]", arg208_1: "bf16[1024]", arg209_1: "bf16[3072, 1024]", arg210_1: "bf16[3072, 1024]", arg211_1: "bf16[1024, 3072]", arg212_1: "bf16[1024]", arg213_1: "bf16[2048, 1024]", arg214_1: "bf16[128]", arg215_1: "bf16[1024, 1024]", arg216_1: "bf16[128]", arg217_1: "bf16[1024, 1024]", arg218_1: "bf16[1024, 2048]", arg219_1: "bf16[1024]", arg220_1: "bf16[3072, 1024]", arg221_1: "bf16[3072, 1024]", arg222_1: "bf16[1024, 3072]", arg223_1: "bf16[1024]", arg224_1: "bf16[2048, 1024]", arg225_1: "bf16[128]", arg226_1: "bf16[1024, 1024]", arg227_1: "bf16[128]", arg228_1: "bf16[1024, 1024]", arg229_1: "bf16[1024, 2048]", arg230_1: "bf16[1024]", arg231_1: "bf16[3072, 1024]", arg232_1: "bf16[3072, 1024]", arg233_1: "bf16[1024, 3072]", arg234_1: "bf16[1024]", arg235_1: "bf16[2048, 1024]", arg236_1: "bf16[128]", arg237_1: "bf16[1024, 1024]", arg238_1: "bf16[128]", arg239_1: "bf16[1024, 1024]", arg240_1: "bf16[1024, 2048]", arg241_1: "bf16[1024]", arg242_1: "bf16[3072, 1024]", arg243_1: "bf16[3072, 1024]", arg244_1: "bf16[1024, 3072]", arg245_1: "bf16[1024]", arg246_1: "bf16[2048, 1024]", arg247_1: "bf16[128]", arg248_1: "bf16[1024, 1024]", arg249_1: "bf16[128]", arg250_1: "bf16[1024, 1024]", arg251_1: "bf16[1024, 2048]", arg252_1: "bf16[1024]", arg253_1: "bf16[3072, 1024]", arg254_1: "bf16[3072, 1024]", arg255_1: "bf16[1024, 3072]", arg256_1: "bf16[1024]", arg257_1: "bf16[2048, 1024]", arg258_1: "bf16[128]", arg259_1: "bf16[1024, 1024]", arg260_1: "bf16[128]", arg261_1: "bf16[1024, 1024]", arg262_1: "bf16[1024, 2048]", arg263_1: "bf16[1024]", arg264_1: "bf16[3072, 1024]", arg265_1: "bf16[3072, 1024]", arg266_1: "bf16[1024, 3072]", arg267_1: "bf16[1024]", arg268_1: "bf16[2048, 1024]", arg269_1: "bf16[128]", arg270_1: "bf16[1024, 1024]", arg271_1: "bf16[128]", arg272_1: "bf16[1024, 1024]", arg273_1: "bf16[1024, 2048]", arg274_1: "bf16[1024]", arg275_1: "bf16[3072, 1024]", arg276_1: "bf16[3072, 1024]", arg277_1: "bf16[1024, 3072]", arg278_1: "bf16[1024]", arg279_1: "bf16[2048, 1024]", arg280_1: "bf16[128]", arg281_1: "bf16[1024, 1024]", arg282_1: "bf16[128]", arg283_1: "bf16[1024, 1024]", arg284_1: "bf16[1024, 2048]", arg285_1: "bf16[1024]", arg286_1: "bf16[3072, 1024]", arg287_1: "bf16[3072, 1024]", arg288_1: "bf16[1024, 3072]", arg289_1: "bf16[1024]", arg290_1: "bf16[2048, 1024]", arg291_1: "bf16[128]", arg292_1: "bf16[1024, 1024]", arg293_1: "bf16[128]", arg294_1: "bf16[1024, 1024]", arg295_1: "bf16[1024, 2048]", arg296_1: "bf16[1024]", arg297_1: "bf16[3072, 1024]", arg298_1: "bf16[3072, 1024]", arg299_1: "bf16[1024, 3072]", arg300_1: "bf16[1024]", arg301_1: "bf16[2048, 1024]", arg302_1: "bf16[128]", arg303_1: "bf16[1024, 1024]", arg304_1: "bf16[128]", arg305_1: "bf16[1024, 1024]", arg306_1: "bf16[1024, 2048]", arg307_1: "bf16[1024]", arg308_1: "bf16[3072, 1024]", arg309_1: "bf16[3072, 1024]", arg310_1: "bf16[1024, 3072]", arg311_1: "bf16[1024]", arg312_1: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:392 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "bf16[4, 512, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_3: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 2)
        mean: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_3: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_3: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt);  convert_element_type_3 = rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_4: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_4: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg3_1, convert_element_type_4);  arg3_1 = convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_4: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_4, [2048, 1024])
        permute_1: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
        view_5: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm, [4, 512, 2048]);  mm = None
        view_6: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_5, [4, 512, -1, 128]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_7: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_2: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_7, 2)
        mean_1: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_4: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_5: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_7, rsqrt_1);  convert_element_type_7 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_8: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        mul_6: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg5_1, convert_element_type_8);  arg5_1 = convert_element_type_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_2: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_6, [0, 2, 1, 3]);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:138 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        expand_2: "f32[1, 64, 1]" = torch.ops.aten.expand.default(unsqueeze_11, [1, -1, 1]);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:143 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_3: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_2, [1, 64, 1]);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:399 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:400 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:139 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_12: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        convert_element_type: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_12, torch.float32);  unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:143 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_4: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type, [1, 1, 512]);  convert_element_type = None
        mul: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_3, expand_4);  expand_3 = expand_4 = None
        permute: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:144 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_13: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_5: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_13, [1, 512, 2, 64]);  unsqueeze_13 = None
        clone: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_3: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(clone, [1, 512, 128]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:145 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 512, 128]" = torch.ops.aten.cos.default(view_3)
        mul_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:148 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_1: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_14: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_9: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_2, unsqueeze_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_5: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_2, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_5);  slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_4: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_2, 3, 0, 64);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_1: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg, slice_4], -1);  neg = slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:146 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 512, 128]" = torch.ops.aten.sin.default(view_3);  view_3 = None
        mul_2: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:148 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_15: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_10: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_15);  cat_1 = None
        add_6: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_7: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_4, [2048, 1024])
        permute_3: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_1: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_7, permute_3);  view_7 = permute_3 = None
        view_8: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_1, [4, 512, 1024]);  mm_1 = None
        view_9: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_8, [4, 512, -1, 128]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_11: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_3: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_11, 2)
        mean_2: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_5: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_7: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_11, rsqrt_2);  convert_element_type_11 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_12: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        mul_8: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg7_1, convert_element_type_12);  arg7_1 = convert_element_type_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_4: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_8, [0, 2, 1, 3]);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_11: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_14);  unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_7: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_4, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_7);  slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_6: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_4, 3, 0, 64);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_2: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_1, slice_6], -1);  neg_1 = slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_12: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_15);  cat_2 = unsqueeze_15 = None
        add_7: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_16: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_7, 2);  add_7 = None
        expand_6: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_16, [4, 8, 2, 512, 128]);  unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_2: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_13: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_2, [4, 16, 512, 128]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_10: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_4, [2048, 1024]);  mul_4 = None
        permute_5: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        mm_2: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
        view_11: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 1024]);  mm_2 = None
        view_12: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_11, [4, 512, -1, 128]);  view_11 = None
        permute_6: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_17: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_6, 2);  permute_6 = None
        expand_7: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_17, [4, 8, 2, 512, 128]);  unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_3: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_14: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_3, [4, 16, 512, 128]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[512]" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_7: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_8: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        unsqueeze_9: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[512]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_4: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_5: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand: "i64[4, 512]" = torch.ops.aten.expand.default(unsqueeze, [4, -1]);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[4, 1]" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub: "i64[4, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[4, 513]" = torch.ops.aten.cat.default([sub, expand], -1);  sub = expand = None
        slice_3: "i64[4, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 513)
        slice_2: "i64[4, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 512);  cat = None
        sub_1: "i64[4, 512]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[4, 512]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[4, 512]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[4, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[4, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[4, 1, 512, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[4, 1, 1, 512]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[4, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[4, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_1, [4, -1, 512, 512]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_6, view_13, view_14, where, False, scale = 0.08838834764831845);  add_6 = view_13 = view_14 = where = None
        getitem: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_7, [4, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_16: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_15, [2048, 2048]);  view_15 = None
        permute_8: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_3: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
        view_17: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 1024]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_8: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(embedding, view_17);  embedding = view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_17: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_4: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_17, 2)
        mean_3: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_9: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_13: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_17, rsqrt_3);  convert_element_type_17 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_18: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        mul_14: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg10_1, convert_element_type_18);  arg10_1 = convert_element_type_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_18: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_14, [2048, 1024])
        permute_9: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_4: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_18, permute_9);  view_18 = permute_9 = None
        view_19: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_4, [4, 512, 3072]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_21: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        neg_2: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_21)
        exp: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_10: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_21, add_10);  convert_element_type_21 = add_10 = None
        convert_element_type_22: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_20: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_14, [2048, 1024]);  mul_14 = None
        permute_10: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        mm_5: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_20, permute_10);  view_20 = permute_10 = None
        view_21: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_5, [4, 512, 3072]);  mm_5 = None
        mul_15: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, view_21);  convert_element_type_22 = view_21 = None
        view_22: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_15, [2048, 3072]);  mul_15 = None
        permute_11: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_6: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
        view_23: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_6, [4, 512, 1024]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_11: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_8, view_23);  add_8 = view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_27: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_5: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_27, 2)
        mean_4: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_12: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_16: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_27, rsqrt_4);  convert_element_type_27 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_28: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        mul_17: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg14_1, convert_element_type_28);  arg14_1 = convert_element_type_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_24: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_17, [2048, 1024])
        permute_12: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_7: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_24, permute_12);  view_24 = permute_12 = None
        view_25: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_7, [4, 512, 2048]);  mm_7 = None
        view_26: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_25, [4, 512, -1, 128]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_31: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_26, torch.float32);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_6: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_31, 2)
        mean_5: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_13: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_18: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_31, rsqrt_5);  convert_element_type_31 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_32: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        mul_19: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg16_1, convert_element_type_32);  arg16_1 = convert_element_type_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_13: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_19, [0, 2, 1, 3]);  mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_18: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_22: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_13, unsqueeze_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_9: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_13, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_3: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_9);  slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_8: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_13, 3, 0, 64);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_3: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_3, slice_8], -1);  neg_3 = slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_19: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_23: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_19);  cat_3 = None
        add_15: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_27: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_17, [2048, 1024])
        permute_14: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        mm_8: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_27, permute_14);  view_27 = permute_14 = None
        view_28: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_8, [4, 512, 1024]);  mm_8 = None
        view_29: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_28, [4, 512, -1, 128]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_35: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_7: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_35, 2)
        mean_6: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_14: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_20: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_35, rsqrt_6);  convert_element_type_35 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_36: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        mul_21: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg18_1, convert_element_type_36);  arg18_1 = convert_element_type_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_15: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_21, [0, 2, 1, 3]);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_24: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_15, unsqueeze_18);  unsqueeze_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_11: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_15, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_4: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_11);  slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_10: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_15, 3, 0, 64);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_4: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_4, slice_10], -1);  neg_4 = slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_25: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_19);  cat_4 = unsqueeze_19 = None
        add_16: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_20: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_16, 2);  add_16 = None
        expand_8: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_20, [4, 8, 2, 512, 128]);  unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_4: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_33: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_4, [4, 16, 512, 128]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_30: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_17, [2048, 1024]);  mul_17 = None
        permute_16: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_9: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_30, permute_16);  view_30 = permute_16 = None
        view_31: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_9, [4, 512, 1024]);  mm_9 = None
        view_32: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_31, [4, 512, -1, 128]);  view_31 = None
        permute_17: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_21: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_17, 2);  permute_17 = None
        expand_9: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_21, [4, 8, 2, 512, 128]);  unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_5: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_34: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_5, [4, 16, 512, 128]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_4: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_15, view_33, view_34, where_1, False, scale = 0.08838834764831845);  add_15 = view_33 = view_34 = where_1 = None
        getitem_9: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_35: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_18, [4, 512, -1]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_36: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_35, [2048, 2048]);  view_35 = None
        permute_19: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_10: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_36, permute_19);  view_36 = permute_19 = None
        view_37: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_10, [4, 512, 1024]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_17: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_11, view_37);  add_11 = view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_41: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_8: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_41, 2)
        mean_7: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_18: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_26: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_41, rsqrt_7);  convert_element_type_41 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_42: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None
        mul_27: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg21_1, convert_element_type_42);  arg21_1 = convert_element_type_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_38: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_27, [2048, 1024])
        permute_20: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_11: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_38, permute_20);  view_38 = permute_20 = None
        view_39: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_11, [4, 512, 3072]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_45: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        neg_5: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_45)
        exp_1: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_19: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_45, add_19);  convert_element_type_45 = add_19 = None
        convert_element_type_46: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_40: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_27, [2048, 1024]);  mul_27 = None
        permute_21: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_12: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_40, permute_21);  view_40 = permute_21 = None
        view_41: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_12, [4, 512, 3072]);  mm_12 = None
        mul_28: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_46, view_41);  convert_element_type_46 = view_41 = None
        view_42: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_28, [2048, 3072]);  mul_28 = None
        permute_22: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_13: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_42, permute_22);  view_42 = permute_22 = None
        view_43: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_13, [4, 512, 1024]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_20: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_17, view_43);  add_17 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_51: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_9: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_51, 2)
        mean_8: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_21: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_29: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_51, rsqrt_8);  convert_element_type_51 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_52: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        mul_30: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg25_1, convert_element_type_52);  arg25_1 = convert_element_type_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_44: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_30, [2048, 1024])
        permute_23: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        mm_14: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_44, permute_23);  view_44 = permute_23 = None
        view_45: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_14, [4, 512, 2048]);  mm_14 = None
        view_46: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_45, [4, 512, -1, 128]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_55: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_10: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_55, 2)
        mean_9: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_22: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_31: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_55, rsqrt_9);  convert_element_type_55 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_56: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        mul_32: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg27_1, convert_element_type_56);  arg27_1 = convert_element_type_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_24: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_32, [0, 2, 1, 3]);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_22: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_35: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_24, unsqueeze_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_13: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_24, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_6: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_13);  slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_12: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_24, 3, 0, 64);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_5: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_6, slice_12], -1);  neg_6 = slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_23: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_36: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_23);  cat_5 = None
        add_24: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_47: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_30, [2048, 1024])
        permute_25: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_15: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_47, permute_25);  view_47 = permute_25 = None
        view_48: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_15, [4, 512, 1024]);  mm_15 = None
        view_49: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_48, [4, 512, -1, 128]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_59: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_11: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_59, 2)
        mean_10: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_23: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_33: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_59, rsqrt_10);  convert_element_type_59 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_60: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None
        mul_34: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg29_1, convert_element_type_60);  arg29_1 = convert_element_type_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_26: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_34, [0, 2, 1, 3]);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_37: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_26, unsqueeze_22);  unsqueeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_15: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_26, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_7: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_15);  slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_14: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_26, 3, 0, 64);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_6: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_7, slice_14], -1);  neg_7 = slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_38: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_23);  cat_6 = unsqueeze_23 = None
        add_25: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_24: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_25, 2);  add_25 = None
        expand_10: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_24, [4, 8, 2, 512, 128]);  unsqueeze_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_6: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_53: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_6, [4, 16, 512, 128]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_50: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_30, [2048, 1024]);  mul_30 = None
        permute_27: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_16: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_50, permute_27);  view_50 = permute_27 = None
        view_51: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_16, [4, 512, 1024]);  mm_16 = None
        view_52: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_51, [4, 512, -1, 128]);  view_51 = None
        permute_28: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_25: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_28, 2);  permute_28 = None
        expand_11: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_25, [4, 8, 2, 512, 128]);  unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_7: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_54: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_7, [4, 16, 512, 128]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_6: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_24, view_53, view_54, where_2, False, scale = 0.08838834764831845);  add_24 = view_53 = view_54 = where_2 = None
        getitem_18: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_55: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_29, [4, 512, -1]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_56: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_55, [2048, 2048]);  view_55 = None
        permute_30: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_17: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_56, permute_30);  view_56 = permute_30 = None
        view_57: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_17, [4, 512, 1024]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_26: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_20, view_57);  add_20 = view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_65: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_12: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_65, 2)
        mean_11: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_27: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_39: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_65, rsqrt_11);  convert_element_type_65 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_66: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        mul_40: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg32_1, convert_element_type_66);  arg32_1 = convert_element_type_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_58: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_40, [2048, 1024])
        permute_31: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_18: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_58, permute_31);  view_58 = permute_31 = None
        view_59: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_18, [4, 512, 3072]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_69: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        neg_8: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_69)
        exp_2: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_28: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_69, add_28);  convert_element_type_69 = add_28 = None
        convert_element_type_70: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_60: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_40, [2048, 1024]);  mul_40 = None
        permute_32: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_19: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_60, permute_32);  view_60 = permute_32 = None
        view_61: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_19, [4, 512, 3072]);  mm_19 = None
        mul_41: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_70, view_61);  convert_element_type_70 = view_61 = None
        view_62: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_41, [2048, 3072]);  mul_41 = None
        permute_33: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        mm_20: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_62, permute_33);  view_62 = permute_33 = None
        view_63: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_20, [4, 512, 1024]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_29: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_26, view_63);  add_26 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_75: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_13: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_75, 2)
        mean_12: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_30: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_42: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_75, rsqrt_12);  convert_element_type_75 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_76: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None
        mul_43: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg36_1, convert_element_type_76);  arg36_1 = convert_element_type_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_64: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_43, [2048, 1024])
        permute_34: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_21: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_64, permute_34);  view_64 = permute_34 = None
        view_65: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_21, [4, 512, 2048]);  mm_21 = None
        view_66: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_65, [4, 512, -1, 128]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_79: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_66, torch.float32);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_14: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_79, 2)
        mean_13: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_31: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_44: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_79, rsqrt_13);  convert_element_type_79 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_80: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16);  mul_44 = None
        mul_45: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg38_1, convert_element_type_80);  arg38_1 = convert_element_type_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_35: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_45, [0, 2, 1, 3]);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_26: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_48: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_35, unsqueeze_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_17: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_35, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_9: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_17);  slice_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_16: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_35, 3, 0, 64);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_7: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_9, slice_16], -1);  neg_9 = slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_27: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_49: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_27);  cat_7 = None
        add_33: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_67: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_43, [2048, 1024])
        permute_36: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_22: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_67, permute_36);  view_67 = permute_36 = None
        view_68: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_22, [4, 512, 1024]);  mm_22 = None
        view_69: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_68, [4, 512, -1, 128]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_83: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_15: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_83, 2)
        mean_14: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_32: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_46: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_83, rsqrt_14);  convert_element_type_83 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_84: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None
        mul_47: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg40_1, convert_element_type_84);  arg40_1 = convert_element_type_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_37: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_47, [0, 2, 1, 3]);  mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_50: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_37, unsqueeze_26);  unsqueeze_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_19: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_37, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_10: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_19);  slice_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_18: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_37, 3, 0, 64);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_8: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_10, slice_18], -1);  neg_10 = slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_51: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_27);  cat_8 = unsqueeze_27 = None
        add_34: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_28: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_34, 2);  add_34 = None
        expand_12: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_28, [4, 8, 2, 512, 128]);  unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_8: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_73: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_8, [4, 16, 512, 128]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_70: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_43, [2048, 1024]);  mul_43 = None
        permute_38: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_23: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_70, permute_38);  view_70 = permute_38 = None
        view_71: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_23, [4, 512, 1024]);  mm_23 = None
        view_72: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_71, [4, 512, -1, 128]);  view_71 = None
        permute_39: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_29: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_39, 2);  permute_39 = None
        expand_13: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_29, [4, 8, 2, 512, 128]);  unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_9: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_74: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_9, [4, 16, 512, 128]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_8: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_33, view_73, view_74, where_3, False, scale = 0.08838834764831845);  add_33 = view_73 = view_74 = where_3 = None
        getitem_27: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_75: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_40, [4, 512, -1]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_76: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_75, [2048, 2048]);  view_75 = None
        permute_41: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_24: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_76, permute_41);  view_76 = permute_41 = None
        view_77: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_24, [4, 512, 1024]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_35: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_29, view_77);  add_29 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_89: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_16: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_89, 2)
        mean_15: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_36: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_52: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_89, rsqrt_15);  convert_element_type_89 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_90: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None
        mul_53: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg43_1, convert_element_type_90);  arg43_1 = convert_element_type_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_78: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_53, [2048, 1024])
        permute_42: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        mm_25: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_78, permute_42);  view_78 = permute_42 = None
        view_79: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_25, [4, 512, 3072]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_93: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        neg_11: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_93)
        exp_3: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_37: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_93, add_37);  convert_element_type_93 = add_37 = None
        convert_element_type_94: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_80: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_53, [2048, 1024]);  mul_53 = None
        permute_43: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_26: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_80, permute_43);  view_80 = permute_43 = None
        view_81: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_26, [4, 512, 3072]);  mm_26 = None
        mul_54: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_94, view_81);  convert_element_type_94 = view_81 = None
        view_82: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_54, [2048, 3072]);  mul_54 = None
        permute_44: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_27: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_82, permute_44);  view_82 = permute_44 = None
        view_83: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_27, [4, 512, 1024]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_38: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_35, view_83);  add_35 = view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_99: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_17: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_99, 2)
        mean_16: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_39: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_55: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_99, rsqrt_16);  convert_element_type_99 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_100: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        mul_56: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg47_1, convert_element_type_100);  arg47_1 = convert_element_type_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_84: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_56, [2048, 1024])
        permute_45: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        mm_28: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_84, permute_45);  view_84 = permute_45 = None
        view_85: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_28, [4, 512, 2048]);  mm_28 = None
        view_86: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_85, [4, 512, -1, 128]);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_103: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_86, torch.float32);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_18: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_103, 2)
        mean_17: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_40: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_57: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_103, rsqrt_17);  convert_element_type_103 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_104: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        mul_58: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg49_1, convert_element_type_104);  arg49_1 = convert_element_type_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_46: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_58, [0, 2, 1, 3]);  mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_30: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_61: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_46, unsqueeze_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_21: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_46, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_12: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_21);  slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_20: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_46, 3, 0, 64);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_9: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_12, slice_20], -1);  neg_12 = slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_31: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_62: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_31);  cat_9 = None
        add_42: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_87: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_56, [2048, 1024])
        permute_47: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_29: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_87, permute_47);  view_87 = permute_47 = None
        view_88: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_29, [4, 512, 1024]);  mm_29 = None
        view_89: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_88, [4, 512, -1, 128]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_107: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_89, torch.float32);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_19: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_107, 2)
        mean_18: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_41: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        mul_59: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_107, rsqrt_18);  convert_element_type_107 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_108: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        mul_60: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg51_1, convert_element_type_108);  arg51_1 = convert_element_type_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_48: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_60, [0, 2, 1, 3]);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_63: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_48, unsqueeze_30);  unsqueeze_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_23: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_48, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_13: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_23);  slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_22: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_48, 3, 0, 64);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_10: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_13, slice_22], -1);  neg_13 = slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_64: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_31);  cat_10 = unsqueeze_31 = None
        add_43: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_32: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_43, 2);  add_43 = None
        expand_14: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_32, [4, 8, 2, 512, 128]);  unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_10: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_93: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_10, [4, 16, 512, 128]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_90: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_56, [2048, 1024]);  mul_56 = None
        permute_49: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_30: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_90, permute_49);  view_90 = permute_49 = None
        view_91: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_30, [4, 512, 1024]);  mm_30 = None
        view_92: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_91, [4, 512, -1, 128]);  view_91 = None
        permute_50: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_33: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_50, 2);  permute_50 = None
        expand_15: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_33, [4, 8, 2, 512, 128]);  unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_11: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_94: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_11, [4, 16, 512, 128]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_10: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_42, view_93, view_94, where_4, False, scale = 0.08838834764831845);  add_42 = view_93 = view_94 = where_4 = None
        getitem_36: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_95: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_51, [4, 512, -1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_96: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_95, [2048, 2048]);  view_95 = None
        permute_52: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_31: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_96, permute_52);  view_96 = permute_52 = None
        view_97: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_31, [4, 512, 1024]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_44: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_38, view_97);  add_38 = view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_113: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_20: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_113, 2)
        mean_19: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_45: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_65: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_113, rsqrt_19);  convert_element_type_113 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_114: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        mul_66: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg54_1, convert_element_type_114);  arg54_1 = convert_element_type_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_98: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_66, [2048, 1024])
        permute_53: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_32: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_98, permute_53);  view_98 = permute_53 = None
        view_99: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_32, [4, 512, 3072]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_117: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        neg_14: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_117)
        exp_4: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_46: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_117, add_46);  convert_element_type_117 = add_46 = None
        convert_element_type_118: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_100: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_66, [2048, 1024]);  mul_66 = None
        permute_54: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_33: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_100, permute_54);  view_100 = permute_54 = None
        view_101: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_33, [4, 512, 3072]);  mm_33 = None
        mul_67: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_118, view_101);  convert_element_type_118 = view_101 = None
        view_102: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_67, [2048, 3072]);  mul_67 = None
        permute_55: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        mm_34: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_102, permute_55);  view_102 = permute_55 = None
        view_103: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_34, [4, 512, 1024]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_47: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_44, view_103);  add_44 = view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_123: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_47, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_21: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_123, 2)
        mean_20: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_48: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_68: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_123, rsqrt_20);  convert_element_type_123 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_124: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_68, torch.bfloat16);  mul_68 = None
        mul_69: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg58_1, convert_element_type_124);  arg58_1 = convert_element_type_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_104: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_69, [2048, 1024])
        permute_56: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_35: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_104, permute_56);  view_104 = permute_56 = None
        view_105: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_35, [4, 512, 2048]);  mm_35 = None
        view_106: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_105, [4, 512, -1, 128]);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_127: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_106, torch.float32);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_22: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_127, 2)
        mean_21: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_70: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_127, rsqrt_21);  convert_element_type_127 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_128: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None
        mul_71: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg60_1, convert_element_type_128);  arg60_1 = convert_element_type_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_57: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_71, [0, 2, 1, 3]);  mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_34: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_74: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_57, unsqueeze_34)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_25: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_57, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_15: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_25);  slice_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_24: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_57, 3, 0, 64);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_11: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_15, slice_24], -1);  neg_15 = slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_35: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_75: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_11, unsqueeze_35);  cat_11 = None
        add_51: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_107: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_69, [2048, 1024])
        permute_58: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_36: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_107, permute_58);  view_107 = permute_58 = None
        view_108: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_36, [4, 512, 1024]);  mm_36 = None
        view_109: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_108, [4, 512, -1, 128]);  view_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_131: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_23: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_131, 2)
        mean_22: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_50: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_72: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_131, rsqrt_22);  convert_element_type_131 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_132: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        mul_73: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg62_1, convert_element_type_132);  arg62_1 = convert_element_type_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_59: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_73, [0, 2, 1, 3]);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_76: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_59, unsqueeze_34);  unsqueeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_27: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_59, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_16: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_27);  slice_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_26: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_59, 3, 0, 64);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_12: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_16, slice_26], -1);  neg_16 = slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_77: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_12, unsqueeze_35);  cat_12 = unsqueeze_35 = None
        add_52: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_76, mul_77);  mul_76 = mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_36: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_52, 2);  add_52 = None
        expand_16: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_36, [4, 8, 2, 512, 128]);  unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_12: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_113: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_12, [4, 16, 512, 128]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_110: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_69, [2048, 1024]);  mul_69 = None
        permute_60: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_37: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_110, permute_60);  view_110 = permute_60 = None
        view_111: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_37, [4, 512, 1024]);  mm_37 = None
        view_112: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_111, [4, 512, -1, 128]);  view_111 = None
        permute_61: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_37: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_61, 2);  permute_61 = None
        expand_17: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_37, [4, 8, 2, 512, 128]);  unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_13: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_114: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_13, [4, 16, 512, 128]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_12: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_51, view_113, view_114, where_5, False, scale = 0.08838834764831845);  add_51 = view_113 = view_114 = where_5 = None
        getitem_45: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_115: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_62, [4, 512, -1]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_116: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_115, [2048, 2048]);  view_115 = None
        permute_63: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_38: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_116, permute_63);  view_116 = permute_63 = None
        view_117: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_38, [4, 512, 1024]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_53: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_47, view_117);  add_47 = view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_137: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_53, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_24: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_137, 2)
        mean_23: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_54: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_78: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_137, rsqrt_23);  convert_element_type_137 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_138: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        mul_79: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg65_1, convert_element_type_138);  arg65_1 = convert_element_type_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_118: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_79, [2048, 1024])
        permute_64: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        mm_39: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_118, permute_64);  view_118 = permute_64 = None
        view_119: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_39, [4, 512, 3072]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_141: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        neg_17: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_141)
        exp_5: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_55: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_141, add_55);  convert_element_type_141 = add_55 = None
        convert_element_type_142: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_120: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_79, [2048, 1024]);  mul_79 = None
        permute_65: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_40: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_120, permute_65);  view_120 = permute_65 = None
        view_121: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_40, [4, 512, 3072]);  mm_40 = None
        mul_80: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_142, view_121);  convert_element_type_142 = view_121 = None
        view_122: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_80, [2048, 3072]);  mul_80 = None
        permute_66: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_41: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_122, permute_66);  view_122 = permute_66 = None
        view_123: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_41, [4, 512, 1024]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_56: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_53, view_123);  add_53 = view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_147: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_56, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_25: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_147, 2)
        mean_24: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_57: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_81: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_147, rsqrt_24);  convert_element_type_147 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_148: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None
        mul_82: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg69_1, convert_element_type_148);  arg69_1 = convert_element_type_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_124: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_82, [2048, 1024])
        permute_67: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_42: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_124, permute_67);  view_124 = permute_67 = None
        view_125: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_42, [4, 512, 2048]);  mm_42 = None
        view_126: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_125, [4, 512, -1, 128]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_151: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_126, torch.float32);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_26: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_151, 2)
        mean_25: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_58: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_83: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_151, rsqrt_25);  convert_element_type_151 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_152: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None
        mul_84: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg71_1, convert_element_type_152);  arg71_1 = convert_element_type_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_68: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_84, [0, 2, 1, 3]);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_38: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_87: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_68, unsqueeze_38)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_29: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_68, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_18: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_29);  slice_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_28: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_68, 3, 0, 64);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_13: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_18, slice_28], -1);  neg_18 = slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_39: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_88: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_39);  cat_13 = None
        add_60: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_127: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_82, [2048, 1024])
        permute_69: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_43: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_127, permute_69);  view_127 = permute_69 = None
        view_128: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_43, [4, 512, 1024]);  mm_43 = None
        view_129: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_128, [4, 512, -1, 128]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_155: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_27: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_155, 2)
        mean_26: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_59: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_85: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_155, rsqrt_26);  convert_element_type_155 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_156: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        mul_86: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg73_1, convert_element_type_156);  arg73_1 = convert_element_type_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_70: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_86, [0, 2, 1, 3]);  mul_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_89: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_70, unsqueeze_38);  unsqueeze_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_31: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_70, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_19: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_31);  slice_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_30: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_70, 3, 0, 64);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_14: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_19, slice_30], -1);  neg_19 = slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_90: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_14, unsqueeze_39);  cat_14 = unsqueeze_39 = None
        add_61: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_40: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_61, 2);  add_61 = None
        expand_18: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_40, [4, 8, 2, 512, 128]);  unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_14: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_133: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_14, [4, 16, 512, 128]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_130: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_82, [2048, 1024]);  mul_82 = None
        permute_71: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_44: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_130, permute_71);  view_130 = permute_71 = None
        view_131: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_44, [4, 512, 1024]);  mm_44 = None
        view_132: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_131, [4, 512, -1, 128]);  view_131 = None
        permute_72: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_41: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_72, 2);  permute_72 = None
        expand_19: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_41, [4, 8, 2, 512, 128]);  unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_15: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_134: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_15, [4, 16, 512, 128]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_14: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_60, view_133, view_134, where_6, False, scale = 0.08838834764831845);  add_60 = view_133 = view_134 = where_6 = None
        getitem_54: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_135: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_73, [4, 512, -1]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_136: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_135, [2048, 2048]);  view_135 = None
        permute_74: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        mm_45: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_136, permute_74);  view_136 = permute_74 = None
        view_137: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_45, [4, 512, 1024]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_62: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_56, view_137);  add_56 = view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_161: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_28: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_161, 2)
        mean_27: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_91: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_161, rsqrt_27);  convert_element_type_161 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_162: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_91, torch.bfloat16);  mul_91 = None
        mul_92: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg76_1, convert_element_type_162);  arg76_1 = convert_element_type_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_138: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_92, [2048, 1024])
        permute_75: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        mm_46: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_138, permute_75);  view_138 = permute_75 = None
        view_139: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_46, [4, 512, 3072]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_165: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None
        neg_20: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_165)
        exp_6: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_64: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_165, add_64);  convert_element_type_165 = add_64 = None
        convert_element_type_166: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_140: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_92, [2048, 1024]);  mul_92 = None
        permute_76: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_47: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_140, permute_76);  view_140 = permute_76 = None
        view_141: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_47, [4, 512, 3072]);  mm_47 = None
        mul_93: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_166, view_141);  convert_element_type_166 = view_141 = None
        view_142: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_93, [2048, 3072]);  mul_93 = None
        permute_77: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_48: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_142, permute_77);  view_142 = permute_77 = None
        view_143: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_48, [4, 512, 1024]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_65: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_62, view_143);  add_62 = view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_171: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_65, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_29: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_171, 2)
        mean_28: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_66: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_94: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_171, rsqrt_28);  convert_element_type_171 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_172: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None
        mul_95: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg80_1, convert_element_type_172);  arg80_1 = convert_element_type_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_144: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_95, [2048, 1024])
        permute_78: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_49: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_144, permute_78);  view_144 = permute_78 = None
        view_145: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_49, [4, 512, 2048]);  mm_49 = None
        view_146: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_145, [4, 512, -1, 128]);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_175: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_30: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_175, 2)
        mean_29: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_67: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_96: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_175, rsqrt_29);  convert_element_type_175 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_176: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None
        mul_97: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg82_1, convert_element_type_176);  arg82_1 = convert_element_type_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_79: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_97, [0, 2, 1, 3]);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_42: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_100: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_79, unsqueeze_42)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_33: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_79, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_21: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_33);  slice_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_32: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_79, 3, 0, 64);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_15: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_21, slice_32], -1);  neg_21 = slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_43: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_101: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_15, unsqueeze_43);  cat_15 = None
        add_69: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_100, mul_101);  mul_100 = mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_147: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_95, [2048, 1024])
        permute_80: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_50: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_147, permute_80);  view_147 = permute_80 = None
        view_148: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_50, [4, 512, 1024]);  mm_50 = None
        view_149: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_148, [4, 512, -1, 128]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_179: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_149, torch.float32);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_31: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_179, 2)
        mean_30: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_68: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_98: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_179, rsqrt_30);  convert_element_type_179 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_180: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None
        mul_99: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg84_1, convert_element_type_180);  arg84_1 = convert_element_type_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_81: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_99, [0, 2, 1, 3]);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_102: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_81, unsqueeze_42);  unsqueeze_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_35: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_81, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_22: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_35);  slice_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_34: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_81, 3, 0, 64);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_16: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_22, slice_34], -1);  neg_22 = slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_103: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_16, unsqueeze_43);  cat_16 = unsqueeze_43 = None
        add_70: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_44: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_70, 2);  add_70 = None
        expand_20: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_44, [4, 8, 2, 512, 128]);  unsqueeze_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_16: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_153: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_16, [4, 16, 512, 128]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_150: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_95, [2048, 1024]);  mul_95 = None
        permute_82: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_51: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_150, permute_82);  view_150 = permute_82 = None
        view_151: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_51, [4, 512, 1024]);  mm_51 = None
        view_152: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_151, [4, 512, -1, 128]);  view_151 = None
        permute_83: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_45: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_83, 2);  permute_83 = None
        expand_21: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_45, [4, 8, 2, 512, 128]);  unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_17: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_154: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_17, [4, 16, 512, 128]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_16: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_69, view_153, view_154, where_7, False, scale = 0.08838834764831845);  add_69 = view_153 = view_154 = where_7 = None
        getitem_63: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_155: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_84, [4, 512, -1]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_156: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_155, [2048, 2048]);  view_155 = None
        permute_85: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_52: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_156, permute_85);  view_156 = permute_85 = None
        view_157: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_52, [4, 512, 1024]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_71: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_65, view_157);  add_65 = view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_185: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_32: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_185, 2)
        mean_31: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_72: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_104: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_185, rsqrt_31);  convert_element_type_185 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_186: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None
        mul_105: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg87_1, convert_element_type_186);  arg87_1 = convert_element_type_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_158: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_105, [2048, 1024])
        permute_86: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        mm_53: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_158, permute_86);  view_158 = permute_86 = None
        view_159: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_53, [4, 512, 3072]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_189: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        neg_23: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_189)
        exp_7: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_73: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_189, add_73);  convert_element_type_189 = add_73 = None
        convert_element_type_190: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_160: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_105, [2048, 1024]);  mul_105 = None
        permute_87: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_54: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_160, permute_87);  view_160 = permute_87 = None
        view_161: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_54, [4, 512, 3072]);  mm_54 = None
        mul_106: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_190, view_161);  convert_element_type_190 = view_161 = None
        view_162: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_106, [2048, 3072]);  mul_106 = None
        permute_88: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_55: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_162, permute_88);  view_162 = permute_88 = None
        view_163: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_55, [4, 512, 1024]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_74: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_71, view_163);  add_71 = view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_195: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_74, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_33: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_195, 2)
        mean_32: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_75: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        rsqrt_32: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_107: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_195, rsqrt_32);  convert_element_type_195 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_196: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_107, torch.bfloat16);  mul_107 = None
        mul_108: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg91_1, convert_element_type_196);  arg91_1 = convert_element_type_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_164: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_108, [2048, 1024])
        permute_89: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        mm_56: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_164, permute_89);  view_164 = permute_89 = None
        view_165: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_56, [4, 512, 2048]);  mm_56 = None
        view_166: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_165, [4, 512, -1, 128]);  view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_199: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_166, torch.float32);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_34: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_199, 2)
        mean_33: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_76: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        rsqrt_33: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_109: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_199, rsqrt_33);  convert_element_type_199 = rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_200: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        mul_110: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg93_1, convert_element_type_200);  arg93_1 = convert_element_type_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_90: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_110, [0, 2, 1, 3]);  mul_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_46: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_113: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_90, unsqueeze_46)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_37: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_90, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_24: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_37);  slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_36: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_90, 3, 0, 64);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_17: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_24, slice_36], -1);  neg_24 = slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_47: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_114: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_17, unsqueeze_47);  cat_17 = None
        add_78: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_167: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_108, [2048, 1024])
        permute_91: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_57: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_167, permute_91);  view_167 = permute_91 = None
        view_168: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_57, [4, 512, 1024]);  mm_57 = None
        view_169: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_168, [4, 512, -1, 128]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_203: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_169, torch.float32);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_35: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_203, 2)
        mean_34: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_77: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        rsqrt_34: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_111: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_203, rsqrt_34);  convert_element_type_203 = rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_204: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_111, torch.bfloat16);  mul_111 = None
        mul_112: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg95_1, convert_element_type_204);  arg95_1 = convert_element_type_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_92: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_112, [0, 2, 1, 3]);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_115: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_92, unsqueeze_46);  unsqueeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_39: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_92, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_25: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_39);  slice_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_38: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_92, 3, 0, 64);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_18: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_25, slice_38], -1);  neg_25 = slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_116: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_18, unsqueeze_47);  cat_18 = unsqueeze_47 = None
        add_79: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_48: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_79, 2);  add_79 = None
        expand_22: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_48, [4, 8, 2, 512, 128]);  unsqueeze_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_18: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_173: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_18, [4, 16, 512, 128]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_170: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_108, [2048, 1024]);  mul_108 = None
        permute_93: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_58: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_170, permute_93);  view_170 = permute_93 = None
        view_171: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_58, [4, 512, 1024]);  mm_58 = None
        view_172: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_171, [4, 512, -1, 128]);  view_171 = None
        permute_94: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_49: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_94, 2);  permute_94 = None
        expand_23: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_49, [4, 8, 2, 512, 128]);  unsqueeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_19: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_174: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_19, [4, 16, 512, 128]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_18: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_18, full_default_17);  full_default_18 = full_default_17 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_78, view_173, view_174, where_8, False, scale = 0.08838834764831845);  add_78 = view_173 = view_174 = where_8 = None
        getitem_72: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_175: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_95, [4, 512, -1]);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_176: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_175, [2048, 2048]);  view_175 = None
        permute_96: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        mm_59: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_176, permute_96);  view_176 = permute_96 = None
        view_177: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_59, [4, 512, 1024]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_80: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_74, view_177);  add_74 = view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_209: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_80, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_36: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_209, 2)
        mean_35: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_81: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        rsqrt_35: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        mul_117: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_209, rsqrt_35);  convert_element_type_209 = rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_210: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        mul_118: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg98_1, convert_element_type_210);  arg98_1 = convert_element_type_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_178: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_118, [2048, 1024])
        permute_97: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_60: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_178, permute_97);  view_178 = permute_97 = None
        view_179: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_60, [4, 512, 3072]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_213: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_179, torch.float32);  view_179 = None
        neg_26: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_213)
        exp_8: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_82: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_213, add_82);  convert_element_type_213 = add_82 = None
        convert_element_type_214: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_180: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_118, [2048, 1024]);  mul_118 = None
        permute_98: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_61: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_180, permute_98);  view_180 = permute_98 = None
        view_181: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_61, [4, 512, 3072]);  mm_61 = None
        mul_119: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_214, view_181);  convert_element_type_214 = view_181 = None
        view_182: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_119, [2048, 3072]);  mul_119 = None
        permute_99: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_62: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_182, permute_99);  view_182 = permute_99 = None
        view_183: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_62, [4, 512, 1024]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_83: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_80, view_183);  add_80 = view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_219: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_37: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_219, 2)
        mean_36: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_37, [-1], True);  pow_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_84: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        rsqrt_36: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_120: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_219, rsqrt_36);  convert_element_type_219 = rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_220: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        mul_121: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg102_1, convert_element_type_220);  arg102_1 = convert_element_type_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_184: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_121, [2048, 1024])
        permute_100: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_63: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_184, permute_100);  view_184 = permute_100 = None
        view_185: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_63, [4, 512, 2048]);  mm_63 = None
        view_186: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_185, [4, 512, -1, 128]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_223: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_186, torch.float32);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_38: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_223, 2)
        mean_37: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_85: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        rsqrt_37: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_122: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_223, rsqrt_37);  convert_element_type_223 = rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_224: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_122, torch.bfloat16);  mul_122 = None
        mul_123: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg104_1, convert_element_type_224);  arg104_1 = convert_element_type_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_101: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_123, [0, 2, 1, 3]);  mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_50: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_126: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_101, unsqueeze_50)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_41: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_101, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_27: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_41);  slice_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_40: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_101, 3, 0, 64);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_19: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_27, slice_40], -1);  neg_27 = slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_51: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_127: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_19, unsqueeze_51);  cat_19 = None
        add_87: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_126, mul_127);  mul_126 = mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_187: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_121, [2048, 1024])
        permute_102: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_64: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_187, permute_102);  view_187 = permute_102 = None
        view_188: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_64, [4, 512, 1024]);  mm_64 = None
        view_189: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_188, [4, 512, -1, 128]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_227: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_189, torch.float32);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_39: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_227, 2)
        mean_38: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_86: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        rsqrt_38: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_124: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_227, rsqrt_38);  convert_element_type_227 = rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_228: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_124, torch.bfloat16);  mul_124 = None
        mul_125: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg106_1, convert_element_type_228);  arg106_1 = convert_element_type_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_103: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_125, [0, 2, 1, 3]);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_128: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_103, unsqueeze_50);  unsqueeze_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_43: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_103, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_28: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_43);  slice_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_42: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_103, 3, 0, 64);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_20: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_28, slice_42], -1);  neg_28 = slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_129: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_20, unsqueeze_51);  cat_20 = unsqueeze_51 = None
        add_88: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_128, mul_129);  mul_128 = mul_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_52: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_88, 2);  add_88 = None
        expand_24: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_52, [4, 8, 2, 512, 128]);  unsqueeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_20: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_193: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_20, [4, 16, 512, 128]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_190: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_121, [2048, 1024]);  mul_121 = None
        permute_104: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_65: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_190, permute_104);  view_190 = permute_104 = None
        view_191: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_65, [4, 512, 1024]);  mm_65 = None
        view_192: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_191, [4, 512, -1, 128]);  view_191 = None
        permute_105: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_53: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_105, 2);  permute_105 = None
        expand_25: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_53, [4, 8, 2, 512, 128]);  unsqueeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_21: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_194: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_21, [4, 16, 512, 128]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_20: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_20, full_default_19);  full_default_20 = full_default_19 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_87, view_193, view_194, where_9, False, scale = 0.08838834764831845);  add_87 = view_193 = view_194 = where_9 = None
        getitem_81: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3]);  getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_195: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_106, [4, 512, -1]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_196: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_195, [2048, 2048]);  view_195 = None
        permute_107: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_66: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_196, permute_107);  view_196 = permute_107 = None
        view_197: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_66, [4, 512, 1024]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_89: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_83, view_197);  add_83 = view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_233: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_89, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_40: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 2)
        mean_39: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_90: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        rsqrt_39: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_130: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_233, rsqrt_39);  convert_element_type_233 = rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_234: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_130, torch.bfloat16);  mul_130 = None
        mul_131: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg109_1, convert_element_type_234);  arg109_1 = convert_element_type_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_198: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_131, [2048, 1024])
        permute_108: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        mm_67: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_198, permute_108);  view_198 = permute_108 = None
        view_199: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_67, [4, 512, 3072]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_237: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_199, torch.float32);  view_199 = None
        neg_29: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_237)
        exp_9: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_91: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_237, add_91);  convert_element_type_237 = add_91 = None
        convert_element_type_238: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_200: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_131, [2048, 1024]);  mul_131 = None
        permute_109: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        mm_68: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_200, permute_109);  view_200 = permute_109 = None
        view_201: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_68, [4, 512, 3072]);  mm_68 = None
        mul_132: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_238, view_201);  convert_element_type_238 = view_201 = None
        view_202: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_132, [2048, 3072]);  mul_132 = None
        permute_110: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_69: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_202, permute_110);  view_202 = permute_110 = None
        view_203: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_69, [4, 512, 1024]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_92: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_89, view_203);  add_89 = view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_243: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_92, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_41: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_243, 2)
        mean_40: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_41, [-1], True);  pow_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_93: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        rsqrt_40: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_133: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_243, rsqrt_40);  convert_element_type_243 = rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_244: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None
        mul_134: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg113_1, convert_element_type_244);  arg113_1 = convert_element_type_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_204: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_134, [2048, 1024])
        permute_111: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_70: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_204, permute_111);  view_204 = permute_111 = None
        view_205: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_70, [4, 512, 2048]);  mm_70 = None
        view_206: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_205, [4, 512, -1, 128]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_247: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_206, torch.float32);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_42: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_247, 2)
        mean_41: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_94: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        rsqrt_41: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_135: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_247, rsqrt_41);  convert_element_type_247 = rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_248: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_135, torch.bfloat16);  mul_135 = None
        mul_136: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg115_1, convert_element_type_248);  arg115_1 = convert_element_type_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_112: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_136, [0, 2, 1, 3]);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_54: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_139: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_112, unsqueeze_54)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_45: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_112, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_30: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_45);  slice_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_44: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_112, 3, 0, 64);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_21: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_30, slice_44], -1);  neg_30 = slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_55: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_140: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_21, unsqueeze_55);  cat_21 = None
        add_96: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_207: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_134, [2048, 1024])
        permute_113: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        mm_71: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_207, permute_113);  view_207 = permute_113 = None
        view_208: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_71, [4, 512, 1024]);  mm_71 = None
        view_209: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_208, [4, 512, -1, 128]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_251: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_43: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_251, 2)
        mean_42: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_95: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_42, 1e-06);  mean_42 = None
        rsqrt_42: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_137: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_251, rsqrt_42);  convert_element_type_251 = rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_252: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None
        mul_138: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg117_1, convert_element_type_252);  arg117_1 = convert_element_type_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_114: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_138, [0, 2, 1, 3]);  mul_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_141: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_114, unsqueeze_54);  unsqueeze_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_47: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_114, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_31: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_47);  slice_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_46: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_114, 3, 0, 64);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_22: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_31, slice_46], -1);  neg_31 = slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_142: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_22, unsqueeze_55);  cat_22 = unsqueeze_55 = None
        add_97: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_56: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_97, 2);  add_97 = None
        expand_26: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_56, [4, 8, 2, 512, 128]);  unsqueeze_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_22: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_213: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_22, [4, 16, 512, 128]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_210: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_134, [2048, 1024]);  mul_134 = None
        permute_115: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_72: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_210, permute_115);  view_210 = permute_115 = None
        view_211: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_72, [4, 512, 1024]);  mm_72 = None
        view_212: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_211, [4, 512, -1, 128]);  view_211 = None
        permute_116: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_57: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_116, 2);  permute_116 = None
        expand_27: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_57, [4, 8, 2, 512, 128]);  unsqueeze_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_23: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_214: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_23, [4, 16, 512, 128]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_22: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_96, view_213, view_214, where_10, False, scale = 0.08838834764831845);  add_96 = view_213 = view_214 = where_10 = None
        getitem_90: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_215: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_117, [4, 512, -1]);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_216: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_215, [2048, 2048]);  view_215 = None
        permute_118: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_73: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_216, permute_118);  view_216 = permute_118 = None
        view_217: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_73, [4, 512, 1024]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_98: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_92, view_217);  add_92 = view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_257: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_98, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_44: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_257, 2)
        mean_43: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_99: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_43, 1e-06);  mean_43 = None
        rsqrt_43: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_143: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_257, rsqrt_43);  convert_element_type_257 = rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_258: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None
        mul_144: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg120_1, convert_element_type_258);  arg120_1 = convert_element_type_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_218: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_144, [2048, 1024])
        permute_119: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_74: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_218, permute_119);  view_218 = permute_119 = None
        view_219: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_74, [4, 512, 3072]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_261: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_219, torch.float32);  view_219 = None
        neg_32: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_261)
        exp_10: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_100: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_261, add_100);  convert_element_type_261 = add_100 = None
        convert_element_type_262: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_220: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_144, [2048, 1024]);  mul_144 = None
        permute_120: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_75: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_220, permute_120);  view_220 = permute_120 = None
        view_221: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_75, [4, 512, 3072]);  mm_75 = None
        mul_145: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_262, view_221);  convert_element_type_262 = view_221 = None
        view_222: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_145, [2048, 3072]);  mul_145 = None
        permute_121: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_76: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_222, permute_121);  view_222 = permute_121 = None
        view_223: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_76, [4, 512, 1024]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_101: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_98, view_223);  add_98 = view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_267: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_101, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_45: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_267, 2)
        mean_44: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_45, [-1], True);  pow_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_102: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_44, 1e-06);  mean_44 = None
        rsqrt_44: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_146: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_267, rsqrt_44);  convert_element_type_267 = rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_268: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_146, torch.bfloat16);  mul_146 = None
        mul_147: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg124_1, convert_element_type_268);  arg124_1 = convert_element_type_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_224: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_147, [2048, 1024])
        permute_122: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        mm_77: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_224, permute_122);  view_224 = permute_122 = None
        view_225: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_77, [4, 512, 2048]);  mm_77 = None
        view_226: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_225, [4, 512, -1, 128]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_271: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_226, torch.float32);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_46: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_271, 2)
        mean_45: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_103: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_45, 1e-06);  mean_45 = None
        rsqrt_45: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_148: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_271, rsqrt_45);  convert_element_type_271 = rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_272: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_148, torch.bfloat16);  mul_148 = None
        mul_149: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg126_1, convert_element_type_272);  arg126_1 = convert_element_type_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_123: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_149, [0, 2, 1, 3]);  mul_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_58: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_152: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_123, unsqueeze_58)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_49: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_123, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_33: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_49);  slice_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_48: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_123, 3, 0, 64);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_23: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_33, slice_48], -1);  neg_33 = slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_59: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_153: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_23, unsqueeze_59);  cat_23 = None
        add_105: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_227: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_147, [2048, 1024])
        permute_124: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_78: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_227, permute_124);  view_227 = permute_124 = None
        view_228: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_78, [4, 512, 1024]);  mm_78 = None
        view_229: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_228, [4, 512, -1, 128]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_275: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_229, torch.float32);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_47: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_275, 2)
        mean_46: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_104: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_46, 1e-06);  mean_46 = None
        rsqrt_46: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_150: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_275, rsqrt_46);  convert_element_type_275 = rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_276: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None
        mul_151: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg128_1, convert_element_type_276);  arg128_1 = convert_element_type_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_125: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_151, [0, 2, 1, 3]);  mul_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_154: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_125, unsqueeze_58);  unsqueeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_51: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_125, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_34: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_51);  slice_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_50: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_125, 3, 0, 64);  permute_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_24: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_34, slice_50], -1);  neg_34 = slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_155: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_24, unsqueeze_59);  cat_24 = unsqueeze_59 = None
        add_106: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_60: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_106, 2);  add_106 = None
        expand_28: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_60, [4, 8, 2, 512, 128]);  unsqueeze_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_24: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_233: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_24, [4, 16, 512, 128]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_230: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_147, [2048, 1024]);  mul_147 = None
        permute_126: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        mm_79: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_230, permute_126);  view_230 = permute_126 = None
        view_231: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_79, [4, 512, 1024]);  mm_79 = None
        view_232: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_231, [4, 512, -1, 128]);  view_231 = None
        permute_127: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_61: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_127, 2);  permute_127 = None
        expand_29: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_61, [4, 8, 2, 512, 128]);  unsqueeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_25: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_234: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_25, [4, 16, 512, 128]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_24: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_24, full_default_23);  full_default_24 = full_default_23 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_105, view_233, view_234, where_11, False, scale = 0.08838834764831845);  add_105 = view_233 = view_234 = where_11 = None
        getitem_99: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_128, [4, 512, -1]);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_236: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_235, [2048, 2048]);  view_235 = None
        permute_129: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        mm_80: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_236, permute_129);  view_236 = permute_129 = None
        view_237: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_80, [4, 512, 1024]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_107: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_101, view_237);  add_101 = view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_281: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_48: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_281, 2)
        mean_47: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_108: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_47, 1e-06);  mean_47 = None
        rsqrt_47: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_156: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_281, rsqrt_47);  convert_element_type_281 = rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_282: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_156, torch.bfloat16);  mul_156 = None
        mul_157: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg131_1, convert_element_type_282);  arg131_1 = convert_element_type_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_238: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_157, [2048, 1024])
        permute_130: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        mm_81: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_238, permute_130);  view_238 = permute_130 = None
        view_239: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_81, [4, 512, 3072]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_285: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        neg_35: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_285)
        exp_11: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_35);  neg_35 = None
        add_109: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_285, add_109);  convert_element_type_285 = add_109 = None
        convert_element_type_286: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_240: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_157, [2048, 1024]);  mul_157 = None
        permute_131: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_82: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_240, permute_131);  view_240 = permute_131 = None
        view_241: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_82, [4, 512, 3072]);  mm_82 = None
        mul_158: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_286, view_241);  convert_element_type_286 = view_241 = None
        view_242: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_158, [2048, 3072]);  mul_158 = None
        permute_132: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        mm_83: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_242, permute_132);  view_242 = permute_132 = None
        view_243: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_83, [4, 512, 1024]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_110: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_107, view_243);  add_107 = view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_291: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_110, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_49: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_291, 2)
        mean_48: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_49, [-1], True);  pow_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_111: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_48, 1e-06);  mean_48 = None
        rsqrt_48: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_159: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_291, rsqrt_48);  convert_element_type_291 = rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_292: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_159, torch.bfloat16);  mul_159 = None
        mul_160: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg135_1, convert_element_type_292);  arg135_1 = convert_element_type_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_244: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_160, [2048, 1024])
        permute_133: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        mm_84: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_244, permute_133);  view_244 = permute_133 = None
        view_245: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_84, [4, 512, 2048]);  mm_84 = None
        view_246: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_245, [4, 512, -1, 128]);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_295: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_50: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_295, 2)
        mean_49: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_112: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_49, 1e-06);  mean_49 = None
        rsqrt_49: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_161: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_295, rsqrt_49);  convert_element_type_295 = rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_296: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16);  mul_161 = None
        mul_162: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg137_1, convert_element_type_296);  arg137_1 = convert_element_type_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_134: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_162, [0, 2, 1, 3]);  mul_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_62: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_165: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_134, unsqueeze_62)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_53: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_134, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_36: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_53);  slice_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_52: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_134, 3, 0, 64);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_25: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_36, slice_52], -1);  neg_36 = slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_63: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_166: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_25, unsqueeze_63);  cat_25 = None
        add_114: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_247: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_160, [2048, 1024])
        permute_135: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        mm_85: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_247, permute_135);  view_247 = permute_135 = None
        view_248: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_85, [4, 512, 1024]);  mm_85 = None
        view_249: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_248, [4, 512, -1, 128]);  view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_299: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_51: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_299, 2)
        mean_50: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_113: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_50, 1e-06);  mean_50 = None
        rsqrt_50: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_163: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_299, rsqrt_50);  convert_element_type_299 = rsqrt_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_300: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_163, torch.bfloat16);  mul_163 = None
        mul_164: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg139_1, convert_element_type_300);  arg139_1 = convert_element_type_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_136: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_164, [0, 2, 1, 3]);  mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_167: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_136, unsqueeze_62);  unsqueeze_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_55: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_136, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_37: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_55);  slice_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_54: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_136, 3, 0, 64);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_26: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_37, slice_54], -1);  neg_37 = slice_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_168: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_26, unsqueeze_63);  cat_26 = unsqueeze_63 = None
        add_115: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_64: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_115, 2);  add_115 = None
        expand_30: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_64, [4, 8, 2, 512, 128]);  unsqueeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_26: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_253: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_26, [4, 16, 512, 128]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_250: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_160, [2048, 1024]);  mul_160 = None
        permute_137: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        mm_86: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_250, permute_137);  view_250 = permute_137 = None
        view_251: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_86, [4, 512, 1024]);  mm_86 = None
        view_252: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_251, [4, 512, -1, 128]);  view_251 = None
        permute_138: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_65: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_138, 2);  permute_138 = None
        expand_31: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_65, [4, 8, 2, 512, 128]);  unsqueeze_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_27: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_254: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_27, [4, 16, 512, 128]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_26: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_26, full_default_25);  full_default_26 = full_default_25 = None
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_114, view_253, view_254, where_12, False, scale = 0.08838834764831845);  add_114 = view_253 = view_254 = where_12 = None
        getitem_108: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_12[0];  _scaled_dot_product_cudnn_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_139: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_255: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_139, [4, 512, -1]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_256: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_255, [2048, 2048]);  view_255 = None
        permute_140: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        mm_87: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_256, permute_140);  view_256 = permute_140 = None
        view_257: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_87, [4, 512, 1024]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_116: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_110, view_257);  add_110 = view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_305: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_116, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_52: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_305, 2)
        mean_51: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_117: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_51, 1e-06);  mean_51 = None
        rsqrt_51: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        mul_169: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_305, rsqrt_51);  convert_element_type_305 = rsqrt_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_306: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_169, torch.bfloat16);  mul_169 = None
        mul_170: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg142_1, convert_element_type_306);  arg142_1 = convert_element_type_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_258: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_170, [2048, 1024])
        permute_141: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        mm_88: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_258, permute_141);  view_258 = permute_141 = None
        view_259: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_88, [4, 512, 3072]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_309: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_259, torch.float32);  view_259 = None
        neg_38: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_309)
        exp_12: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_38);  neg_38 = None
        add_118: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_309, add_118);  convert_element_type_309 = add_118 = None
        convert_element_type_310: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_260: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_170, [2048, 1024]);  mul_170 = None
        permute_142: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        mm_89: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_260, permute_142);  view_260 = permute_142 = None
        view_261: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_89, [4, 512, 3072]);  mm_89 = None
        mul_171: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_310, view_261);  convert_element_type_310 = view_261 = None
        view_262: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_171, [2048, 3072]);  mul_171 = None
        permute_143: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        mm_90: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_262, permute_143);  view_262 = permute_143 = None
        view_263: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_90, [4, 512, 1024]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_119: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_116, view_263);  add_116 = view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_315: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_119, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_53: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_315, 2)
        mean_52: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_53, [-1], True);  pow_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_120: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_52, 1e-06);  mean_52 = None
        rsqrt_52: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_172: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_315, rsqrt_52);  convert_element_type_315 = rsqrt_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_316: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_172, torch.bfloat16);  mul_172 = None
        mul_173: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg146_1, convert_element_type_316);  arg146_1 = convert_element_type_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_264: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_173, [2048, 1024])
        permute_144: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_91: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_264, permute_144);  view_264 = permute_144 = None
        view_265: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_91, [4, 512, 2048]);  mm_91 = None
        view_266: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_265, [4, 512, -1, 128]);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_319: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_266, torch.float32);  view_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_54: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_319, 2)
        mean_53: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_121: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_53, 1e-06);  mean_53 = None
        rsqrt_53: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        mul_174: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_319, rsqrt_53);  convert_element_type_319 = rsqrt_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_320: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        mul_175: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg148_1, convert_element_type_320);  arg148_1 = convert_element_type_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_145: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_175, [0, 2, 1, 3]);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_66: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_178: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_145, unsqueeze_66)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_57: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_145, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_39: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_57);  slice_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_56: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_145, 3, 0, 64);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_27: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_39, slice_56], -1);  neg_39 = slice_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_67: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_179: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_27, unsqueeze_67);  cat_27 = None
        add_123: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_267: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_173, [2048, 1024])
        permute_146: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_92: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_267, permute_146);  view_267 = permute_146 = None
        view_268: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_92, [4, 512, 1024]);  mm_92 = None
        view_269: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_268, [4, 512, -1, 128]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_323: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_269, torch.float32);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_55: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_323, 2)
        mean_54: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_122: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_54, 1e-06);  mean_54 = None
        rsqrt_54: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_176: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_323, rsqrt_54);  convert_element_type_323 = rsqrt_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_324: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_176, torch.bfloat16);  mul_176 = None
        mul_177: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg150_1, convert_element_type_324);  arg150_1 = convert_element_type_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_147: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_177, [0, 2, 1, 3]);  mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_180: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_147, unsqueeze_66);  unsqueeze_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_59: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_147, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_40: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_59);  slice_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_58: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_147, 3, 0, 64);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_28: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_40, slice_58], -1);  neg_40 = slice_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_181: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_28, unsqueeze_67);  cat_28 = unsqueeze_67 = None
        add_124: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_68: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_124, 2);  add_124 = None
        expand_32: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_68, [4, 8, 2, 512, 128]);  unsqueeze_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_28: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_273: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_28, [4, 16, 512, 128]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_270: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_173, [2048, 1024]);  mul_173 = None
        permute_148: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_93: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_270, permute_148);  view_270 = permute_148 = None
        view_271: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_93, [4, 512, 1024]);  mm_93 = None
        view_272: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_271, [4, 512, -1, 128]);  view_271 = None
        permute_149: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_69: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_149, 2);  permute_149 = None
        expand_33: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_69, [4, 8, 2, 512, 128]);  unsqueeze_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_29: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_274: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_29, [4, 16, 512, 128]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_28: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_28, full_default_27);  full_default_28 = full_default_27 = None
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_123, view_273, view_274, where_13, False, scale = 0.08838834764831845);  add_123 = view_273 = view_274 = where_13 = None
        getitem_117: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_13[0];  _scaled_dot_product_cudnn_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_150: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_275: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_150, [4, 512, -1]);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_276: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_275, [2048, 2048]);  view_275 = None
        permute_151: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        mm_94: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_276, permute_151);  view_276 = permute_151 = None
        view_277: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_94, [4, 512, 1024]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_125: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_119, view_277);  add_119 = view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_329: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_125, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_56: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_329, 2)
        mean_55: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_126: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_55, 1e-06);  mean_55 = None
        rsqrt_55: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_182: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_329, rsqrt_55);  convert_element_type_329 = rsqrt_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_330: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None
        mul_183: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg153_1, convert_element_type_330);  arg153_1 = convert_element_type_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_278: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_183, [2048, 1024])
        permute_152: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        mm_95: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_278, permute_152);  view_278 = permute_152 = None
        view_279: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_95, [4, 512, 3072]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_333: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        neg_41: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_333)
        exp_13: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_41);  neg_41 = None
        add_127: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_333, add_127);  convert_element_type_333 = add_127 = None
        convert_element_type_334: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_280: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_183, [2048, 1024]);  mul_183 = None
        permute_153: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        mm_96: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_280, permute_153);  view_280 = permute_153 = None
        view_281: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_96, [4, 512, 3072]);  mm_96 = None
        mul_184: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_334, view_281);  convert_element_type_334 = view_281 = None
        view_282: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_184, [2048, 3072]);  mul_184 = None
        permute_154: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        mm_97: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_282, permute_154);  view_282 = permute_154 = None
        view_283: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_97, [4, 512, 1024]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_128: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_125, view_283);  add_125 = view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_339: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_128, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_57: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_339, 2)
        mean_56: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_57, [-1], True);  pow_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_129: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_56, 1e-06);  mean_56 = None
        rsqrt_56: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_185: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_339, rsqrt_56);  convert_element_type_339 = rsqrt_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_340: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_185, torch.bfloat16);  mul_185 = None
        mul_186: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg157_1, convert_element_type_340);  arg157_1 = convert_element_type_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_284: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_186, [2048, 1024])
        permute_155: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        mm_98: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_284, permute_155);  view_284 = permute_155 = None
        view_285: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_98, [4, 512, 2048]);  mm_98 = None
        view_286: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_285, [4, 512, -1, 128]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_343: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_286, torch.float32);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_58: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_343, 2)
        mean_57: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_130: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_57, 1e-06);  mean_57 = None
        rsqrt_57: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_187: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_343, rsqrt_57);  convert_element_type_343 = rsqrt_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_344: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_187, torch.bfloat16);  mul_187 = None
        mul_188: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg159_1, convert_element_type_344);  arg159_1 = convert_element_type_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_156: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_188, [0, 2, 1, 3]);  mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_70: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_191: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_156, unsqueeze_70)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_61: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_156, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_42: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_61);  slice_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_60: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_156, 3, 0, 64);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_29: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_42, slice_60], -1);  neg_42 = slice_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_71: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_192: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_29, unsqueeze_71);  cat_29 = None
        add_132: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_191, mul_192);  mul_191 = mul_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_287: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_186, [2048, 1024])
        permute_157: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_99: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_287, permute_157);  view_287 = permute_157 = None
        view_288: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_99, [4, 512, 1024]);  mm_99 = None
        view_289: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_288, [4, 512, -1, 128]);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_347: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_289, torch.float32);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_59: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_347, 2)
        mean_58: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_59, [-1], True);  pow_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_131: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_58, 1e-06);  mean_58 = None
        rsqrt_58: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        mul_189: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_347, rsqrt_58);  convert_element_type_347 = rsqrt_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_348: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        mul_190: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg161_1, convert_element_type_348);  arg161_1 = convert_element_type_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_158: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_190, [0, 2, 1, 3]);  mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_193: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_158, unsqueeze_70);  unsqueeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_63: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_158, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_43: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_63);  slice_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_62: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_158, 3, 0, 64);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_30: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_43, slice_62], -1);  neg_43 = slice_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_194: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_30, unsqueeze_71);  cat_30 = unsqueeze_71 = None
        add_133: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_72: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_133, 2);  add_133 = None
        expand_34: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_72, [4, 8, 2, 512, 128]);  unsqueeze_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_30: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_293: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_30, [4, 16, 512, 128]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_290: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_186, [2048, 1024]);  mul_186 = None
        permute_159: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        mm_100: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_290, permute_159);  view_290 = permute_159 = None
        view_291: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_100, [4, 512, 1024]);  mm_100 = None
        view_292: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_291, [4, 512, -1, 128]);  view_291 = None
        permute_160: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_73: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_160, 2);  permute_160 = None
        expand_35: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_73, [4, 8, 2, 512, 128]);  unsqueeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_31: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_294: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_31, [4, 16, 512, 128]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_30: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_30, full_default_29);  full_default_30 = full_default_29 = None
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_132, view_293, view_294, where_14, False, scale = 0.08838834764831845);  add_132 = view_293 = view_294 = where_14 = None
        getitem_126: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_14[0];  _scaled_dot_product_cudnn_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_161: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_295: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_161, [4, 512, -1]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_296: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_295, [2048, 2048]);  view_295 = None
        permute_162: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_101: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_296, permute_162);  view_296 = permute_162 = None
        view_297: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_101, [4, 512, 1024]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_134: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_128, view_297);  add_128 = view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_353: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_134, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_60: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_353, 2)
        mean_59: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_60, [-1], True);  pow_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_135: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_59, 1e-06);  mean_59 = None
        rsqrt_59: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        mul_195: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_353, rsqrt_59);  convert_element_type_353 = rsqrt_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_354: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_195, torch.bfloat16);  mul_195 = None
        mul_196: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg164_1, convert_element_type_354);  arg164_1 = convert_element_type_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_298: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_196, [2048, 1024])
        permute_163: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        mm_102: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_298, permute_163);  view_298 = permute_163 = None
        view_299: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_102, [4, 512, 3072]);  mm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_357: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        neg_44: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_357)
        exp_14: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_44);  neg_44 = None
        add_136: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_357, add_136);  convert_element_type_357 = add_136 = None
        convert_element_type_358: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_300: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_196, [2048, 1024]);  mul_196 = None
        permute_164: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        mm_103: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_300, permute_164);  view_300 = permute_164 = None
        view_301: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_103, [4, 512, 3072]);  mm_103 = None
        mul_197: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_358, view_301);  convert_element_type_358 = view_301 = None
        view_302: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_197, [2048, 3072]);  mul_197 = None
        permute_165: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        mm_104: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_302, permute_165);  view_302 = permute_165 = None
        view_303: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_104, [4, 512, 1024]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_137: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_134, view_303);  add_134 = view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_363: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_137, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_61: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_363, 2)
        mean_60: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_61, [-1], True);  pow_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_138: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_60, 1e-06);  mean_60 = None
        rsqrt_60: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_198: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_363, rsqrt_60);  convert_element_type_363 = rsqrt_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_364: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_198, torch.bfloat16);  mul_198 = None
        mul_199: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg168_1, convert_element_type_364);  arg168_1 = convert_element_type_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_304: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_199, [2048, 1024])
        permute_166: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_105: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_304, permute_166);  view_304 = permute_166 = None
        view_305: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_105, [4, 512, 2048]);  mm_105 = None
        view_306: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_305, [4, 512, -1, 128]);  view_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_367: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_306, torch.float32);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_62: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_367, 2)
        mean_61: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_62, [-1], True);  pow_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_139: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_61, 1e-06);  mean_61 = None
        rsqrt_61: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_200: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_367, rsqrt_61);  convert_element_type_367 = rsqrt_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_368: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_200, torch.bfloat16);  mul_200 = None
        mul_201: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg170_1, convert_element_type_368);  arg170_1 = convert_element_type_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_167: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_201, [0, 2, 1, 3]);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_74: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_204: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_167, unsqueeze_74)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_65: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_167, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_45: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_65);  slice_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_64: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_167, 3, 0, 64);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_31: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_45, slice_64], -1);  neg_45 = slice_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_75: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_205: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_31, unsqueeze_75);  cat_31 = None
        add_141: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_307: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_199, [2048, 1024])
        permute_168: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        mm_106: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_307, permute_168);  view_307 = permute_168 = None
        view_308: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_106, [4, 512, 1024]);  mm_106 = None
        view_309: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_308, [4, 512, -1, 128]);  view_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_371: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_309, torch.float32);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_63: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_371, 2)
        mean_62: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_63, [-1], True);  pow_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_140: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_62, 1e-06);  mean_62 = None
        rsqrt_62: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_202: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_371, rsqrt_62);  convert_element_type_371 = rsqrt_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_372: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_202, torch.bfloat16);  mul_202 = None
        mul_203: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg172_1, convert_element_type_372);  arg172_1 = convert_element_type_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_169: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_203, [0, 2, 1, 3]);  mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_206: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_169, unsqueeze_74);  unsqueeze_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_67: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_169, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_46: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_67);  slice_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_66: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_169, 3, 0, 64);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_32: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_46, slice_66], -1);  neg_46 = slice_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_207: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_32, unsqueeze_75);  cat_32 = unsqueeze_75 = None
        add_142: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_76: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_142, 2);  add_142 = None
        expand_36: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_76, [4, 8, 2, 512, 128]);  unsqueeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_32: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_313: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_32, [4, 16, 512, 128]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_310: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_199, [2048, 1024]);  mul_199 = None
        permute_170: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_107: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_310, permute_170);  view_310 = permute_170 = None
        view_311: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_107, [4, 512, 1024]);  mm_107 = None
        view_312: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_311, [4, 512, -1, 128]);  view_311 = None
        permute_171: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_77: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_171, 2);  permute_171 = None
        expand_37: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_77, [4, 8, 2, 512, 128]);  unsqueeze_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_33: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_314: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_33, [4, 16, 512, 128]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_32: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_32, full_default_31);  full_default_32 = full_default_31 = None
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_141, view_313, view_314, where_15, False, scale = 0.08838834764831845);  add_141 = view_313 = view_314 = where_15 = None
        getitem_135: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_15[0];  _scaled_dot_product_cudnn_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_172: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_315: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_172, [4, 512, -1]);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_316: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_315, [2048, 2048]);  view_315 = None
        permute_173: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        mm_108: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_316, permute_173);  view_316 = permute_173 = None
        view_317: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_108, [4, 512, 1024]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_143: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_137, view_317);  add_137 = view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_377: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_143, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_64: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_377, 2)
        mean_63: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_64, [-1], True);  pow_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_144: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_63, 1e-06);  mean_63 = None
        rsqrt_63: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_208: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_377, rsqrt_63);  convert_element_type_377 = rsqrt_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_378: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_208, torch.bfloat16);  mul_208 = None
        mul_209: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg175_1, convert_element_type_378);  arg175_1 = convert_element_type_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_318: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_209, [2048, 1024])
        permute_174: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        mm_109: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_318, permute_174);  view_318 = permute_174 = None
        view_319: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_109, [4, 512, 3072]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_381: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_319, torch.float32);  view_319 = None
        neg_47: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_381)
        exp_15: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_47);  neg_47 = None
        add_145: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_381, add_145);  convert_element_type_381 = add_145 = None
        convert_element_type_382: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_320: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_209, [2048, 1024]);  mul_209 = None
        permute_175: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        mm_110: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_320, permute_175);  view_320 = permute_175 = None
        view_321: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_110, [4, 512, 3072]);  mm_110 = None
        mul_210: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_382, view_321);  convert_element_type_382 = view_321 = None
        view_322: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_210, [2048, 3072]);  mul_210 = None
        permute_176: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        mm_111: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_322, permute_176);  view_322 = permute_176 = None
        view_323: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_111, [4, 512, 1024]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_146: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_143, view_323);  add_143 = view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_387: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_146, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_65: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_387, 2)
        mean_64: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_65, [-1], True);  pow_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_147: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_64, 1e-06);  mean_64 = None
        rsqrt_64: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        mul_211: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_387, rsqrt_64);  convert_element_type_387 = rsqrt_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_388: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_211, torch.bfloat16);  mul_211 = None
        mul_212: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg179_1, convert_element_type_388);  arg179_1 = convert_element_type_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_324: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_212, [2048, 1024])
        permute_177: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        mm_112: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_324, permute_177);  view_324 = permute_177 = None
        view_325: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_112, [4, 512, 2048]);  mm_112 = None
        view_326: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_325, [4, 512, -1, 128]);  view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_391: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_326, torch.float32);  view_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_66: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_391, 2)
        mean_65: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_66, [-1], True);  pow_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_148: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_65, 1e-06);  mean_65 = None
        rsqrt_65: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_213: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_391, rsqrt_65);  convert_element_type_391 = rsqrt_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_392: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_213, torch.bfloat16);  mul_213 = None
        mul_214: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg181_1, convert_element_type_392);  arg181_1 = convert_element_type_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_178: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_214, [0, 2, 1, 3]);  mul_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_78: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_217: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_178, unsqueeze_78)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_69: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_178, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_48: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_69);  slice_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_68: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_178, 3, 0, 64);  permute_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_33: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_48, slice_68], -1);  neg_48 = slice_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_79: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_218: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_33, unsqueeze_79);  cat_33 = None
        add_150: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_327: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_212, [2048, 1024])
        permute_179: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        mm_113: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_327, permute_179);  view_327 = permute_179 = None
        view_328: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_113, [4, 512, 1024]);  mm_113 = None
        view_329: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_328, [4, 512, -1, 128]);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_395: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_329, torch.float32);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_67: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_395, 2)
        mean_66: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_67, [-1], True);  pow_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_149: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_66, 1e-06);  mean_66 = None
        rsqrt_66: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_215: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_395, rsqrt_66);  convert_element_type_395 = rsqrt_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_396: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_215, torch.bfloat16);  mul_215 = None
        mul_216: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg183_1, convert_element_type_396);  arg183_1 = convert_element_type_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_180: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_216, [0, 2, 1, 3]);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_219: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_180, unsqueeze_78);  unsqueeze_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_71: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_180, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_49: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_71);  slice_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_70: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_180, 3, 0, 64);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_34: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_49, slice_70], -1);  neg_49 = slice_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_220: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_34, unsqueeze_79);  cat_34 = unsqueeze_79 = None
        add_151: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_80: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_151, 2);  add_151 = None
        expand_38: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_80, [4, 8, 2, 512, 128]);  unsqueeze_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_34: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_333: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_34, [4, 16, 512, 128]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_330: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_212, [2048, 1024]);  mul_212 = None
        permute_181: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        mm_114: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_330, permute_181);  view_330 = permute_181 = None
        view_331: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_114, [4, 512, 1024]);  mm_114 = None
        view_332: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_331, [4, 512, -1, 128]);  view_331 = None
        permute_182: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_81: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_182, 2);  permute_182 = None
        expand_39: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_81, [4, 8, 2, 512, 128]);  unsqueeze_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_35: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_334: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_35, [4, 16, 512, 128]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_34: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_34, full_default_33);  full_default_34 = full_default_33 = None
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_150, view_333, view_334, where_16, False, scale = 0.08838834764831845);  add_150 = view_333 = view_334 = where_16 = None
        getitem_144: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_16[0];  _scaled_dot_product_cudnn_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_183: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_335: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_183, [4, 512, -1]);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_336: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_335, [2048, 2048]);  view_335 = None
        permute_184: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_115: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_336, permute_184);  view_336 = permute_184 = None
        view_337: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_115, [4, 512, 1024]);  mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_152: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_146, view_337);  add_146 = view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_401: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_152, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_68: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_401, 2)
        mean_67: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_68, [-1], True);  pow_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_153: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_67, 1e-06);  mean_67 = None
        rsqrt_67: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        mul_221: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_401, rsqrt_67);  convert_element_type_401 = rsqrt_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_402: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None
        mul_222: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg186_1, convert_element_type_402);  arg186_1 = convert_element_type_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_338: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_222, [2048, 1024])
        permute_185: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        mm_116: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_338, permute_185);  view_338 = permute_185 = None
        view_339: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_116, [4, 512, 3072]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_405: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        neg_50: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_405)
        exp_16: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_50);  neg_50 = None
        add_154: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_405, add_154);  convert_element_type_405 = add_154 = None
        convert_element_type_406: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_340: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_222, [2048, 1024]);  mul_222 = None
        permute_186: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        mm_117: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_340, permute_186);  view_340 = permute_186 = None
        view_341: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_117, [4, 512, 3072]);  mm_117 = None
        mul_223: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_406, view_341);  convert_element_type_406 = view_341 = None
        view_342: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_223, [2048, 3072]);  mul_223 = None
        permute_187: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        mm_118: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_342, permute_187);  view_342 = permute_187 = None
        view_343: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_118, [4, 512, 1024]);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_155: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_152, view_343);  add_152 = view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_411: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_155, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_69: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_411, 2)
        mean_68: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_69, [-1], True);  pow_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_156: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_68, 1e-06);  mean_68 = None
        rsqrt_68: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        mul_224: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_411, rsqrt_68);  convert_element_type_411 = rsqrt_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_412: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_224, torch.bfloat16);  mul_224 = None
        mul_225: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg190_1, convert_element_type_412);  arg190_1 = convert_element_type_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_344: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_225, [2048, 1024])
        permute_188: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        mm_119: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_344, permute_188);  view_344 = permute_188 = None
        view_345: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_119, [4, 512, 2048]);  mm_119 = None
        view_346: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_345, [4, 512, -1, 128]);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_415: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_346, torch.float32);  view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_70: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_415, 2)
        mean_69: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_70, [-1], True);  pow_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_157: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_69, 1e-06);  mean_69 = None
        rsqrt_69: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        mul_226: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_415, rsqrt_69);  convert_element_type_415 = rsqrt_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_416: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_226, torch.bfloat16);  mul_226 = None
        mul_227: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg192_1, convert_element_type_416);  arg192_1 = convert_element_type_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_189: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_227, [0, 2, 1, 3]);  mul_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_82: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_230: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_189, unsqueeze_82)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_73: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_189, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_51: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_73);  slice_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_72: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_189, 3, 0, 64);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_35: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_51, slice_72], -1);  neg_51 = slice_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_83: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_231: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_35, unsqueeze_83);  cat_35 = None
        add_159: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_347: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_225, [2048, 1024])
        permute_190: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        mm_120: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_347, permute_190);  view_347 = permute_190 = None
        view_348: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_120, [4, 512, 1024]);  mm_120 = None
        view_349: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_348, [4, 512, -1, 128]);  view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_419: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_71: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_419, 2)
        mean_70: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_71, [-1], True);  pow_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_158: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_70, 1e-06);  mean_70 = None
        rsqrt_70: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        mul_228: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_419, rsqrt_70);  convert_element_type_419 = rsqrt_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_420: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_228, torch.bfloat16);  mul_228 = None
        mul_229: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg194_1, convert_element_type_420);  arg194_1 = convert_element_type_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_191: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_229, [0, 2, 1, 3]);  mul_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_232: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_191, unsqueeze_82);  unsqueeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_75: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_191, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_52: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_75);  slice_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_74: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_191, 3, 0, 64);  permute_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_36: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_52, slice_74], -1);  neg_52 = slice_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_233: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_36, unsqueeze_83);  cat_36 = unsqueeze_83 = None
        add_160: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_84: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_160, 2);  add_160 = None
        expand_40: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_84, [4, 8, 2, 512, 128]);  unsqueeze_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_36: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_353: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_36, [4, 16, 512, 128]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_350: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_225, [2048, 1024]);  mul_225 = None
        permute_192: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        mm_121: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_350, permute_192);  view_350 = permute_192 = None
        view_351: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_121, [4, 512, 1024]);  mm_121 = None
        view_352: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_351, [4, 512, -1, 128]);  view_351 = None
        permute_193: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_85: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_193, 2);  permute_193 = None
        expand_41: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_85, [4, 8, 2, 512, 128]);  unsqueeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_37: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_354: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_37, [4, 16, 512, 128]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_36: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_36, full_default_35);  full_default_36 = full_default_35 = None
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_159, view_353, view_354, where_17, False, scale = 0.08838834764831845);  add_159 = view_353 = view_354 = where_17 = None
        getitem_153: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_17[0];  _scaled_dot_product_cudnn_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_194: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_355: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_194, [4, 512, -1]);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_356: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_355, [2048, 2048]);  view_355 = None
        permute_195: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        mm_122: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_356, permute_195);  view_356 = permute_195 = None
        view_357: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_122, [4, 512, 1024]);  mm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_161: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_155, view_357);  add_155 = view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_425: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_161, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_72: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_425, 2)
        mean_71: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_72, [-1], True);  pow_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_162: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_71, 1e-06);  mean_71 = None
        rsqrt_71: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        mul_234: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_425, rsqrt_71);  convert_element_type_425 = rsqrt_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_426: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_234, torch.bfloat16);  mul_234 = None
        mul_235: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg197_1, convert_element_type_426);  arg197_1 = convert_element_type_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_358: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_235, [2048, 1024])
        permute_196: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        mm_123: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_358, permute_196);  view_358 = permute_196 = None
        view_359: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_123, [4, 512, 3072]);  mm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_429: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_359, torch.float32);  view_359 = None
        neg_53: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_429)
        exp_17: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_53);  neg_53 = None
        add_163: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        div_17: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_429, add_163);  convert_element_type_429 = add_163 = None
        convert_element_type_430: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_360: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_235, [2048, 1024]);  mul_235 = None
        permute_197: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        mm_124: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_360, permute_197);  view_360 = permute_197 = None
        view_361: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_124, [4, 512, 3072]);  mm_124 = None
        mul_236: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_430, view_361);  convert_element_type_430 = view_361 = None
        view_362: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_236, [2048, 3072]);  mul_236 = None
        permute_198: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        mm_125: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_362, permute_198);  view_362 = permute_198 = None
        view_363: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_125, [4, 512, 1024]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_164: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_161, view_363);  add_161 = view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_435: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_164, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_73: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_435, 2)
        mean_72: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_73, [-1], True);  pow_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_165: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_72, 1e-06);  mean_72 = None
        rsqrt_72: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        mul_237: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_435, rsqrt_72);  convert_element_type_435 = rsqrt_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_436: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_237, torch.bfloat16);  mul_237 = None
        mul_238: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg201_1, convert_element_type_436);  arg201_1 = convert_element_type_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_364: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_238, [2048, 1024])
        permute_199: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        mm_126: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_364, permute_199);  view_364 = permute_199 = None
        view_365: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_126, [4, 512, 2048]);  mm_126 = None
        view_366: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_365, [4, 512, -1, 128]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_439: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_366, torch.float32);  view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_74: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_439, 2)
        mean_73: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_74, [-1], True);  pow_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_166: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_73, 1e-06);  mean_73 = None
        rsqrt_73: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        mul_239: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_439, rsqrt_73);  convert_element_type_439 = rsqrt_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_440: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_239, torch.bfloat16);  mul_239 = None
        mul_240: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg203_1, convert_element_type_440);  arg203_1 = convert_element_type_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_200: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_240, [0, 2, 1, 3]);  mul_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_86: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_243: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_200, unsqueeze_86)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_77: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_200, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_54: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_77);  slice_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_76: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_200, 3, 0, 64);  permute_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_37: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_54, slice_76], -1);  neg_54 = slice_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_87: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_244: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_37, unsqueeze_87);  cat_37 = None
        add_168: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_367: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_238, [2048, 1024])
        permute_201: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        mm_127: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_367, permute_201);  view_367 = permute_201 = None
        view_368: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_127, [4, 512, 1024]);  mm_127 = None
        view_369: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_368, [4, 512, -1, 128]);  view_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_443: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_369, torch.float32);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_75: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_443, 2)
        mean_74: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_75, [-1], True);  pow_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_167: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_74, 1e-06);  mean_74 = None
        rsqrt_74: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
        mul_241: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_443, rsqrt_74);  convert_element_type_443 = rsqrt_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_444: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_241, torch.bfloat16);  mul_241 = None
        mul_242: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg205_1, convert_element_type_444);  arg205_1 = convert_element_type_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_202: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_242, [0, 2, 1, 3]);  mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_245: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_202, unsqueeze_86);  unsqueeze_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_79: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_202, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_55: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_79);  slice_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_78: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_202, 3, 0, 64);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_38: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_55, slice_78], -1);  neg_55 = slice_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_246: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_38, unsqueeze_87);  cat_38 = unsqueeze_87 = None
        add_169: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_88: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_169, 2);  add_169 = None
        expand_42: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_88, [4, 8, 2, 512, 128]);  unsqueeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_38: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_373: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_38, [4, 16, 512, 128]);  clone_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_370: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_238, [2048, 1024]);  mul_238 = None
        permute_203: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        mm_128: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_370, permute_203);  view_370 = permute_203 = None
        view_371: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_128, [4, 512, 1024]);  mm_128 = None
        view_372: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_371, [4, 512, -1, 128]);  view_371 = None
        permute_204: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_89: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_204, 2);  permute_204 = None
        expand_43: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_89, [4, 8, 2, 512, 128]);  unsqueeze_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_39: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_374: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_39, [4, 16, 512, 128]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_38: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_38, full_default_37);  full_default_38 = full_default_37 = None
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_168, view_373, view_374, where_18, False, scale = 0.08838834764831845);  add_168 = view_373 = view_374 = where_18 = None
        getitem_162: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_18[0];  _scaled_dot_product_cudnn_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_205: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_375: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_205, [4, 512, -1]);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_376: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_375, [2048, 2048]);  view_375 = None
        permute_206: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        mm_129: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_376, permute_206);  view_376 = permute_206 = None
        view_377: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_129, [4, 512, 1024]);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_170: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_164, view_377);  add_164 = view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_449: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_170, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_76: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_449, 2)
        mean_75: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_76, [-1], True);  pow_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_171: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_75, 1e-06);  mean_75 = None
        rsqrt_75: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        mul_247: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_449, rsqrt_75);  convert_element_type_449 = rsqrt_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_450: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_247, torch.bfloat16);  mul_247 = None
        mul_248: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg208_1, convert_element_type_450);  arg208_1 = convert_element_type_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_378: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_248, [2048, 1024])
        permute_207: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        mm_130: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_378, permute_207);  view_378 = permute_207 = None
        view_379: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_130, [4, 512, 3072]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_453: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_379, torch.float32);  view_379 = None
        neg_56: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_453)
        exp_18: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_56);  neg_56 = None
        add_172: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        div_18: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_453, add_172);  convert_element_type_453 = add_172 = None
        convert_element_type_454: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_380: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_248, [2048, 1024]);  mul_248 = None
        permute_208: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        mm_131: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_380, permute_208);  view_380 = permute_208 = None
        view_381: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_131, [4, 512, 3072]);  mm_131 = None
        mul_249: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_454, view_381);  convert_element_type_454 = view_381 = None
        view_382: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_249, [2048, 3072]);  mul_249 = None
        permute_209: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        mm_132: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_382, permute_209);  view_382 = permute_209 = None
        view_383: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_132, [4, 512, 1024]);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_173: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_170, view_383);  add_170 = view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_459: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_173, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_77: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_459, 2)
        mean_76: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_77, [-1], True);  pow_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_174: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_76, 1e-06);  mean_76 = None
        rsqrt_76: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_250: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_459, rsqrt_76);  convert_element_type_459 = rsqrt_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_460: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_250, torch.bfloat16);  mul_250 = None
        mul_251: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg212_1, convert_element_type_460);  arg212_1 = convert_element_type_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_384: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_251, [2048, 1024])
        permute_210: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        mm_133: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_384, permute_210);  view_384 = permute_210 = None
        view_385: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_133, [4, 512, 2048]);  mm_133 = None
        view_386: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_385, [4, 512, -1, 128]);  view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_463: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_386, torch.float32);  view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_78: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_463, 2)
        mean_77: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_78, [-1], True);  pow_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_175: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_77, 1e-06);  mean_77 = None
        rsqrt_77: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        mul_252: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_463, rsqrt_77);  convert_element_type_463 = rsqrt_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_464: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_252, torch.bfloat16);  mul_252 = None
        mul_253: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg214_1, convert_element_type_464);  arg214_1 = convert_element_type_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_211: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_253, [0, 2, 1, 3]);  mul_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_90: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_256: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_211, unsqueeze_90)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_81: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_211, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_57: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_81);  slice_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_80: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_211, 3, 0, 64);  permute_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_39: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_57, slice_80], -1);  neg_57 = slice_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_91: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_257: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_39, unsqueeze_91);  cat_39 = None
        add_177: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_387: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_251, [2048, 1024])
        permute_212: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        mm_134: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_387, permute_212);  view_387 = permute_212 = None
        view_388: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_134, [4, 512, 1024]);  mm_134 = None
        view_389: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_388, [4, 512, -1, 128]);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_467: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_389, torch.float32);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_79: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_467, 2)
        mean_78: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_79, [-1], True);  pow_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_176: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_78, 1e-06);  mean_78 = None
        rsqrt_78: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        mul_254: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_467, rsqrt_78);  convert_element_type_467 = rsqrt_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_468: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_254, torch.bfloat16);  mul_254 = None
        mul_255: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg216_1, convert_element_type_468);  arg216_1 = convert_element_type_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_213: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_255, [0, 2, 1, 3]);  mul_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_258: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_213, unsqueeze_90);  unsqueeze_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_83: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_213, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_58: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_83);  slice_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_82: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_213, 3, 0, 64);  permute_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_40: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_58, slice_82], -1);  neg_58 = slice_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_259: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_40, unsqueeze_91);  cat_40 = unsqueeze_91 = None
        add_178: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_92: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_178, 2);  add_178 = None
        expand_44: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_92, [4, 8, 2, 512, 128]);  unsqueeze_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_40: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_393: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_40, [4, 16, 512, 128]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_390: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_251, [2048, 1024]);  mul_251 = None
        permute_214: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        mm_135: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_390, permute_214);  view_390 = permute_214 = None
        view_391: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_135, [4, 512, 1024]);  mm_135 = None
        view_392: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_391, [4, 512, -1, 128]);  view_391 = None
        permute_215: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_93: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_215, 2);  permute_215 = None
        expand_45: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_93, [4, 8, 2, 512, 128]);  unsqueeze_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_41: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_394: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_41, [4, 16, 512, 128]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_40: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_40, full_default_39);  full_default_40 = full_default_39 = None
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_177, view_393, view_394, where_19, False, scale = 0.08838834764831845);  add_177 = view_393 = view_394 = where_19 = None
        getitem_171: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_19[0];  _scaled_dot_product_cudnn_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_216: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_395: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_216, [4, 512, -1]);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_396: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_395, [2048, 2048]);  view_395 = None
        permute_217: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        mm_136: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_396, permute_217);  view_396 = permute_217 = None
        view_397: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_136, [4, 512, 1024]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_179: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_173, view_397);  add_173 = view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_473: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_179, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_80: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_473, 2)
        mean_79: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_80, [-1], True);  pow_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_180: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_79, 1e-06);  mean_79 = None
        rsqrt_79: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        mul_260: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_473, rsqrt_79);  convert_element_type_473 = rsqrt_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_474: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_260, torch.bfloat16);  mul_260 = None
        mul_261: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg219_1, convert_element_type_474);  arg219_1 = convert_element_type_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_398: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_261, [2048, 1024])
        permute_218: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        mm_137: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_398, permute_218);  view_398 = permute_218 = None
        view_399: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_137, [4, 512, 3072]);  mm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_477: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_399, torch.float32);  view_399 = None
        neg_59: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_477)
        exp_19: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_59);  neg_59 = None
        add_181: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_477, add_181);  convert_element_type_477 = add_181 = None
        convert_element_type_478: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_400: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_261, [2048, 1024]);  mul_261 = None
        permute_219: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        mm_138: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_400, permute_219);  view_400 = permute_219 = None
        view_401: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_138, [4, 512, 3072]);  mm_138 = None
        mul_262: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_478, view_401);  convert_element_type_478 = view_401 = None
        view_402: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_262, [2048, 3072]);  mul_262 = None
        permute_220: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        mm_139: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_402, permute_220);  view_402 = permute_220 = None
        view_403: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_139, [4, 512, 1024]);  mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_182: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_179, view_403);  add_179 = view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_483: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_182, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_81: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_483, 2)
        mean_80: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_81, [-1], True);  pow_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_183: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_80, 1e-06);  mean_80 = None
        rsqrt_80: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        mul_263: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_483, rsqrt_80);  convert_element_type_483 = rsqrt_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_484: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_263, torch.bfloat16);  mul_263 = None
        mul_264: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg223_1, convert_element_type_484);  arg223_1 = convert_element_type_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_404: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_264, [2048, 1024])
        permute_221: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        mm_140: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_404, permute_221);  view_404 = permute_221 = None
        view_405: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_140, [4, 512, 2048]);  mm_140 = None
        view_406: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_405, [4, 512, -1, 128]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_487: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_406, torch.float32);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_82: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_487, 2)
        mean_81: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_82, [-1], True);  pow_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_184: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_81, 1e-06);  mean_81 = None
        rsqrt_81: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        mul_265: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_487, rsqrt_81);  convert_element_type_487 = rsqrt_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_488: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_265, torch.bfloat16);  mul_265 = None
        mul_266: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg225_1, convert_element_type_488);  arg225_1 = convert_element_type_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_222: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_266, [0, 2, 1, 3]);  mul_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_94: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_269: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_222, unsqueeze_94)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_85: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_222, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_60: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_85);  slice_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_84: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_222, 3, 0, 64);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_41: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_60, slice_84], -1);  neg_60 = slice_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_95: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_270: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_41, unsqueeze_95);  cat_41 = None
        add_186: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_407: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_264, [2048, 1024])
        permute_223: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        mm_141: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_407, permute_223);  view_407 = permute_223 = None
        view_408: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_141, [4, 512, 1024]);  mm_141 = None
        view_409: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_408, [4, 512, -1, 128]);  view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_491: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_409, torch.float32);  view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_83: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_491, 2)
        mean_82: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_83, [-1], True);  pow_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_185: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_82, 1e-06);  mean_82 = None
        rsqrt_82: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        mul_267: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_491, rsqrt_82);  convert_element_type_491 = rsqrt_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_492: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_267, torch.bfloat16);  mul_267 = None
        mul_268: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg227_1, convert_element_type_492);  arg227_1 = convert_element_type_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_224: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_268, [0, 2, 1, 3]);  mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_271: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_224, unsqueeze_94);  unsqueeze_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_87: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_224, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_61: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_87);  slice_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_86: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_224, 3, 0, 64);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_42: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_61, slice_86], -1);  neg_61 = slice_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_272: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_42, unsqueeze_95);  cat_42 = unsqueeze_95 = None
        add_187: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_96: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_187, 2);  add_187 = None
        expand_46: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_96, [4, 8, 2, 512, 128]);  unsqueeze_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_42: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_413: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_42, [4, 16, 512, 128]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_410: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_264, [2048, 1024]);  mul_264 = None
        permute_225: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        mm_142: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_410, permute_225);  view_410 = permute_225 = None
        view_411: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_142, [4, 512, 1024]);  mm_142 = None
        view_412: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_411, [4, 512, -1, 128]);  view_411 = None
        permute_226: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_97: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_226, 2);  permute_226 = None
        expand_47: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_97, [4, 8, 2, 512, 128]);  unsqueeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_43: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_414: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_43, [4, 16, 512, 128]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_42: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_42, full_default_41);  full_default_42 = full_default_41 = None
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_186, view_413, view_414, where_20, False, scale = 0.08838834764831845);  add_186 = view_413 = view_414 = where_20 = None
        getitem_180: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_20[0];  _scaled_dot_product_cudnn_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_227: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_415: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_227, [4, 512, -1]);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_416: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_415, [2048, 2048]);  view_415 = None
        permute_228: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        mm_143: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_416, permute_228);  view_416 = permute_228 = None
        view_417: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_143, [4, 512, 1024]);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_188: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_182, view_417);  add_182 = view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_497: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_188, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_84: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_497, 2)
        mean_83: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_84, [-1], True);  pow_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_189: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_83, 1e-06);  mean_83 = None
        rsqrt_83: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
        mul_273: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_497, rsqrt_83);  convert_element_type_497 = rsqrt_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_498: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_273, torch.bfloat16);  mul_273 = None
        mul_274: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg230_1, convert_element_type_498);  arg230_1 = convert_element_type_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_418: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_274, [2048, 1024])
        permute_229: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        mm_144: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_418, permute_229);  view_418 = permute_229 = None
        view_419: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_144, [4, 512, 3072]);  mm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_501: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None
        neg_62: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_501)
        exp_20: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_62);  neg_62 = None
        add_190: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        div_20: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_501, add_190);  convert_element_type_501 = add_190 = None
        convert_element_type_502: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_420: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_274, [2048, 1024]);  mul_274 = None
        permute_230: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        mm_145: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_420, permute_230);  view_420 = permute_230 = None
        view_421: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_145, [4, 512, 3072]);  mm_145 = None
        mul_275: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_502, view_421);  convert_element_type_502 = view_421 = None
        view_422: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_275, [2048, 3072]);  mul_275 = None
        permute_231: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        mm_146: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_422, permute_231);  view_422 = permute_231 = None
        view_423: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_146, [4, 512, 1024]);  mm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_191: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_188, view_423);  add_188 = view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_507: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_191, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_85: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_507, 2)
        mean_84: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_85, [-1], True);  pow_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_192: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_84, 1e-06);  mean_84 = None
        rsqrt_84: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_276: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_507, rsqrt_84);  convert_element_type_507 = rsqrt_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_508: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_276, torch.bfloat16);  mul_276 = None
        mul_277: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg234_1, convert_element_type_508);  arg234_1 = convert_element_type_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_424: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_277, [2048, 1024])
        permute_232: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        mm_147: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_424, permute_232);  view_424 = permute_232 = None
        view_425: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_147, [4, 512, 2048]);  mm_147 = None
        view_426: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_425, [4, 512, -1, 128]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_511: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_426, torch.float32);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_86: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_511, 2)
        mean_85: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_86, [-1], True);  pow_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_193: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_85, 1e-06);  mean_85 = None
        rsqrt_85: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
        mul_278: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_511, rsqrt_85);  convert_element_type_511 = rsqrt_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_512: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_278, torch.bfloat16);  mul_278 = None
        mul_279: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg236_1, convert_element_type_512);  arg236_1 = convert_element_type_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_233: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_279, [0, 2, 1, 3]);  mul_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_98: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_282: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_233, unsqueeze_98)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_89: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_233, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_63: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_89);  slice_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_88: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_233, 3, 0, 64);  permute_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_43: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_63, slice_88], -1);  neg_63 = slice_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_99: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_283: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_43, unsqueeze_99);  cat_43 = None
        add_195: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_427: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_277, [2048, 1024])
        permute_234: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        mm_148: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_427, permute_234);  view_427 = permute_234 = None
        view_428: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_148, [4, 512, 1024]);  mm_148 = None
        view_429: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_428, [4, 512, -1, 128]);  view_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_515: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_429, torch.float32);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_87: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_515, 2)
        mean_86: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_87, [-1], True);  pow_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_194: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_86, 1e-06);  mean_86 = None
        rsqrt_86: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        mul_280: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_515, rsqrt_86);  convert_element_type_515 = rsqrt_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_516: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_280, torch.bfloat16);  mul_280 = None
        mul_281: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg238_1, convert_element_type_516);  arg238_1 = convert_element_type_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_235: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_281, [0, 2, 1, 3]);  mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_284: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_235, unsqueeze_98);  unsqueeze_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_91: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_235, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_64: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_91);  slice_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_90: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_235, 3, 0, 64);  permute_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_44: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_64, slice_90], -1);  neg_64 = slice_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_285: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_44, unsqueeze_99);  cat_44 = unsqueeze_99 = None
        add_196: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_100: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_196, 2);  add_196 = None
        expand_48: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_100, [4, 8, 2, 512, 128]);  unsqueeze_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_44: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_433: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_44, [4, 16, 512, 128]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_430: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_277, [2048, 1024]);  mul_277 = None
        permute_236: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        mm_149: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_430, permute_236);  view_430 = permute_236 = None
        view_431: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_149, [4, 512, 1024]);  mm_149 = None
        view_432: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_431, [4, 512, -1, 128]);  view_431 = None
        permute_237: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_101: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_237, 2);  permute_237 = None
        expand_49: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_101, [4, 8, 2, 512, 128]);  unsqueeze_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_45: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_434: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_45, [4, 16, 512, 128]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_44: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_44, full_default_43);  full_default_44 = full_default_43 = None
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_195, view_433, view_434, where_21, False, scale = 0.08838834764831845);  add_195 = view_433 = view_434 = where_21 = None
        getitem_189: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_21[0];  _scaled_dot_product_cudnn_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_238: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_435: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_238, [4, 512, -1]);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_436: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_435, [2048, 2048]);  view_435 = None
        permute_239: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        mm_150: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_436, permute_239);  view_436 = permute_239 = None
        view_437: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_150, [4, 512, 1024]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_197: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_191, view_437);  add_191 = view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_521: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_197, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_88: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_521, 2)
        mean_87: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_88, [-1], True);  pow_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_198: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_87, 1e-06);  mean_87 = None
        rsqrt_87: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        mul_286: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_521, rsqrt_87);  convert_element_type_521 = rsqrt_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_522: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_286, torch.bfloat16);  mul_286 = None
        mul_287: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg241_1, convert_element_type_522);  arg241_1 = convert_element_type_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_438: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_287, [2048, 1024])
        permute_240: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        mm_151: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_438, permute_240);  view_438 = permute_240 = None
        view_439: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_151, [4, 512, 3072]);  mm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_525: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_439, torch.float32);  view_439 = None
        neg_65: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_525)
        exp_21: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_65);  neg_65 = None
        add_199: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        div_21: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_525, add_199);  convert_element_type_525 = add_199 = None
        convert_element_type_526: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_440: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_287, [2048, 1024]);  mul_287 = None
        permute_241: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        mm_152: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_440, permute_241);  view_440 = permute_241 = None
        view_441: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_152, [4, 512, 3072]);  mm_152 = None
        mul_288: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_526, view_441);  convert_element_type_526 = view_441 = None
        view_442: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_288, [2048, 3072]);  mul_288 = None
        permute_242: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        mm_153: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_442, permute_242);  view_442 = permute_242 = None
        view_443: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_153, [4, 512, 1024]);  mm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_200: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_197, view_443);  add_197 = view_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_531: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_200, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_89: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_531, 2)
        mean_88: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_89, [-1], True);  pow_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_201: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_88, 1e-06);  mean_88 = None
        rsqrt_88: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        mul_289: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_531, rsqrt_88);  convert_element_type_531 = rsqrt_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_532: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_289, torch.bfloat16);  mul_289 = None
        mul_290: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg245_1, convert_element_type_532);  arg245_1 = convert_element_type_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_444: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_290, [2048, 1024])
        permute_243: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        mm_154: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_444, permute_243);  view_444 = permute_243 = None
        view_445: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_154, [4, 512, 2048]);  mm_154 = None
        view_446: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_445, [4, 512, -1, 128]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_535: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_446, torch.float32);  view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_90: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_535, 2)
        mean_89: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_90, [-1], True);  pow_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_202: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_89, 1e-06);  mean_89 = None
        rsqrt_89: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        mul_291: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_535, rsqrt_89);  convert_element_type_535 = rsqrt_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_536: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_291, torch.bfloat16);  mul_291 = None
        mul_292: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg247_1, convert_element_type_536);  arg247_1 = convert_element_type_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_244: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_292, [0, 2, 1, 3]);  mul_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_102: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_295: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_244, unsqueeze_102)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_93: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_244, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_66: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_93);  slice_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_92: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_244, 3, 0, 64);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_45: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_66, slice_92], -1);  neg_66 = slice_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_103: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_296: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_45, unsqueeze_103);  cat_45 = None
        add_204: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_447: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_290, [2048, 1024])
        permute_245: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        mm_155: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_447, permute_245);  view_447 = permute_245 = None
        view_448: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_155, [4, 512, 1024]);  mm_155 = None
        view_449: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_448, [4, 512, -1, 128]);  view_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_539: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_449, torch.float32);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_91: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_539, 2)
        mean_90: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_91, [-1], True);  pow_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_203: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_90, 1e-06);  mean_90 = None
        rsqrt_90: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        mul_293: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_539, rsqrt_90);  convert_element_type_539 = rsqrt_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_540: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_293, torch.bfloat16);  mul_293 = None
        mul_294: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg249_1, convert_element_type_540);  arg249_1 = convert_element_type_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_246: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_294, [0, 2, 1, 3]);  mul_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_297: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_246, unsqueeze_102);  unsqueeze_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_95: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_246, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_67: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_95);  slice_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_94: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_246, 3, 0, 64);  permute_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_46: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_67, slice_94], -1);  neg_67 = slice_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_298: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_46, unsqueeze_103);  cat_46 = unsqueeze_103 = None
        add_205: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_104: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_205, 2);  add_205 = None
        expand_50: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_104, [4, 8, 2, 512, 128]);  unsqueeze_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_46: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_453: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_46, [4, 16, 512, 128]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_450: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_290, [2048, 1024]);  mul_290 = None
        permute_247: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        mm_156: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_450, permute_247);  view_450 = permute_247 = None
        view_451: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_156, [4, 512, 1024]);  mm_156 = None
        view_452: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_451, [4, 512, -1, 128]);  view_451 = None
        permute_248: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_105: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_248, 2);  permute_248 = None
        expand_51: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_105, [4, 8, 2, 512, 128]);  unsqueeze_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_47: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_454: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_47, [4, 16, 512, 128]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_46: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_46, full_default_45);  full_default_46 = full_default_45 = None
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_204, view_453, view_454, where_22, False, scale = 0.08838834764831845);  add_204 = view_453 = view_454 = where_22 = None
        getitem_198: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_22[0];  _scaled_dot_product_cudnn_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_249: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3]);  getitem_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_455: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_249, [4, 512, -1]);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_456: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_455, [2048, 2048]);  view_455 = None
        permute_250: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        mm_157: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_456, permute_250);  view_456 = permute_250 = None
        view_457: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_157, [4, 512, 1024]);  mm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_206: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_200, view_457);  add_200 = view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_545: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_206, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_92: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_545, 2)
        mean_91: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_92, [-1], True);  pow_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_207: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_91, 1e-06);  mean_91 = None
        rsqrt_91: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        mul_299: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_545, rsqrt_91);  convert_element_type_545 = rsqrt_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_546: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_299, torch.bfloat16);  mul_299 = None
        mul_300: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg252_1, convert_element_type_546);  arg252_1 = convert_element_type_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_458: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_300, [2048, 1024])
        permute_251: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        mm_158: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_458, permute_251);  view_458 = permute_251 = None
        view_459: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_158, [4, 512, 3072]);  mm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_549: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        neg_68: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_549)
        exp_22: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_68);  neg_68 = None
        add_208: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_549, add_208);  convert_element_type_549 = add_208 = None
        convert_element_type_550: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_460: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_300, [2048, 1024]);  mul_300 = None
        permute_252: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        mm_159: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_460, permute_252);  view_460 = permute_252 = None
        view_461: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_159, [4, 512, 3072]);  mm_159 = None
        mul_301: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_550, view_461);  convert_element_type_550 = view_461 = None
        view_462: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_301, [2048, 3072]);  mul_301 = None
        permute_253: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        mm_160: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_462, permute_253);  view_462 = permute_253 = None
        view_463: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_160, [4, 512, 1024]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_209: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_206, view_463);  add_206 = view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_555: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_209, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_93: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_555, 2)
        mean_92: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_93, [-1], True);  pow_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_210: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_92, 1e-06);  mean_92 = None
        rsqrt_92: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        mul_302: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_555, rsqrt_92);  convert_element_type_555 = rsqrt_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_556: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_302, torch.bfloat16);  mul_302 = None
        mul_303: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg256_1, convert_element_type_556);  arg256_1 = convert_element_type_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_464: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_303, [2048, 1024])
        permute_254: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        mm_161: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_464, permute_254);  view_464 = permute_254 = None
        view_465: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_161, [4, 512, 2048]);  mm_161 = None
        view_466: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_465, [4, 512, -1, 128]);  view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_559: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_466, torch.float32);  view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_94: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_559, 2)
        mean_93: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_94, [-1], True);  pow_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_211: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_93, 1e-06);  mean_93 = None
        rsqrt_93: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        mul_304: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_559, rsqrt_93);  convert_element_type_559 = rsqrt_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_560: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_304, torch.bfloat16);  mul_304 = None
        mul_305: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg258_1, convert_element_type_560);  arg258_1 = convert_element_type_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_255: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_305, [0, 2, 1, 3]);  mul_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_106: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_308: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_255, unsqueeze_106)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_97: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_255, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_69: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_97);  slice_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_96: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_255, 3, 0, 64);  permute_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_47: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_69, slice_96], -1);  neg_69 = slice_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_107: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_309: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_47, unsqueeze_107);  cat_47 = None
        add_213: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_308, mul_309);  mul_308 = mul_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_467: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_303, [2048, 1024])
        permute_256: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        mm_162: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_467, permute_256);  view_467 = permute_256 = None
        view_468: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_162, [4, 512, 1024]);  mm_162 = None
        view_469: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_468, [4, 512, -1, 128]);  view_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_563: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_469, torch.float32);  view_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_95: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_563, 2)
        mean_94: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_95, [-1], True);  pow_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_212: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_94, 1e-06);  mean_94 = None
        rsqrt_94: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        mul_306: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_563, rsqrt_94);  convert_element_type_563 = rsqrt_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_564: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_306, torch.bfloat16);  mul_306 = None
        mul_307: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg260_1, convert_element_type_564);  arg260_1 = convert_element_type_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_257: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_307, [0, 2, 1, 3]);  mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_310: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_257, unsqueeze_106);  unsqueeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_99: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_257, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_70: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_99);  slice_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_98: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_257, 3, 0, 64);  permute_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_48: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_70, slice_98], -1);  neg_70 = slice_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_311: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_48, unsqueeze_107);  cat_48 = unsqueeze_107 = None
        add_214: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_310, mul_311);  mul_310 = mul_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_108: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_214, 2);  add_214 = None
        expand_52: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_108, [4, 8, 2, 512, 128]);  unsqueeze_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_48: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_473: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_48, [4, 16, 512, 128]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_470: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_303, [2048, 1024]);  mul_303 = None
        permute_258: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        mm_163: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_470, permute_258);  view_470 = permute_258 = None
        view_471: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_163, [4, 512, 1024]);  mm_163 = None
        view_472: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_471, [4, 512, -1, 128]);  view_471 = None
        permute_259: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_109: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_259, 2);  permute_259 = None
        expand_53: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_109, [4, 8, 2, 512, 128]);  unsqueeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_49: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_474: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_49, [4, 16, 512, 128]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_48: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_48, full_default_47);  full_default_48 = full_default_47 = None
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_213, view_473, view_474, where_23, False, scale = 0.08838834764831845);  add_213 = view_473 = view_474 = where_23 = None
        getitem_207: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_23[0];  _scaled_dot_product_cudnn_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_260: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_475: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_260, [4, 512, -1]);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_476: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_475, [2048, 2048]);  view_475 = None
        permute_261: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        mm_164: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_476, permute_261);  view_476 = permute_261 = None
        view_477: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_164, [4, 512, 1024]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_215: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_209, view_477);  add_209 = view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_569: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_215, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_96: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_569, 2)
        mean_95: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_96, [-1], True);  pow_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_216: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_95, 1e-06);  mean_95 = None
        rsqrt_95: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        mul_312: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_569, rsqrt_95);  convert_element_type_569 = rsqrt_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_570: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_312, torch.bfloat16);  mul_312 = None
        mul_313: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg263_1, convert_element_type_570);  arg263_1 = convert_element_type_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_478: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_313, [2048, 1024])
        permute_262: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        mm_165: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_478, permute_262);  view_478 = permute_262 = None
        view_479: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_165, [4, 512, 3072]);  mm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_573: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_479, torch.float32);  view_479 = None
        neg_71: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_573)
        exp_23: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_71);  neg_71 = None
        add_217: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        div_23: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_573, add_217);  convert_element_type_573 = add_217 = None
        convert_element_type_574: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_480: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_313, [2048, 1024]);  mul_313 = None
        permute_263: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        mm_166: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_480, permute_263);  view_480 = permute_263 = None
        view_481: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_166, [4, 512, 3072]);  mm_166 = None
        mul_314: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_574, view_481);  convert_element_type_574 = view_481 = None
        view_482: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_314, [2048, 3072]);  mul_314 = None
        permute_264: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        mm_167: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_482, permute_264);  view_482 = permute_264 = None
        view_483: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_167, [4, 512, 1024]);  mm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_218: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_215, view_483);  add_215 = view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_579: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_218, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_97: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_579, 2)
        mean_96: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_97, [-1], True);  pow_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_219: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_96, 1e-06);  mean_96 = None
        rsqrt_96: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        mul_315: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_579, rsqrt_96);  convert_element_type_579 = rsqrt_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_580: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_315, torch.bfloat16);  mul_315 = None
        mul_316: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg267_1, convert_element_type_580);  arg267_1 = convert_element_type_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_484: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_316, [2048, 1024])
        permute_265: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        mm_168: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_484, permute_265);  view_484 = permute_265 = None
        view_485: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_168, [4, 512, 2048]);  mm_168 = None
        view_486: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_485, [4, 512, -1, 128]);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_583: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_486, torch.float32);  view_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_98: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_583, 2)
        mean_97: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_98, [-1], True);  pow_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_220: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_97, 1e-06);  mean_97 = None
        rsqrt_97: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        mul_317: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_583, rsqrt_97);  convert_element_type_583 = rsqrt_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_584: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_317, torch.bfloat16);  mul_317 = None
        mul_318: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg269_1, convert_element_type_584);  arg269_1 = convert_element_type_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_266: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_318, [0, 2, 1, 3]);  mul_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_110: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_321: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_266, unsqueeze_110)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_101: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_266, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_72: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_101);  slice_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_100: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_266, 3, 0, 64);  permute_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_49: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_72, slice_100], -1);  neg_72 = slice_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_111: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_322: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_49, unsqueeze_111);  cat_49 = None
        add_222: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_487: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_316, [2048, 1024])
        permute_267: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        mm_169: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_487, permute_267);  view_487 = permute_267 = None
        view_488: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_169, [4, 512, 1024]);  mm_169 = None
        view_489: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_488, [4, 512, -1, 128]);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_587: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_489, torch.float32);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_99: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_587, 2)
        mean_98: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_99, [-1], True);  pow_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_221: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_98, 1e-06);  mean_98 = None
        rsqrt_98: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
        mul_319: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_587, rsqrt_98);  convert_element_type_587 = rsqrt_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_588: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_319, torch.bfloat16);  mul_319 = None
        mul_320: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg271_1, convert_element_type_588);  arg271_1 = convert_element_type_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_268: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_320, [0, 2, 1, 3]);  mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_323: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_268, unsqueeze_110);  unsqueeze_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_103: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_268, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_73: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_103);  slice_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_102: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_268, 3, 0, 64);  permute_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_50: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_73, slice_102], -1);  neg_73 = slice_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_324: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_50, unsqueeze_111);  cat_50 = unsqueeze_111 = None
        add_223: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_112: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_223, 2);  add_223 = None
        expand_54: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_112, [4, 8, 2, 512, 128]);  unsqueeze_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_50: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_493: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_50, [4, 16, 512, 128]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_490: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_316, [2048, 1024]);  mul_316 = None
        permute_269: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        mm_170: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_490, permute_269);  view_490 = permute_269 = None
        view_491: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_170, [4, 512, 1024]);  mm_170 = None
        view_492: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_491, [4, 512, -1, 128]);  view_491 = None
        permute_270: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_113: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_270, 2);  permute_270 = None
        expand_55: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_113, [4, 8, 2, 512, 128]);  unsqueeze_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_51: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_494: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_51, [4, 16, 512, 128]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_50: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_50, full_default_49);  full_default_50 = full_default_49 = None
        _scaled_dot_product_cudnn_attention_24 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_222, view_493, view_494, where_24, False, scale = 0.08838834764831845);  add_222 = view_493 = view_494 = where_24 = None
        getitem_216: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_24[0];  _scaled_dot_product_cudnn_attention_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_271: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_495: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_271, [4, 512, -1]);  permute_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_496: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_495, [2048, 2048]);  view_495 = None
        permute_272: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        mm_171: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_496, permute_272);  view_496 = permute_272 = None
        view_497: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_171, [4, 512, 1024]);  mm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_224: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_218, view_497);  add_218 = view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_593: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_224, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_100: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_593, 2)
        mean_99: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_100, [-1], True);  pow_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_225: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_99, 1e-06);  mean_99 = None
        rsqrt_99: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        mul_325: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_593, rsqrt_99);  convert_element_type_593 = rsqrt_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_594: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_325, torch.bfloat16);  mul_325 = None
        mul_326: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg274_1, convert_element_type_594);  arg274_1 = convert_element_type_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_498: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_326, [2048, 1024])
        permute_273: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        mm_172: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_498, permute_273);  view_498 = permute_273 = None
        view_499: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_172, [4, 512, 3072]);  mm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_597: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_499, torch.float32);  view_499 = None
        neg_74: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_597)
        exp_24: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_74);  neg_74 = None
        add_226: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        div_24: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_597, add_226);  convert_element_type_597 = add_226 = None
        convert_element_type_598: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_500: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_326, [2048, 1024]);  mul_326 = None
        permute_274: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
        mm_173: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_500, permute_274);  view_500 = permute_274 = None
        view_501: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_173, [4, 512, 3072]);  mm_173 = None
        mul_327: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_598, view_501);  convert_element_type_598 = view_501 = None
        view_502: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_327, [2048, 3072]);  mul_327 = None
        permute_275: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        mm_174: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_502, permute_275);  view_502 = permute_275 = None
        view_503: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_174, [4, 512, 1024]);  mm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_227: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_224, view_503);  add_224 = view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_603: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_227, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_101: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_603, 2)
        mean_100: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_101, [-1], True);  pow_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_228: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_100, 1e-06);  mean_100 = None
        rsqrt_100: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
        mul_328: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_603, rsqrt_100);  convert_element_type_603 = rsqrt_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_604: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_328, torch.bfloat16);  mul_328 = None
        mul_329: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg278_1, convert_element_type_604);  arg278_1 = convert_element_type_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_504: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_329, [2048, 1024])
        permute_276: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        mm_175: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_504, permute_276);  view_504 = permute_276 = None
        view_505: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_175, [4, 512, 2048]);  mm_175 = None
        view_506: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_505, [4, 512, -1, 128]);  view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_607: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_506, torch.float32);  view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_102: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_607, 2)
        mean_101: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_102, [-1], True);  pow_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_229: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_101, 1e-06);  mean_101 = None
        rsqrt_101: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        mul_330: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_607, rsqrt_101);  convert_element_type_607 = rsqrt_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_608: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_330, torch.bfloat16);  mul_330 = None
        mul_331: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg280_1, convert_element_type_608);  arg280_1 = convert_element_type_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_277: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_331, [0, 2, 1, 3]);  mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_114: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_334: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_277, unsqueeze_114)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_105: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_277, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_75: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_105);  slice_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_104: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_277, 3, 0, 64);  permute_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_51: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_75, slice_104], -1);  neg_75 = slice_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_115: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_335: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_51, unsqueeze_115);  cat_51 = None
        add_231: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_334, mul_335);  mul_334 = mul_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_507: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_329, [2048, 1024])
        permute_278: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        mm_176: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_507, permute_278);  view_507 = permute_278 = None
        view_508: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_176, [4, 512, 1024]);  mm_176 = None
        view_509: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_508, [4, 512, -1, 128]);  view_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_611: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_509, torch.float32);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_103: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_611, 2)
        mean_102: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_103, [-1], True);  pow_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_230: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_102, 1e-06);  mean_102 = None
        rsqrt_102: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
        mul_332: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_611, rsqrt_102);  convert_element_type_611 = rsqrt_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_612: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_332, torch.bfloat16);  mul_332 = None
        mul_333: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg282_1, convert_element_type_612);  arg282_1 = convert_element_type_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_279: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_333, [0, 2, 1, 3]);  mul_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_336: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_279, unsqueeze_114);  unsqueeze_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_107: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_279, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_76: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_107);  slice_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_106: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_279, 3, 0, 64);  permute_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_52: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_76, slice_106], -1);  neg_76 = slice_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_337: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_52, unsqueeze_115);  cat_52 = unsqueeze_115 = None
        add_232: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_116: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_232, 2);  add_232 = None
        expand_56: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_116, [4, 8, 2, 512, 128]);  unsqueeze_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_52: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_513: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_52, [4, 16, 512, 128]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_510: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_329, [2048, 1024]);  mul_329 = None
        permute_280: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        mm_177: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_510, permute_280);  view_510 = permute_280 = None
        view_511: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_177, [4, 512, 1024]);  mm_177 = None
        view_512: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_511, [4, 512, -1, 128]);  view_511 = None
        permute_281: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_117: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_281, 2);  permute_281 = None
        expand_57: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_117, [4, 8, 2, 512, 128]);  unsqueeze_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_53: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_514: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_53, [4, 16, 512, 128]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_52: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_52, full_default_51);  full_default_52 = full_default_51 = None
        _scaled_dot_product_cudnn_attention_25 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_231, view_513, view_514, where_25, False, scale = 0.08838834764831845);  add_231 = view_513 = view_514 = where_25 = None
        getitem_225: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_25[0];  _scaled_dot_product_cudnn_attention_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_282: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3]);  getitem_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_515: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_282, [4, 512, -1]);  permute_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_516: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_515, [2048, 2048]);  view_515 = None
        permute_283: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        mm_178: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_516, permute_283);  view_516 = permute_283 = None
        view_517: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_178, [4, 512, 1024]);  mm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_233: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_227, view_517);  add_227 = view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_617: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_233, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_104: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_617, 2)
        mean_103: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_104, [-1], True);  pow_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_234: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_103, 1e-06);  mean_103 = None
        rsqrt_103: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        mul_338: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_617, rsqrt_103);  convert_element_type_617 = rsqrt_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_618: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_338, torch.bfloat16);  mul_338 = None
        mul_339: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg285_1, convert_element_type_618);  arg285_1 = convert_element_type_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_518: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_339, [2048, 1024])
        permute_284: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        mm_179: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_518, permute_284);  view_518 = permute_284 = None
        view_519: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_179, [4, 512, 3072]);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_621: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_519, torch.float32);  view_519 = None
        neg_77: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_621)
        exp_25: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_77);  neg_77 = None
        add_235: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_621, add_235);  convert_element_type_621 = add_235 = None
        convert_element_type_622: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_520: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_339, [2048, 1024]);  mul_339 = None
        permute_285: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        mm_180: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_520, permute_285);  view_520 = permute_285 = None
        view_521: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_180, [4, 512, 3072]);  mm_180 = None
        mul_340: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_622, view_521);  convert_element_type_622 = view_521 = None
        view_522: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_340, [2048, 3072]);  mul_340 = None
        permute_286: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        mm_181: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_522, permute_286);  view_522 = permute_286 = None
        view_523: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_181, [4, 512, 1024]);  mm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_236: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_233, view_523);  add_233 = view_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_627: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_236, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_105: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_627, 2)
        mean_104: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_105, [-1], True);  pow_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_237: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_104, 1e-06);  mean_104 = None
        rsqrt_104: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
        mul_341: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_627, rsqrt_104);  convert_element_type_627 = rsqrt_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_628: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_341, torch.bfloat16);  mul_341 = None
        mul_342: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg289_1, convert_element_type_628);  arg289_1 = convert_element_type_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_524: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_342, [2048, 1024])
        permute_287: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        mm_182: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_524, permute_287);  view_524 = permute_287 = None
        view_525: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_182, [4, 512, 2048]);  mm_182 = None
        view_526: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_525, [4, 512, -1, 128]);  view_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_631: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_526, torch.float32);  view_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_106: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_631, 2)
        mean_105: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_106, [-1], True);  pow_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_238: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_105, 1e-06);  mean_105 = None
        rsqrt_105: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
        mul_343: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_631, rsqrt_105);  convert_element_type_631 = rsqrt_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_632: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_343, torch.bfloat16);  mul_343 = None
        mul_344: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg291_1, convert_element_type_632);  arg291_1 = convert_element_type_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_288: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_344, [0, 2, 1, 3]);  mul_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_118: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_347: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_288, unsqueeze_118)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_109: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_288, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_78: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_109);  slice_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_108: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_288, 3, 0, 64);  permute_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_53: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_78, slice_108], -1);  neg_78 = slice_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_119: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_348: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_53, unsqueeze_119);  cat_53 = None
        add_240: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_527: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_342, [2048, 1024])
        permute_289: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
        mm_183: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_527, permute_289);  view_527 = permute_289 = None
        view_528: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_183, [4, 512, 1024]);  mm_183 = None
        view_529: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_528, [4, 512, -1, 128]);  view_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_635: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_107: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_635, 2)
        mean_106: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_107, [-1], True);  pow_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_239: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_106, 1e-06);  mean_106 = None
        rsqrt_106: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        mul_345: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_635, rsqrt_106);  convert_element_type_635 = rsqrt_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_636: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_345, torch.bfloat16);  mul_345 = None
        mul_346: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg293_1, convert_element_type_636);  arg293_1 = convert_element_type_636 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_290: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_346, [0, 2, 1, 3]);  mul_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_349: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_290, unsqueeze_118);  unsqueeze_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_111: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_290, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_79: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_111);  slice_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_110: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_290, 3, 0, 64);  permute_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_54: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_79, slice_110], -1);  neg_79 = slice_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_350: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_54, unsqueeze_119);  cat_54 = unsqueeze_119 = None
        add_241: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_349, mul_350);  mul_349 = mul_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_120: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_241, 2);  add_241 = None
        expand_58: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_120, [4, 8, 2, 512, 128]);  unsqueeze_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_54: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_533: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_54, [4, 16, 512, 128]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_530: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_342, [2048, 1024]);  mul_342 = None
        permute_291: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        mm_184: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_530, permute_291);  view_530 = permute_291 = None
        view_531: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_184, [4, 512, 1024]);  mm_184 = None
        view_532: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_531, [4, 512, -1, 128]);  view_531 = None
        permute_292: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_121: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_292, 2);  permute_292 = None
        expand_59: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_121, [4, 8, 2, 512, 128]);  unsqueeze_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_55: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_534: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_55, [4, 16, 512, 128]);  clone_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_54: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_53: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_54, full_default_53);  full_default_54 = full_default_53 = None
        _scaled_dot_product_cudnn_attention_26 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_240, view_533, view_534, where_26, False, scale = 0.08838834764831845);  add_240 = view_533 = view_534 = where_26 = None
        getitem_234: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_26[0];  _scaled_dot_product_cudnn_attention_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_293: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_234, [0, 2, 1, 3]);  getitem_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_535: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_293, [4, 512, -1]);  permute_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_536: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_535, [2048, 2048]);  view_535 = None
        permute_294: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        mm_185: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_536, permute_294);  view_536 = permute_294 = None
        view_537: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_185, [4, 512, 1024]);  mm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_242: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_236, view_537);  add_236 = view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_641: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_242, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_108: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_641, 2)
        mean_107: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_108, [-1], True);  pow_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_243: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_107, 1e-06);  mean_107 = None
        rsqrt_107: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        mul_351: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_641, rsqrt_107);  convert_element_type_641 = rsqrt_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_642: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_351, torch.bfloat16);  mul_351 = None
        mul_352: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg296_1, convert_element_type_642);  arg296_1 = convert_element_type_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_538: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_352, [2048, 1024])
        permute_295: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        mm_186: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_538, permute_295);  view_538 = permute_295 = None
        view_539: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_186, [4, 512, 3072]);  mm_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_645: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_539, torch.float32);  view_539 = None
        neg_80: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_645)
        exp_26: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_80);  neg_80 = None
        add_244: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        div_26: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_645, add_244);  convert_element_type_645 = add_244 = None
        convert_element_type_646: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_540: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_352, [2048, 1024]);  mul_352 = None
        permute_296: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        mm_187: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_540, permute_296);  view_540 = permute_296 = None
        view_541: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_187, [4, 512, 3072]);  mm_187 = None
        mul_353: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_646, view_541);  convert_element_type_646 = view_541 = None
        view_542: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_353, [2048, 3072]);  mul_353 = None
        permute_297: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        mm_188: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_542, permute_297);  view_542 = permute_297 = None
        view_543: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_188, [4, 512, 1024]);  mm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_245: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_242, view_543);  add_242 = view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_651: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_245, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_109: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_651, 2)
        mean_108: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_109, [-1], True);  pow_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_246: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_108, 1e-06);  mean_108 = None
        rsqrt_108: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        mul_354: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_651, rsqrt_108);  convert_element_type_651 = rsqrt_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_652: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_354, torch.bfloat16);  mul_354 = None
        mul_355: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg300_1, convert_element_type_652);  arg300_1 = convert_element_type_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_544: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_355, [2048, 1024])
        permute_298: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        mm_189: "bf16[2048, 2048]" = torch.ops.aten.mm.default(view_544, permute_298);  view_544 = permute_298 = None
        view_545: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_189, [4, 512, 2048]);  mm_189 = None
        view_546: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(view_545, [4, 512, -1, 128]);  view_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_655: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_546, torch.float32);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_110: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_655, 2)
        mean_109: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_110, [-1], True);  pow_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_247: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_109, 1e-06);  mean_109 = None
        rsqrt_109: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_356: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_655, rsqrt_109);  convert_element_type_655 = rsqrt_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_656: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_356, torch.bfloat16);  mul_356 = None
        mul_357: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg302_1, convert_element_type_656);  arg302_1 = convert_element_type_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_299: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_357, [0, 2, 1, 3]);  mul_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_122: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_360: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_299, unsqueeze_122)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_113: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_299, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_81: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_113);  slice_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_112: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_299, 3, 0, 64);  permute_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_55: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_81, slice_112], -1);  neg_81 = slice_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_123: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_361: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_55, unsqueeze_123);  cat_55 = None
        add_249: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_360, mul_361);  mul_360 = mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        view_547: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_355, [2048, 1024])
        permute_300: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        mm_190: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_547, permute_300);  view_547 = permute_300 = None
        view_548: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_190, [4, 512, 1024]);  mm_190 = None
        view_549: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_548, [4, 512, -1, 128]);  view_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_659: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_549, torch.float32);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_111: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_659, 2)
        mean_110: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_111, [-1], True);  pow_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_248: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_110, 1e-06);  mean_110 = None
        rsqrt_110: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
        mul_358: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_659, rsqrt_110);  convert_element_type_659 = rsqrt_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_660: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_358, torch.bfloat16);  mul_358 = None
        mul_359: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg304_1, convert_element_type_660);  arg304_1 = convert_element_type_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_301: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_359, [0, 2, 1, 3]);  mul_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_362: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_301, unsqueeze_122);  unsqueeze_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_115: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_301, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_82: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_115);  slice_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_114: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_301, 3, 0, 64);  permute_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_56: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_82, slice_114], -1);  neg_82 = slice_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_363: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_56, unsqueeze_123);  cat_56 = unsqueeze_123 = None
        add_250: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_362, mul_363);  mul_362 = mul_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_124: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_250, 2);  add_250 = None
        expand_60: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_124, [4, 8, 2, 512, 128]);  unsqueeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_56: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_553: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_56, [4, 16, 512, 128]);  clone_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_550: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_355, [2048, 1024]);  mul_355 = None
        permute_302: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        mm_191: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_550, permute_302);  view_550 = permute_302 = None
        view_551: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_191, [4, 512, 1024]);  mm_191 = None
        view_552: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(view_551, [4, 512, -1, 128]);  view_551 = None
        permute_303: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_125: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_303, 2);  permute_303 = None
        expand_61: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_125, [4, 8, 2, 512, 128]);  unsqueeze_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_57: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_554: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_57, [4, 16, 512, 128]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_56: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_55: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_56, full_default_55);  expand_1 = full_default_56 = full_default_55 = None
        _scaled_dot_product_cudnn_attention_27 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_249, view_553, view_554, where_27, False, scale = 0.08838834764831845);  add_249 = view_553 = view_554 = where_27 = None
        getitem_243: "bf16[4, 16, 512, 128]" = _scaled_dot_product_cudnn_attention_27[0];  _scaled_dot_product_cudnn_attention_27 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[4, 513]" = torch.ops.aten.constant_pad_nd.default(arg312_1, [0, 1], -100.0);  arg312_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_116: "i64[4, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_58: "i64[4, 512]" = torch.ops.aten.clone.default(slice_116, memory_format = torch.contiguous_format);  slice_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_567: "i64[2048]" = torch.ops.aten.reshape.default(clone_58, [-1]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        ne_2: "b8[2048]" = torch.ops.aten.ne.Scalar(view_567, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_304: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(getitem_243, [0, 2, 1, 3]);  getitem_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:289 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_555: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(permute_304, [4, 512, -1]);  permute_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        view_556: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(view_555, [2048, 2048]);  view_555 = None
        permute_305: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        mm_192: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_556, permute_305);  view_556 = permute_305 = None
        view_557: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_192, [4, 512, 1024]);  mm_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_251: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_245, view_557);  add_245 = view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_665: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_251, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_112: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_665, 2)
        mean_111: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_112, [-1], True);  pow_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_252: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_111, 1e-06);  mean_111 = None
        rsqrt_111: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
        mul_364: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_665, rsqrt_111);  convert_element_type_665 = rsqrt_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_666: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_364, torch.bfloat16);  mul_364 = None
        mul_365: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg307_1, convert_element_type_666);  arg307_1 = convert_element_type_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_558: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_365, [2048, 1024])
        permute_306: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
        mm_193: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_558, permute_306);  view_558 = permute_306 = None
        view_559: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_193, [4, 512, 3072]);  mm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_669: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_559, torch.float32);  view_559 = None
        neg_83: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_669)
        exp_27: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_83);  neg_83 = None
        add_253: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        div_27: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_669, add_253);  convert_element_type_669 = add_253 = None
        convert_element_type_670: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_560: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_365, [2048, 1024]);  mul_365 = None
        permute_307: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        mm_194: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_560, permute_307);  view_560 = permute_307 = None
        view_561: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_194, [4, 512, 3072]);  mm_194 = None
        mul_366: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_670, view_561);  convert_element_type_670 = view_561 = None
        view_562: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_366, [2048, 3072]);  mul_366 = None
        permute_308: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        mm_195: "bf16[2048, 1024]" = torch.ops.aten.mm.default(view_562, permute_308);  view_562 = permute_308 = None
        view_563: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_195, [4, 512, 1024]);  mm_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_254: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_251, view_563);  add_251 = view_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_675: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_254, torch.float32);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_113: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_675, 2)
        mean_112: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_113, [-1], True);  pow_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_255: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_112, 1e-06);  mean_112 = None
        rsqrt_112: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        mul_367: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_675, rsqrt_112);  convert_element_type_675 = rsqrt_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_676: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_367, torch.bfloat16);  mul_367 = None
        mul_368: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg311_1, convert_element_type_676);  arg311_1 = convert_element_type_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:505 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_564: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_368, [2048, 1024]);  mul_368 = None
        permute_309: "bf16[1024, 151936]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_196: "bf16[2048, 151936]" = torch.ops.aten.mm.default(view_564, permute_309);  view_564 = permute_309 = None
        view_565: "bf16[4, 512, 151936]" = torch.ops.aten.reshape.default(mm_196, [4, 512, 151936]);  mm_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_679: "f32[4, 512, 151936]" = torch.ops.prims.convert_element_type.default(view_565, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_566: "f32[2048, 151936]" = torch.ops.aten.reshape.default(convert_element_type_679, [-1, 151936]);  convert_element_type_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax: "f32[2048, 1]" = torch.ops.aten.amax.default(view_566, [1], True)
        sub_2: "f32[2048, 151936]" = torch.ops.aten.sub.Tensor(view_566, amax);  view_566 = amax = None
        exp_28: "f32[2048, 151936]" = torch.ops.aten.exp.default(sub_2)
        sum_1: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [1], True);  exp_28 = None
        log: "f32[2048, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_3: "f32[2048, 151936]" = torch.ops.aten.sub.Tensor(sub_2, log);  sub_2 = log = None
        ne_1: "b8[2048]" = torch.ops.aten.ne.Scalar(view_567, -100)
        full_default_57: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "i64[2048]" = torch.ops.aten.where.self(ne_1, view_567, full_default_57);  ne_1 = full_default_57 = None
        unsqueeze_126: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(where_28, 1);  where_28 = None
        gather: "f32[2048, 1]" = torch.ops.aten.gather.default(sub_3, 1, unsqueeze_126);  sub_3 = unsqueeze_126 = None
        squeeze: "f32[2048]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_84: "f32[2048]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_58: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_29: "f32[2048]" = torch.ops.aten.where.self(ne_2, neg_84, full_default_58);  ne_2 = neg_84 = full_default_58 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_29);  where_29 = None
        ne_3: "b8[2048]" = torch.ops.aten.ne.Scalar(view_567, -100);  view_567 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
        convert_element_type_680: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        div_28: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_680);  sum_3 = convert_element_type_680 = None
        return (div_28, view_565)
