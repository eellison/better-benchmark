import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "f16[32000, 4096]", arg2_1: "f16[64]", arg3_1: "f16[4096]", arg4_1: "f16[4096, 4096]", arg5_1: "f16[4096, 4096]", arg6_1: "f16[4096, 4096]", arg7_1: "f16[4096, 4096]", arg8_1: "f16[4096]", arg9_1: "f16[11008, 4096]", arg10_1: "f16[11008, 4096]", arg11_1: "f16[4096, 11008]", arg12_1: "f16[4096]", arg13_1: "f16[4096, 4096]", arg14_1: "f16[4096, 4096]", arg15_1: "f16[4096, 4096]", arg16_1: "f16[4096, 4096]", arg17_1: "f16[4096]", arg18_1: "f16[11008, 4096]", arg19_1: "f16[11008, 4096]", arg20_1: "f16[4096, 11008]", arg21_1: "f16[4096]", arg22_1: "f16[4096, 4096]", arg23_1: "f16[4096, 4096]", arg24_1: "f16[4096, 4096]", arg25_1: "f16[4096, 4096]", arg26_1: "f16[4096]", arg27_1: "f16[11008, 4096]", arg28_1: "f16[11008, 4096]", arg29_1: "f16[4096, 11008]", arg30_1: "f16[4096]", arg31_1: "f16[4096, 4096]", arg32_1: "f16[4096, 4096]", arg33_1: "f16[4096, 4096]", arg34_1: "f16[4096, 4096]", arg35_1: "f16[4096]", arg36_1: "f16[11008, 4096]", arg37_1: "f16[11008, 4096]", arg38_1: "f16[4096, 11008]", arg39_1: "f16[4096]", arg40_1: "f16[4096, 4096]", arg41_1: "f16[4096, 4096]", arg42_1: "f16[4096, 4096]", arg43_1: "f16[4096, 4096]", arg44_1: "f16[4096]", arg45_1: "f16[11008, 4096]", arg46_1: "f16[11008, 4096]", arg47_1: "f16[4096, 11008]", arg48_1: "f16[4096]", arg49_1: "f16[4096, 4096]", arg50_1: "f16[4096, 4096]", arg51_1: "f16[4096, 4096]", arg52_1: "f16[4096, 4096]", arg53_1: "f16[4096]", arg54_1: "f16[11008, 4096]", arg55_1: "f16[11008, 4096]", arg56_1: "f16[4096, 11008]", arg57_1: "f16[4096]", arg58_1: "f16[4096, 4096]", arg59_1: "f16[4096, 4096]", arg60_1: "f16[4096, 4096]", arg61_1: "f16[4096, 4096]", arg62_1: "f16[4096]", arg63_1: "f16[11008, 4096]", arg64_1: "f16[11008, 4096]", arg65_1: "f16[4096, 11008]", arg66_1: "f16[4096]", arg67_1: "f16[4096, 4096]", arg68_1: "f16[4096, 4096]", arg69_1: "f16[4096, 4096]", arg70_1: "f16[4096, 4096]", arg71_1: "f16[4096]", arg72_1: "f16[11008, 4096]", arg73_1: "f16[11008, 4096]", arg74_1: "f16[4096, 11008]", arg75_1: "f16[4096]", arg76_1: "f16[4096, 4096]", arg77_1: "f16[4096, 4096]", arg78_1: "f16[4096, 4096]", arg79_1: "f16[4096, 4096]", arg80_1: "f16[4096]", arg81_1: "f16[11008, 4096]", arg82_1: "f16[11008, 4096]", arg83_1: "f16[4096, 11008]", arg84_1: "f16[4096]", arg85_1: "f16[4096, 4096]", arg86_1: "f16[4096, 4096]", arg87_1: "f16[4096, 4096]", arg88_1: "f16[4096, 4096]", arg89_1: "f16[4096]", arg90_1: "f16[11008, 4096]", arg91_1: "f16[11008, 4096]", arg92_1: "f16[4096, 11008]", arg93_1: "f16[4096]", arg94_1: "f16[4096, 4096]", arg95_1: "f16[4096, 4096]", arg96_1: "f16[4096, 4096]", arg97_1: "f16[4096, 4096]", arg98_1: "f16[4096]", arg99_1: "f16[11008, 4096]", arg100_1: "f16[11008, 4096]", arg101_1: "f16[4096, 11008]", arg102_1: "f16[4096]", arg103_1: "f16[4096, 4096]", arg104_1: "f16[4096, 4096]", arg105_1: "f16[4096, 4096]", arg106_1: "f16[4096, 4096]", arg107_1: "f16[4096]", arg108_1: "f16[11008, 4096]", arg109_1: "f16[11008, 4096]", arg110_1: "f16[4096, 11008]", arg111_1: "f16[4096]", arg112_1: "f16[4096, 4096]", arg113_1: "f16[4096, 4096]", arg114_1: "f16[4096, 4096]", arg115_1: "f16[4096, 4096]", arg116_1: "f16[4096]", arg117_1: "f16[11008, 4096]", arg118_1: "f16[11008, 4096]", arg119_1: "f16[4096, 11008]", arg120_1: "f16[4096]", arg121_1: "f16[4096, 4096]", arg122_1: "f16[4096, 4096]", arg123_1: "f16[4096, 4096]", arg124_1: "f16[4096, 4096]", arg125_1: "f16[4096]", arg126_1: "f16[11008, 4096]", arg127_1: "f16[11008, 4096]", arg128_1: "f16[4096, 11008]", arg129_1: "f16[4096]", arg130_1: "f16[4096, 4096]", arg131_1: "f16[4096, 4096]", arg132_1: "f16[4096, 4096]", arg133_1: "f16[4096, 4096]", arg134_1: "f16[4096]", arg135_1: "f16[11008, 4096]", arg136_1: "f16[11008, 4096]", arg137_1: "f16[4096, 11008]", arg138_1: "f16[4096]", arg139_1: "f16[4096, 4096]", arg140_1: "f16[4096, 4096]", arg141_1: "f16[4096, 4096]", arg142_1: "f16[4096, 4096]", arg143_1: "f16[4096]", arg144_1: "f16[11008, 4096]", arg145_1: "f16[11008, 4096]", arg146_1: "f16[4096, 11008]", arg147_1: "f16[4096]", arg148_1: "f16[4096, 4096]", arg149_1: "f16[4096, 4096]", arg150_1: "f16[4096, 4096]", arg151_1: "f16[4096, 4096]", arg152_1: "f16[4096]", arg153_1: "f16[11008, 4096]", arg154_1: "f16[11008, 4096]", arg155_1: "f16[4096, 11008]", arg156_1: "f16[4096]", arg157_1: "f16[4096, 4096]", arg158_1: "f16[4096, 4096]", arg159_1: "f16[4096, 4096]", arg160_1: "f16[4096, 4096]", arg161_1: "f16[4096]", arg162_1: "f16[11008, 4096]", arg163_1: "f16[11008, 4096]", arg164_1: "f16[4096, 11008]", arg165_1: "f16[4096]", arg166_1: "f16[4096, 4096]", arg167_1: "f16[4096, 4096]", arg168_1: "f16[4096, 4096]", arg169_1: "f16[4096, 4096]", arg170_1: "f16[4096]", arg171_1: "f16[11008, 4096]", arg172_1: "f16[11008, 4096]", arg173_1: "f16[4096, 11008]", arg174_1: "f16[4096]", arg175_1: "f16[4096, 4096]", arg176_1: "f16[4096, 4096]", arg177_1: "f16[4096, 4096]", arg178_1: "f16[4096, 4096]", arg179_1: "f16[4096]", arg180_1: "f16[11008, 4096]", arg181_1: "f16[11008, 4096]", arg182_1: "f16[4096, 11008]", arg183_1: "f16[4096]", arg184_1: "f16[4096, 4096]", arg185_1: "f16[4096, 4096]", arg186_1: "f16[4096, 4096]", arg187_1: "f16[4096, 4096]", arg188_1: "f16[4096]", arg189_1: "f16[11008, 4096]", arg190_1: "f16[11008, 4096]", arg191_1: "f16[4096, 11008]", arg192_1: "f16[4096]", arg193_1: "f16[4096, 4096]", arg194_1: "f16[4096, 4096]", arg195_1: "f16[4096, 4096]", arg196_1: "f16[4096, 4096]", arg197_1: "f16[4096]", arg198_1: "f16[11008, 4096]", arg199_1: "f16[11008, 4096]", arg200_1: "f16[4096, 11008]", arg201_1: "f16[4096]", arg202_1: "f16[4096, 4096]", arg203_1: "f16[4096, 4096]", arg204_1: "f16[4096, 4096]", arg205_1: "f16[4096, 4096]", arg206_1: "f16[4096]", arg207_1: "f16[11008, 4096]", arg208_1: "f16[11008, 4096]", arg209_1: "f16[4096, 11008]", arg210_1: "f16[4096]", arg211_1: "f16[4096, 4096]", arg212_1: "f16[4096, 4096]", arg213_1: "f16[4096, 4096]", arg214_1: "f16[4096, 4096]", arg215_1: "f16[4096]", arg216_1: "f16[11008, 4096]", arg217_1: "f16[11008, 4096]", arg218_1: "f16[4096, 11008]", arg219_1: "f16[4096]", arg220_1: "f16[4096, 4096]", arg221_1: "f16[4096, 4096]", arg222_1: "f16[4096, 4096]", arg223_1: "f16[4096, 4096]", arg224_1: "f16[4096]", arg225_1: "f16[11008, 4096]", arg226_1: "f16[11008, 4096]", arg227_1: "f16[4096, 11008]", arg228_1: "f16[4096]", arg229_1: "f16[4096, 4096]", arg230_1: "f16[4096, 4096]", arg231_1: "f16[4096, 4096]", arg232_1: "f16[4096, 4096]", arg233_1: "f16[4096]", arg234_1: "f16[11008, 4096]", arg235_1: "f16[11008, 4096]", arg236_1: "f16[4096, 11008]", arg237_1: "f16[4096]", arg238_1: "f16[4096, 4096]", arg239_1: "f16[4096, 4096]", arg240_1: "f16[4096, 4096]", arg241_1: "f16[4096, 4096]", arg242_1: "f16[4096]", arg243_1: "f16[11008, 4096]", arg244_1: "f16[11008, 4096]", arg245_1: "f16[4096, 11008]", arg246_1: "f16[4096]", arg247_1: "f16[4096, 4096]", arg248_1: "f16[4096, 4096]", arg249_1: "f16[4096, 4096]", arg250_1: "f16[4096, 4096]", arg251_1: "f16[4096]", arg252_1: "f16[11008, 4096]", arg253_1: "f16[11008, 4096]", arg254_1: "f16[4096, 11008]", arg255_1: "f16[4096]", arg256_1: "f16[4096, 4096]", arg257_1: "f16[4096, 4096]", arg258_1: "f16[4096, 4096]", arg259_1: "f16[4096, 4096]", arg260_1: "f16[4096]", arg261_1: "f16[11008, 4096]", arg262_1: "f16[11008, 4096]", arg263_1: "f16[4096, 11008]", arg264_1: "f16[4096]", arg265_1: "f16[4096, 4096]", arg266_1: "f16[4096, 4096]", arg267_1: "f16[4096, 4096]", arg268_1: "f16[4096, 4096]", arg269_1: "f16[4096]", arg270_1: "f16[11008, 4096]", arg271_1: "f16[11008, 4096]", arg272_1: "f16[4096, 11008]", arg273_1: "f16[4096]", arg274_1: "f16[4096, 4096]", arg275_1: "f16[4096, 4096]", arg276_1: "f16[4096, 4096]", arg277_1: "f16[4096, 4096]", arg278_1: "f16[4096]", arg279_1: "f16[11008, 4096]", arg280_1: "f16[11008, 4096]", arg281_1: "f16[4096, 11008]", arg282_1: "f16[4096]", arg283_1: "f16[4096, 4096]", arg284_1: "f16[4096, 4096]", arg285_1: "f16[4096, 4096]", arg286_1: "f16[4096, 4096]", arg287_1: "f16[4096]", arg288_1: "f16[11008, 4096]", arg289_1: "f16[11008, 4096]", arg290_1: "f16[4096, 11008]", arg291_1: "f16[4096]", arg292_1: "f16[32000, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llava/modeling_llava.py:240 in forward, code: inputs_embeds = self.get_input_embeddings()(input_ids)
        embedding: "f16[1, 512, 4096]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_4: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_4, 2)
        mean: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_3: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_3: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_4, rsqrt);  convert_element_type_4 = rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_5: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_3, torch.float16);  mul_3 = None
        mul_4: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg3_1, convert_element_type_5);  arg3_1 = convert_element_type_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_4: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_4, [512, 4096])
        permute_1: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "f16[512, 4096]" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
        view_5: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm, [1, 512, 4096]);  mm = None
        view_6: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_5, [1, 512, -1, 128]);  view_5 = None
        permute_2: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:125 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_7: "f16[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_8: "f16[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        convert_element_type: "f32[1, 64, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_8, torch.float32);  unsqueeze_8 = None
        expand_1: "f32[1, 64, 1]" = torch.ops.aten.expand.default(convert_element_type, [1, -1, 1]);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:130 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_2: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_1, [1, 64, 1]);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:396 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:397 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:126 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_9: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        convert_element_type_1: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_9, torch.float32);  unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:130 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_3: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_1, [1, 1, 512]);  convert_element_type_1 = None
        mul: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_2, expand_3);  expand_2 = expand_3 = None
        permute: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:131 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_10: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_4: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_10, [1, 512, 2, 64]);  unsqueeze_10 = None
        clone: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_3: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(clone, [1, 512, 128]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:132 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 512, 128]" = torch.ops.aten.cos.default(view_3)
        mul_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:135 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "f16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float16);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_11: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_5: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_2, unsqueeze_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_2: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_2, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_2);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_1: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_2, 3, 0, 64);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg, slice_1], -1);  neg = slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:133 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 512, 128]" = torch.ops.aten.sin.default(view_3);  view_3 = None
        mul_2: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:135 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_3: "f16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_12: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_6: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_12);  cat = None
        add_4: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_7: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_4, [512, 4096])
        permute_3: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1: "f16[512, 4096]" = torch.ops.aten.mm.default(view_7, permute_3);  view_7 = permute_3 = None
        view_8: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_1, [1, 512, 4096]);  mm_1 = None
        view_9: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_8, [1, 512, -1, 128]);  view_8 = None
        permute_4: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_7: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_11);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_4: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_4, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_4);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_3: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_4, 3, 0, 64);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_1: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_1, slice_3], -1);  neg_1 = slice_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_8: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_12);  cat_1 = unsqueeze_12 = None
        add_5: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_10: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_4, [512, 4096]);  mul_4 = None
        permute_5: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_2: "f16[512, 4096]" = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
        view_11: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_2, [1, 512, 4096]);  mm_2 = None
        view_12: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_11, [1, 512, -1, 128]);  view_11 = None
        permute_6: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_2: "i64[512]" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_4: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_2, 0);  add_2 = None
        unsqueeze_5: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[512]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_1: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_2: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 1);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_6, unsqueeze_3);  unsqueeze_6 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 512, 512]" = torch.ops.aten.expand.default(le, [1, -1, 512, 512]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_1, full_default);  full_default_1 = full_default = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_4, add_5, permute_6, where, False, scale = 0.08838834764831845);  add_4 = where = None
        getitem: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_13: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_7, [1, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_14: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_13, [512, 4096]);  view_13 = None
        permute_8: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_3: "f16[512, 4096]" = torch.ops.aten.mm.default(view_14, permute_8);  view_14 = permute_8 = None
        view_15: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_3, [1, 512, 4096]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_6: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(embedding, view_15);  embedding = view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_14: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_2: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_14, 2)
        mean_1: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_7: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_9: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_14, rsqrt_1);  convert_element_type_14 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_15: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_9, torch.float16);  mul_9 = None
        mul_10: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg8_1, convert_element_type_15);  arg8_1 = convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_16: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_10, [512, 4096])
        permute_9: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_4: "f16[512, 11008]" = torch.ops.aten.mm.default(view_16, permute_9);  view_16 = permute_9 = None
        view_17: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_4, [1, 512, 11008]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_18: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        neg_2: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_18)
        exp: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_8: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_18, add_8);  convert_element_type_18 = add_8 = None
        convert_element_type_19: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_18: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_10, [512, 4096]);  mul_10 = None
        permute_10: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_5: "f16[512, 11008]" = torch.ops.aten.mm.default(view_18, permute_10);  view_18 = permute_10 = None
        view_19: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_5, [1, 512, 11008]);  mm_5 = None
        mul_11: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_19, view_19);  convert_element_type_19 = view_19 = None
        view_20: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_11, [512, 11008]);  mul_11 = None
        permute_11: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_6: "f16[512, 4096]" = torch.ops.aten.mm.default(view_20, permute_11);  view_20 = permute_11 = None
        view_21: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_6, [1, 512, 4096]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_9: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_6, view_21);  add_6 = view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_24: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_3: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_24, 2)
        mean_2: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_12: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_24, rsqrt_2);  convert_element_type_24 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_25: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_12, torch.float16);  mul_12 = None
        mul_13: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg12_1, convert_element_type_25);  arg12_1 = convert_element_type_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_22: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_13, [512, 4096])
        permute_12: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_7: "f16[512, 4096]" = torch.ops.aten.mm.default(view_22, permute_12);  view_22 = permute_12 = None
        view_23: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_7, [1, 512, 4096]);  mm_7 = None
        view_24: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_23, [1, 512, -1, 128]);  view_23 = None
        permute_13: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_13: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_14: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_13, unsqueeze_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_6: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_13, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_3: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_6);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_5: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_13, 3, 0, 64);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_2: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_3, slice_5], -1);  neg_3 = slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_14: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_15: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_14);  cat_2 = None
        add_11: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_25: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_13, [512, 4096])
        permute_14: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        mm_8: "f16[512, 4096]" = torch.ops.aten.mm.default(view_25, permute_14);  view_25 = permute_14 = None
        view_26: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_8, [1, 512, 4096]);  mm_8 = None
        view_27: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_26, [1, 512, -1, 128]);  view_26 = None
        permute_15: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_16: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_15, unsqueeze_13);  unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_8: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_15, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_4: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_8);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_7: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_15, 3, 0, 64);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_3: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_4, slice_7], -1);  neg_4 = slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_17: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_14);  cat_3 = unsqueeze_14 = None
        add_12: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_28: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_13, [512, 4096]);  mul_13 = None
        permute_16: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_9: "f16[512, 4096]" = torch.ops.aten.mm.default(view_28, permute_16);  view_28 = permute_16 = None
        view_29: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_9, [1, 512, 4096]);  mm_9 = None
        view_30: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_29, [1, 512, -1, 128]);  view_29 = None
        permute_17: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_3: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_3, full_default_2);  full_default_3 = full_default_2 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_11, add_12, permute_17, where_1, False, scale = 0.08838834764831845);  add_11 = where_1 = None
        getitem_9: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_31: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_18, [1, 512, -1]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_32: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_31, [512, 4096]);  view_31 = None
        permute_19: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_10: "f16[512, 4096]" = torch.ops.aten.mm.default(view_32, permute_19);  view_32 = permute_19 = None
        view_33: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_10, [1, 512, 4096]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_13: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_9, view_33);  add_9 = view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_34: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_4: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_34, 2)
        mean_3: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_18: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_34, rsqrt_3);  convert_element_type_34 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_35: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_18, torch.float16);  mul_18 = None
        mul_19: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg17_1, convert_element_type_35);  arg17_1 = convert_element_type_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_34: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_19, [512, 4096])
        permute_20: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_11: "f16[512, 11008]" = torch.ops.aten.mm.default(view_34, permute_20);  view_34 = permute_20 = None
        view_35: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_11, [1, 512, 11008]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_38: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        neg_5: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_38)
        exp_1: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_15: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_38, add_15);  convert_element_type_38 = add_15 = None
        convert_element_type_39: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_36: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_19, [512, 4096]);  mul_19 = None
        permute_21: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_12: "f16[512, 11008]" = torch.ops.aten.mm.default(view_36, permute_21);  view_36 = permute_21 = None
        view_37: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_12, [1, 512, 11008]);  mm_12 = None
        mul_20: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_39, view_37);  convert_element_type_39 = view_37 = None
        view_38: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_20, [512, 11008]);  mul_20 = None
        permute_22: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_13: "f16[512, 4096]" = torch.ops.aten.mm.default(view_38, permute_22);  view_38 = permute_22 = None
        view_39: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_13, [1, 512, 4096]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_16: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_13, view_39);  add_13 = view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_44: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_5: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_44, 2)
        mean_4: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_21: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_44, rsqrt_4);  convert_element_type_44 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_45: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_21, torch.float16);  mul_21 = None
        mul_22: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg21_1, convert_element_type_45);  arg21_1 = convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_40: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_22, [512, 4096])
        permute_23: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_14: "f16[512, 4096]" = torch.ops.aten.mm.default(view_40, permute_23);  view_40 = permute_23 = None
        view_41: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_14, [1, 512, 4096]);  mm_14 = None
        view_42: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_41, [1, 512, -1, 128]);  view_41 = None
        permute_24: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_15: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_23: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_24, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_10: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_24, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_6: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_10);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_9: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_24, 3, 0, 64);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_4: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_6, slice_9], -1);  neg_6 = slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_16: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_24: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_16);  cat_4 = None
        add_18: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_43: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_22, [512, 4096])
        permute_25: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_15: "f16[512, 4096]" = torch.ops.aten.mm.default(view_43, permute_25);  view_43 = permute_25 = None
        view_44: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_15, [1, 512, 4096]);  mm_15 = None
        view_45: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_44, [1, 512, -1, 128]);  view_44 = None
        permute_26: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_25: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_26, unsqueeze_15);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_12: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_26, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_7: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_12);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_11: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_26, 3, 0, 64);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_5: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_7, slice_11], -1);  neg_7 = slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_26: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_16);  cat_5 = unsqueeze_16 = None
        add_19: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_46: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_22, [512, 4096]);  mul_22 = None
        permute_27: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_16: "f16[512, 4096]" = torch.ops.aten.mm.default(view_46, permute_27);  view_46 = permute_27 = None
        view_47: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_16, [1, 512, 4096]);  mm_16 = None
        view_48: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_47, [1, 512, -1, 128]);  view_47 = None
        permute_28: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_5: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_5, full_default_4);  full_default_5 = full_default_4 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_18, add_19, permute_28, where_2, False, scale = 0.08838834764831845);  add_18 = where_2 = None
        getitem_18: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_49: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_29, [1, 512, -1]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_50: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_49, [512, 4096]);  view_49 = None
        permute_30: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_17: "f16[512, 4096]" = torch.ops.aten.mm.default(view_50, permute_30);  view_50 = permute_30 = None
        view_51: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_17, [1, 512, 4096]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_20: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_16, view_51);  add_16 = view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_54: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_6: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_54, 2)
        mean_5: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_21: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_27: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_54, rsqrt_5);  convert_element_type_54 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_55: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_27, torch.float16);  mul_27 = None
        mul_28: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg26_1, convert_element_type_55);  arg26_1 = convert_element_type_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_52: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_28, [512, 4096])
        permute_31: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_18: "f16[512, 11008]" = torch.ops.aten.mm.default(view_52, permute_31);  view_52 = permute_31 = None
        view_53: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_18, [1, 512, 11008]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_58: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_53, torch.float32);  view_53 = None
        neg_8: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_58)
        exp_2: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_22: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_58, add_22);  convert_element_type_58 = add_22 = None
        convert_element_type_59: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_2, torch.float16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_54: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_28, [512, 4096]);  mul_28 = None
        permute_32: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_19: "f16[512, 11008]" = torch.ops.aten.mm.default(view_54, permute_32);  view_54 = permute_32 = None
        view_55: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_19, [1, 512, 11008]);  mm_19 = None
        mul_29: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_59, view_55);  convert_element_type_59 = view_55 = None
        view_56: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_29, [512, 11008]);  mul_29 = None
        permute_33: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_20: "f16[512, 4096]" = torch.ops.aten.mm.default(view_56, permute_33);  view_56 = permute_33 = None
        view_57: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_20, [1, 512, 4096]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_23: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_20, view_57);  add_20 = view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_64: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_7: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_64, 2)
        mean_6: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_24: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_30: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_64, rsqrt_6);  convert_element_type_64 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_65: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_30, torch.float16);  mul_30 = None
        mul_31: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg30_1, convert_element_type_65);  arg30_1 = convert_element_type_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_58: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_31, [512, 4096])
        permute_34: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_21: "f16[512, 4096]" = torch.ops.aten.mm.default(view_58, permute_34);  view_58 = permute_34 = None
        view_59: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_21, [1, 512, 4096]);  mm_21 = None
        view_60: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_59, [1, 512, -1, 128]);  view_59 = None
        permute_35: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_17: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_32: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_35, unsqueeze_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_14: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_35, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_9: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_14);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_13: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_35, 3, 0, 64);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_6: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_9, slice_13], -1);  neg_9 = slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_18: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_33: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_18);  cat_6 = None
        add_25: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_61: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_31, [512, 4096])
        permute_36: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        mm_22: "f16[512, 4096]" = torch.ops.aten.mm.default(view_61, permute_36);  view_61 = permute_36 = None
        view_62: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_22, [1, 512, 4096]);  mm_22 = None
        view_63: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_62, [1, 512, -1, 128]);  view_62 = None
        permute_37: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_34: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_37, unsqueeze_17);  unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_16: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_37, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_10: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_16);  slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_15: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_37, 3, 0, 64);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_7: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_10, slice_15], -1);  neg_10 = slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_35: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_18);  cat_7 = unsqueeze_18 = None
        add_26: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_64: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_31, [512, 4096]);  mul_31 = None
        permute_38: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_23: "f16[512, 4096]" = torch.ops.aten.mm.default(view_64, permute_38);  view_64 = permute_38 = None
        view_65: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_23, [1, 512, 4096]);  mm_23 = None
        view_66: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_65, [1, 512, -1, 128]);  view_65 = None
        permute_39: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_7: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_25, add_26, permute_39, where_3, False, scale = 0.08838834764831845);  add_25 = where_3 = None
        getitem_27: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_67: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_40, [1, 512, -1]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_68: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_67, [512, 4096]);  view_67 = None
        permute_41: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_24: "f16[512, 4096]" = torch.ops.aten.mm.default(view_68, permute_41);  view_68 = permute_41 = None
        view_69: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_24, [1, 512, 4096]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_27: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_23, view_69);  add_23 = view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_74: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_8: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_74, 2)
        mean_7: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_36: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_74, rsqrt_7);  convert_element_type_74 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_75: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_36, torch.float16);  mul_36 = None
        mul_37: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg35_1, convert_element_type_75);  arg35_1 = convert_element_type_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_70: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_37, [512, 4096])
        permute_42: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_25: "f16[512, 11008]" = torch.ops.aten.mm.default(view_70, permute_42);  view_70 = permute_42 = None
        view_71: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_25, [1, 512, 11008]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_78: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_71, torch.float32);  view_71 = None
        neg_11: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_78)
        exp_3: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_29: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_78, add_29);  convert_element_type_78 = add_29 = None
        convert_element_type_79: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_3, torch.float16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_72: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_37, [512, 4096]);  mul_37 = None
        permute_43: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_26: "f16[512, 11008]" = torch.ops.aten.mm.default(view_72, permute_43);  view_72 = permute_43 = None
        view_73: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_26, [1, 512, 11008]);  mm_26 = None
        mul_38: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_79, view_73);  convert_element_type_79 = view_73 = None
        view_74: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_38, [512, 11008]);  mul_38 = None
        permute_44: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_27: "f16[512, 4096]" = torch.ops.aten.mm.default(view_74, permute_44);  view_74 = permute_44 = None
        view_75: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_27, [1, 512, 4096]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_30: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_27, view_75);  add_27 = view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_84: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_9: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_84, 2)
        mean_8: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_31: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_39: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_84, rsqrt_8);  convert_element_type_84 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_85: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_39, torch.float16);  mul_39 = None
        mul_40: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg39_1, convert_element_type_85);  arg39_1 = convert_element_type_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_76: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_40, [512, 4096])
        permute_45: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_28: "f16[512, 4096]" = torch.ops.aten.mm.default(view_76, permute_45);  view_76 = permute_45 = None
        view_77: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_28, [1, 512, 4096]);  mm_28 = None
        view_78: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_77, [1, 512, -1, 128]);  view_77 = None
        permute_46: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_19: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_41: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_46, unsqueeze_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_18: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_46, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_12: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_18);  slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_17: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_46, 3, 0, 64);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_8: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_12, slice_17], -1);  neg_12 = slice_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_20: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_42: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_20);  cat_8 = None
        add_32: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_79: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_40, [512, 4096])
        permute_47: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_29: "f16[512, 4096]" = torch.ops.aten.mm.default(view_79, permute_47);  view_79 = permute_47 = None
        view_80: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_29, [1, 512, 4096]);  mm_29 = None
        view_81: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_80, [1, 512, -1, 128]);  view_80 = None
        permute_48: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_43: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_48, unsqueeze_19);  unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_20: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_48, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_13: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_20);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_19: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_48, 3, 0, 64);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_9: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_13, slice_19], -1);  neg_13 = slice_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_44: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_20);  cat_9 = unsqueeze_20 = None
        add_33: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_82: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_40, [512, 4096]);  mul_40 = None
        permute_49: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_30: "f16[512, 4096]" = torch.ops.aten.mm.default(view_82, permute_49);  view_82 = permute_49 = None
        view_83: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_30, [1, 512, 4096]);  mm_30 = None
        view_84: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_83, [1, 512, -1, 128]);  view_83 = None
        permute_50: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_9: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_9, full_default_8);  full_default_9 = full_default_8 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_32, add_33, permute_50, where_4, False, scale = 0.08838834764831845);  add_32 = where_4 = None
        getitem_36: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_85: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_51, [1, 512, -1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_86: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_85, [512, 4096]);  view_85 = None
        permute_52: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        mm_31: "f16[512, 4096]" = torch.ops.aten.mm.default(view_86, permute_52);  view_86 = permute_52 = None
        view_87: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_31, [1, 512, 4096]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_34: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_30, view_87);  add_30 = view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_94: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_10: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_94, 2)
        mean_9: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_45: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_94, rsqrt_9);  convert_element_type_94 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_95: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_45, torch.float16);  mul_45 = None
        mul_46: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg44_1, convert_element_type_95);  arg44_1 = convert_element_type_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_88: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_46, [512, 4096])
        permute_53: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_32: "f16[512, 11008]" = torch.ops.aten.mm.default(view_88, permute_53);  view_88 = permute_53 = None
        view_89: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_32, [1, 512, 11008]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_98: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_89, torch.float32);  view_89 = None
        neg_14: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_98)
        exp_4: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_36: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_98, add_36);  convert_element_type_98 = add_36 = None
        convert_element_type_99: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_4, torch.float16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_90: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_46, [512, 4096]);  mul_46 = None
        permute_54: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_33: "f16[512, 11008]" = torch.ops.aten.mm.default(view_90, permute_54);  view_90 = permute_54 = None
        view_91: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_33, [1, 512, 11008]);  mm_33 = None
        mul_47: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_99, view_91);  convert_element_type_99 = view_91 = None
        view_92: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_47, [512, 11008]);  mul_47 = None
        permute_55: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_34: "f16[512, 4096]" = torch.ops.aten.mm.default(view_92, permute_55);  view_92 = permute_55 = None
        view_93: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_34, [1, 512, 4096]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_37: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_34, view_93);  add_34 = view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_104: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_11: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_104, 2)
        mean_10: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_38: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_48: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_104, rsqrt_10);  convert_element_type_104 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_105: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_48, torch.float16);  mul_48 = None
        mul_49: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg48_1, convert_element_type_105);  arg48_1 = convert_element_type_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_94: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_49, [512, 4096])
        permute_56: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_35: "f16[512, 4096]" = torch.ops.aten.mm.default(view_94, permute_56);  view_94 = permute_56 = None
        view_95: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_35, [1, 512, 4096]);  mm_35 = None
        view_96: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_95, [1, 512, -1, 128]);  view_95 = None
        permute_57: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_21: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_50: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_57, unsqueeze_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_22: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_57, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_15: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_22);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_21: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_57, 3, 0, 64);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_10: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_15, slice_21], -1);  neg_15 = slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_22: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_51: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_22);  cat_10 = None
        add_39: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_97: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_49, [512, 4096])
        permute_58: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_36: "f16[512, 4096]" = torch.ops.aten.mm.default(view_97, permute_58);  view_97 = permute_58 = None
        view_98: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_36, [1, 512, 4096]);  mm_36 = None
        view_99: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_98, [1, 512, -1, 128]);  view_98 = None
        permute_59: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_52: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_59, unsqueeze_21);  unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_24: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_59, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_16: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_24);  slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_23: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_59, 3, 0, 64);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_11: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_16, slice_23], -1);  neg_16 = slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_53: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_11, unsqueeze_22);  cat_11 = unsqueeze_22 = None
        add_40: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_100: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_49, [512, 4096]);  mul_49 = None
        permute_60: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_37: "f16[512, 4096]" = torch.ops.aten.mm.default(view_100, permute_60);  view_100 = permute_60 = None
        view_101: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_37, [1, 512, 4096]);  mm_37 = None
        view_102: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_101, [1, 512, -1, 128]);  view_101 = None
        permute_61: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_11: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_11, full_default_10);  full_default_11 = full_default_10 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_39, add_40, permute_61, where_5, False, scale = 0.08838834764831845);  add_39 = where_5 = None
        getitem_45: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_103: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_62, [1, 512, -1]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_104: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_103, [512, 4096]);  view_103 = None
        permute_63: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_38: "f16[512, 4096]" = torch.ops.aten.mm.default(view_104, permute_63);  view_104 = permute_63 = None
        view_105: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_38, [1, 512, 4096]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_41: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_37, view_105);  add_37 = view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_114: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_12: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_114, 2)
        mean_11: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_54: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_114, rsqrt_11);  convert_element_type_114 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_115: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_54, torch.float16);  mul_54 = None
        mul_55: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg53_1, convert_element_type_115);  arg53_1 = convert_element_type_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_106: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_55, [512, 4096])
        permute_64: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_39: "f16[512, 11008]" = torch.ops.aten.mm.default(view_106, permute_64);  view_106 = permute_64 = None
        view_107: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_39, [1, 512, 11008]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_118: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        neg_17: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_118)
        exp_5: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_43: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_118, add_43);  convert_element_type_118 = add_43 = None
        convert_element_type_119: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_5, torch.float16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_108: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_55, [512, 4096]);  mul_55 = None
        permute_65: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_40: "f16[512, 11008]" = torch.ops.aten.mm.default(view_108, permute_65);  view_108 = permute_65 = None
        view_109: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_40, [1, 512, 11008]);  mm_40 = None
        mul_56: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_119, view_109);  convert_element_type_119 = view_109 = None
        view_110: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_56, [512, 11008]);  mul_56 = None
        permute_66: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_41: "f16[512, 4096]" = torch.ops.aten.mm.default(view_110, permute_66);  view_110 = permute_66 = None
        view_111: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_41, [1, 512, 4096]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_44: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_41, view_111);  add_41 = view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_124: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_13: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_124, 2)
        mean_12: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_45: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_57: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_124, rsqrt_12);  convert_element_type_124 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_125: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_57, torch.float16);  mul_57 = None
        mul_58: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg57_1, convert_element_type_125);  arg57_1 = convert_element_type_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_112: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_58, [512, 4096])
        permute_67: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_42: "f16[512, 4096]" = torch.ops.aten.mm.default(view_112, permute_67);  view_112 = permute_67 = None
        view_113: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_42, [1, 512, 4096]);  mm_42 = None
        view_114: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_113, [1, 512, -1, 128]);  view_113 = None
        permute_68: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_23: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_59: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_68, unsqueeze_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_26: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_68, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_18: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_26);  slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_25: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_68, 3, 0, 64);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_12: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_18, slice_25], -1);  neg_18 = slice_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_24: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_60: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_12, unsqueeze_24);  cat_12 = None
        add_46: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_115: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_58, [512, 4096])
        permute_69: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_43: "f16[512, 4096]" = torch.ops.aten.mm.default(view_115, permute_69);  view_115 = permute_69 = None
        view_116: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_43, [1, 512, 4096]);  mm_43 = None
        view_117: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_116, [1, 512, -1, 128]);  view_116 = None
        permute_70: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_61: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_70, unsqueeze_23);  unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_28: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_70, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_19: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_28);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_27: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_70, 3, 0, 64);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_13: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_19, slice_27], -1);  neg_19 = slice_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_62: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_24);  cat_13 = unsqueeze_24 = None
        add_47: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_118: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_58, [512, 4096]);  mul_58 = None
        permute_71: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_44: "f16[512, 4096]" = torch.ops.aten.mm.default(view_118, permute_71);  view_118 = permute_71 = None
        view_119: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_44, [1, 512, 4096]);  mm_44 = None
        view_120: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_119, [1, 512, -1, 128]);  view_119 = None
        permute_72: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_13: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_46, add_47, permute_72, where_6, False, scale = 0.08838834764831845);  add_46 = where_6 = None
        getitem_54: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_121: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_73, [1, 512, -1]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_122: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_121, [512, 4096]);  view_121 = None
        permute_74: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_45: "f16[512, 4096]" = torch.ops.aten.mm.default(view_122, permute_74);  view_122 = permute_74 = None
        view_123: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_45, [1, 512, 4096]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_48: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_44, view_123);  add_44 = view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_134: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_14: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_134, 2)
        mean_13: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_63: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_134, rsqrt_13);  convert_element_type_134 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_135: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_63, torch.float16);  mul_63 = None
        mul_64: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg62_1, convert_element_type_135);  arg62_1 = convert_element_type_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_124: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_64, [512, 4096])
        permute_75: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_46: "f16[512, 11008]" = torch.ops.aten.mm.default(view_124, permute_75);  view_124 = permute_75 = None
        view_125: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_46, [1, 512, 11008]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_138: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None
        neg_20: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_138)
        exp_6: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_50: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_138, add_50);  convert_element_type_138 = add_50 = None
        convert_element_type_139: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_6, torch.float16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_126: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_64, [512, 4096]);  mul_64 = None
        permute_76: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_47: "f16[512, 11008]" = torch.ops.aten.mm.default(view_126, permute_76);  view_126 = permute_76 = None
        view_127: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_47, [1, 512, 11008]);  mm_47 = None
        mul_65: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_139, view_127);  convert_element_type_139 = view_127 = None
        view_128: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_65, [512, 11008]);  mul_65 = None
        permute_77: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_48: "f16[512, 4096]" = torch.ops.aten.mm.default(view_128, permute_77);  view_128 = permute_77 = None
        view_129: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_48, [1, 512, 4096]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_51: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_48, view_129);  add_48 = view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_144: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_15: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_144, 2)
        mean_14: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_52: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_66: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_144, rsqrt_14);  convert_element_type_144 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_145: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_66, torch.float16);  mul_66 = None
        mul_67: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg66_1, convert_element_type_145);  arg66_1 = convert_element_type_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_130: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_67, [512, 4096])
        permute_78: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_49: "f16[512, 4096]" = torch.ops.aten.mm.default(view_130, permute_78);  view_130 = permute_78 = None
        view_131: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_49, [1, 512, 4096]);  mm_49 = None
        view_132: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_131, [1, 512, -1, 128]);  view_131 = None
        permute_79: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_25: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_68: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_79, unsqueeze_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_30: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_79, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_21: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_30);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_29: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_79, 3, 0, 64);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_14: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_21, slice_29], -1);  neg_21 = slice_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_26: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_69: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_14, unsqueeze_26);  cat_14 = None
        add_53: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_133: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_67, [512, 4096])
        permute_80: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_50: "f16[512, 4096]" = torch.ops.aten.mm.default(view_133, permute_80);  view_133 = permute_80 = None
        view_134: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_50, [1, 512, 4096]);  mm_50 = None
        view_135: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_134, [1, 512, -1, 128]);  view_134 = None
        permute_81: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_70: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_81, unsqueeze_25);  unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_32: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_81, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_22: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_32);  slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_31: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_81, 3, 0, 64);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_15: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_22, slice_31], -1);  neg_22 = slice_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_71: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_15, unsqueeze_26);  cat_15 = unsqueeze_26 = None
        add_54: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_70, mul_71);  mul_70 = mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_136: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_67, [512, 4096]);  mul_67 = None
        permute_82: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_51: "f16[512, 4096]" = torch.ops.aten.mm.default(view_136, permute_82);  view_136 = permute_82 = None
        view_137: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_51, [1, 512, 4096]);  mm_51 = None
        view_138: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_137, [1, 512, -1, 128]);  view_137 = None
        permute_83: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_15: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_15, full_default_14);  full_default_15 = full_default_14 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_53, add_54, permute_83, where_7, False, scale = 0.08838834764831845);  add_53 = where_7 = None
        getitem_63: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_139: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_84, [1, 512, -1]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_140: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_139, [512, 4096]);  view_139 = None
        permute_85: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_52: "f16[512, 4096]" = torch.ops.aten.mm.default(view_140, permute_85);  view_140 = permute_85 = None
        view_141: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_52, [1, 512, 4096]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_55: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_51, view_141);  add_51 = view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_154: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_55, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_16: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_154, 2)
        mean_15: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_56: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_72: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_154, rsqrt_15);  convert_element_type_154 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_155: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_72, torch.float16);  mul_72 = None
        mul_73: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg71_1, convert_element_type_155);  arg71_1 = convert_element_type_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_142: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_73, [512, 4096])
        permute_86: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_53: "f16[512, 11008]" = torch.ops.aten.mm.default(view_142, permute_86);  view_142 = permute_86 = None
        view_143: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_53, [1, 512, 11008]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_158: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None
        neg_23: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_158)
        exp_7: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_57: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_158, add_57);  convert_element_type_158 = add_57 = None
        convert_element_type_159: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_7, torch.float16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_144: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_73, [512, 4096]);  mul_73 = None
        permute_87: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_54: "f16[512, 11008]" = torch.ops.aten.mm.default(view_144, permute_87);  view_144 = permute_87 = None
        view_145: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_54, [1, 512, 11008]);  mm_54 = None
        mul_74: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_159, view_145);  convert_element_type_159 = view_145 = None
        view_146: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_74, [512, 11008]);  mul_74 = None
        permute_88: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_55: "f16[512, 4096]" = torch.ops.aten.mm.default(view_146, permute_88);  view_146 = permute_88 = None
        view_147: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_55, [1, 512, 4096]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_58: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_55, view_147);  add_55 = view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_164: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_17: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_164, 2)
        mean_16: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_59: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_75: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_164, rsqrt_16);  convert_element_type_164 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_165: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_75, torch.float16);  mul_75 = None
        mul_76: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg75_1, convert_element_type_165);  arg75_1 = convert_element_type_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_148: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_76, [512, 4096])
        permute_89: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_56: "f16[512, 4096]" = torch.ops.aten.mm.default(view_148, permute_89);  view_148 = permute_89 = None
        view_149: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_56, [1, 512, 4096]);  mm_56 = None
        view_150: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_149, [1, 512, -1, 128]);  view_149 = None
        permute_90: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_27: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_77: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_90, unsqueeze_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_34: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_90, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_24: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_34);  slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_33: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_90, 3, 0, 64);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_16: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_24, slice_33], -1);  neg_24 = slice_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_28: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_78: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_16, unsqueeze_28);  cat_16 = None
        add_60: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_151: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_76, [512, 4096])
        permute_91: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        mm_57: "f16[512, 4096]" = torch.ops.aten.mm.default(view_151, permute_91);  view_151 = permute_91 = None
        view_152: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_57, [1, 512, 4096]);  mm_57 = None
        view_153: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_152, [1, 512, -1, 128]);  view_152 = None
        permute_92: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_79: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_92, unsqueeze_27);  unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_36: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_92, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_25: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_36);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_35: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_92, 3, 0, 64);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_17: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_25, slice_35], -1);  neg_25 = slice_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_80: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_17, unsqueeze_28);  cat_17 = unsqueeze_28 = None
        add_61: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_79, mul_80);  mul_79 = mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_154: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_76, [512, 4096]);  mul_76 = None
        permute_93: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_58: "f16[512, 4096]" = torch.ops.aten.mm.default(view_154, permute_93);  view_154 = permute_93 = None
        view_155: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_58, [1, 512, 4096]);  mm_58 = None
        view_156: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_155, [1, 512, -1, 128]);  view_155 = None
        permute_94: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_17: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_17, full_default_16);  full_default_17 = full_default_16 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_60, add_61, permute_94, where_8, False, scale = 0.08838834764831845);  add_60 = where_8 = None
        getitem_72: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_157: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_95, [1, 512, -1]);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_158: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_157, [512, 4096]);  view_157 = None
        permute_96: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_59: "f16[512, 4096]" = torch.ops.aten.mm.default(view_158, permute_96);  view_158 = permute_96 = None
        view_159: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_59, [1, 512, 4096]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_62: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_58, view_159);  add_58 = view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_174: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_18: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_174, 2)
        mean_17: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_81: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_174, rsqrt_17);  convert_element_type_174 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_175: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_81, torch.float16);  mul_81 = None
        mul_82: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg80_1, convert_element_type_175);  arg80_1 = convert_element_type_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_160: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_82, [512, 4096])
        permute_97: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_60: "f16[512, 11008]" = torch.ops.aten.mm.default(view_160, permute_97);  view_160 = permute_97 = None
        view_161: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_60, [1, 512, 11008]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_178: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_161, torch.float32);  view_161 = None
        neg_26: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_178)
        exp_8: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_64: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_178, add_64);  convert_element_type_178 = add_64 = None
        convert_element_type_179: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_8, torch.float16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_162: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_82, [512, 4096]);  mul_82 = None
        permute_98: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_61: "f16[512, 11008]" = torch.ops.aten.mm.default(view_162, permute_98);  view_162 = permute_98 = None
        view_163: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_61, [1, 512, 11008]);  mm_61 = None
        mul_83: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_179, view_163);  convert_element_type_179 = view_163 = None
        view_164: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_83, [512, 11008]);  mul_83 = None
        permute_99: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_62: "f16[512, 4096]" = torch.ops.aten.mm.default(view_164, permute_99);  view_164 = permute_99 = None
        view_165: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_62, [1, 512, 4096]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_65: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_62, view_165);  add_62 = view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_184: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_65, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_19: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_184, 2)
        mean_18: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_66: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_84: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_184, rsqrt_18);  convert_element_type_184 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_185: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_84, torch.float16);  mul_84 = None
        mul_85: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg84_1, convert_element_type_185);  arg84_1 = convert_element_type_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_166: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_85, [512, 4096])
        permute_100: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_63: "f16[512, 4096]" = torch.ops.aten.mm.default(view_166, permute_100);  view_166 = permute_100 = None
        view_167: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_63, [1, 512, 4096]);  mm_63 = None
        view_168: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_167, [1, 512, -1, 128]);  view_167 = None
        permute_101: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_29: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_86: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_101, unsqueeze_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_38: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_101, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_27: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_38);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_37: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_101, 3, 0, 64);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_18: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_27, slice_37], -1);  neg_27 = slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_30: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_87: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_18, unsqueeze_30);  cat_18 = None
        add_67: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_86, mul_87);  mul_86 = mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_169: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_85, [512, 4096])
        permute_102: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_64: "f16[512, 4096]" = torch.ops.aten.mm.default(view_169, permute_102);  view_169 = permute_102 = None
        view_170: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_64, [1, 512, 4096]);  mm_64 = None
        view_171: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_170, [1, 512, -1, 128]);  view_170 = None
        permute_103: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_88: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_103, unsqueeze_29);  unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_40: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_103, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_28: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_40);  slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_39: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_103, 3, 0, 64);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_19: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_28, slice_39], -1);  neg_28 = slice_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_89: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_19, unsqueeze_30);  cat_19 = unsqueeze_30 = None
        add_68: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_172: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_85, [512, 4096]);  mul_85 = None
        permute_104: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_65: "f16[512, 4096]" = torch.ops.aten.mm.default(view_172, permute_104);  view_172 = permute_104 = None
        view_173: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_65, [1, 512, 4096]);  mm_65 = None
        view_174: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_173, [1, 512, -1, 128]);  view_173 = None
        permute_105: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_19: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_67, add_68, permute_105, where_9, False, scale = 0.08838834764831845);  add_67 = where_9 = None
        getitem_81: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3]);  getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_175: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_106, [1, 512, -1]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_176: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_175, [512, 4096]);  view_175 = None
        permute_107: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        mm_66: "f16[512, 4096]" = torch.ops.aten.mm.default(view_176, permute_107);  view_176 = permute_107 = None
        view_177: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_66, [1, 512, 4096]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_69: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_65, view_177);  add_65 = view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_194: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_69, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_20: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_194, 2)
        mean_19: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_70: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_90: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_194, rsqrt_19);  convert_element_type_194 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_195: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_90, torch.float16);  mul_90 = None
        mul_91: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg89_1, convert_element_type_195);  arg89_1 = convert_element_type_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_178: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_91, [512, 4096])
        permute_108: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_67: "f16[512, 11008]" = torch.ops.aten.mm.default(view_178, permute_108);  view_178 = permute_108 = None
        view_179: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_67, [1, 512, 11008]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_198: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_179, torch.float32);  view_179 = None
        neg_29: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_198)
        exp_9: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_71: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_198, add_71);  convert_element_type_198 = add_71 = None
        convert_element_type_199: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_9, torch.float16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_180: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_91, [512, 4096]);  mul_91 = None
        permute_109: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        mm_68: "f16[512, 11008]" = torch.ops.aten.mm.default(view_180, permute_109);  view_180 = permute_109 = None
        view_181: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_68, [1, 512, 11008]);  mm_68 = None
        mul_92: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_199, view_181);  convert_element_type_199 = view_181 = None
        view_182: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_92, [512, 11008]);  mul_92 = None
        permute_110: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        mm_69: "f16[512, 4096]" = torch.ops.aten.mm.default(view_182, permute_110);  view_182 = permute_110 = None
        view_183: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_69, [1, 512, 4096]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_72: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_69, view_183);  add_69 = view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_204: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_72, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_21: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_204, 2)
        mean_20: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_73: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_93: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_204, rsqrt_20);  convert_element_type_204 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_205: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_93, torch.float16);  mul_93 = None
        mul_94: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg93_1, convert_element_type_205);  arg93_1 = convert_element_type_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_184: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_94, [512, 4096])
        permute_111: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_70: "f16[512, 4096]" = torch.ops.aten.mm.default(view_184, permute_111);  view_184 = permute_111 = None
        view_185: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_70, [1, 512, 4096]);  mm_70 = None
        view_186: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_185, [1, 512, -1, 128]);  view_185 = None
        permute_112: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_31: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_95: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_112, unsqueeze_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_42: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_112, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_30: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_42);  slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_41: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_112, 3, 0, 64);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_20: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_30, slice_41], -1);  neg_30 = slice_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_32: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_96: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_20, unsqueeze_32);  cat_20 = None
        add_74: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_187: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_94, [512, 4096])
        permute_113: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_71: "f16[512, 4096]" = torch.ops.aten.mm.default(view_187, permute_113);  view_187 = permute_113 = None
        view_188: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_71, [1, 512, 4096]);  mm_71 = None
        view_189: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_188, [1, 512, -1, 128]);  view_188 = None
        permute_114: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_97: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_114, unsqueeze_31);  unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_44: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_114, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_31: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_44);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_43: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_114, 3, 0, 64);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_21: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_31, slice_43], -1);  neg_31 = slice_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_98: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_21, unsqueeze_32);  cat_21 = unsqueeze_32 = None
        add_75: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_190: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_94, [512, 4096]);  mul_94 = None
        permute_115: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_72: "f16[512, 4096]" = torch.ops.aten.mm.default(view_190, permute_115);  view_190 = permute_115 = None
        view_191: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_72, [1, 512, 4096]);  mm_72 = None
        view_192: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_191, [1, 512, -1, 128]);  view_191 = None
        permute_116: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_21: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_21, full_default_20);  full_default_21 = full_default_20 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_74, add_75, permute_116, where_10, False, scale = 0.08838834764831845);  add_74 = where_10 = None
        getitem_90: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_193: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_117, [1, 512, -1]);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_194: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_193, [512, 4096]);  view_193 = None
        permute_118: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        mm_73: "f16[512, 4096]" = torch.ops.aten.mm.default(view_194, permute_118);  view_194 = permute_118 = None
        view_195: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_73, [1, 512, 4096]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_76: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_72, view_195);  add_72 = view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_214: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_76, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_22: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_214, 2)
        mean_21: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_77: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_99: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_214, rsqrt_21);  convert_element_type_214 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_215: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_99, torch.float16);  mul_99 = None
        mul_100: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg98_1, convert_element_type_215);  arg98_1 = convert_element_type_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_196: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_100, [512, 4096])
        permute_119: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_74: "f16[512, 11008]" = torch.ops.aten.mm.default(view_196, permute_119);  view_196 = permute_119 = None
        view_197: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_74, [1, 512, 11008]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_218: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        neg_32: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_218)
        exp_10: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_78: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_218, add_78);  convert_element_type_218 = add_78 = None
        convert_element_type_219: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_10, torch.float16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_198: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_100, [512, 4096]);  mul_100 = None
        permute_120: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_75: "f16[512, 11008]" = torch.ops.aten.mm.default(view_198, permute_120);  view_198 = permute_120 = None
        view_199: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_75, [1, 512, 11008]);  mm_75 = None
        mul_101: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_219, view_199);  convert_element_type_219 = view_199 = None
        view_200: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_101, [512, 11008]);  mul_101 = None
        permute_121: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_76: "f16[512, 4096]" = torch.ops.aten.mm.default(view_200, permute_121);  view_200 = permute_121 = None
        view_201: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_76, [1, 512, 4096]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_79: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_76, view_201);  add_76 = view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_224: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_23: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_224, 2)
        mean_22: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_80: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_102: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_224, rsqrt_22);  convert_element_type_224 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_225: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_102, torch.float16);  mul_102 = None
        mul_103: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg102_1, convert_element_type_225);  arg102_1 = convert_element_type_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_202: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_103, [512, 4096])
        permute_122: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_77: "f16[512, 4096]" = torch.ops.aten.mm.default(view_202, permute_122);  view_202 = permute_122 = None
        view_203: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_77, [1, 512, 4096]);  mm_77 = None
        view_204: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_203, [1, 512, -1, 128]);  view_203 = None
        permute_123: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_33: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_104: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_123, unsqueeze_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_46: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_123, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_33: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_46);  slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_45: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_123, 3, 0, 64);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_22: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_33, slice_45], -1);  neg_33 = slice_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_34: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_105: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_22, unsqueeze_34);  cat_22 = None
        add_81: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_205: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_103, [512, 4096])
        permute_124: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_78: "f16[512, 4096]" = torch.ops.aten.mm.default(view_205, permute_124);  view_205 = permute_124 = None
        view_206: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_78, [1, 512, 4096]);  mm_78 = None
        view_207: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_206, [1, 512, -1, 128]);  view_206 = None
        permute_125: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_106: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_125, unsqueeze_33);  unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_48: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_125, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_34: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_48);  slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_47: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_125, 3, 0, 64);  permute_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_23: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_34, slice_47], -1);  neg_34 = slice_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_107: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_23, unsqueeze_34);  cat_23 = unsqueeze_34 = None
        add_82: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_208: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_103, [512, 4096]);  mul_103 = None
        permute_126: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_79: "f16[512, 4096]" = torch.ops.aten.mm.default(view_208, permute_126);  view_208 = permute_126 = None
        view_209: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_79, [1, 512, 4096]);  mm_79 = None
        view_210: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_209, [1, 512, -1, 128]);  view_209 = None
        permute_127: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_23: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_23, full_default_22);  full_default_23 = full_default_22 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_81, add_82, permute_127, where_11, False, scale = 0.08838834764831845);  add_81 = where_11 = None
        getitem_99: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_211: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_128, [1, 512, -1]);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_212: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_211, [512, 4096]);  view_211 = None
        permute_129: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        mm_80: "f16[512, 4096]" = torch.ops.aten.mm.default(view_212, permute_129);  view_212 = permute_129 = None
        view_213: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_80, [1, 512, 4096]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_83: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_79, view_213);  add_79 = view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_234: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_24: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_234, 2)
        mean_23: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_84: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_108: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_234, rsqrt_23);  convert_element_type_234 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_235: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_108, torch.float16);  mul_108 = None
        mul_109: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg107_1, convert_element_type_235);  arg107_1 = convert_element_type_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_214: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_109, [512, 4096])
        permute_130: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_81: "f16[512, 11008]" = torch.ops.aten.mm.default(view_214, permute_130);  view_214 = permute_130 = None
        view_215: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_81, [1, 512, 11008]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_238: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_215, torch.float32);  view_215 = None
        neg_35: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_238)
        exp_11: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_35);  neg_35 = None
        add_85: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_238, add_85);  convert_element_type_238 = add_85 = None
        convert_element_type_239: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_11, torch.float16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_216: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_109, [512, 4096]);  mul_109 = None
        permute_131: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_82: "f16[512, 11008]" = torch.ops.aten.mm.default(view_216, permute_131);  view_216 = permute_131 = None
        view_217: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_82, [1, 512, 11008]);  mm_82 = None
        mul_110: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_239, view_217);  convert_element_type_239 = view_217 = None
        view_218: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_110, [512, 11008]);  mul_110 = None
        permute_132: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        mm_83: "f16[512, 4096]" = torch.ops.aten.mm.default(view_218, permute_132);  view_218 = permute_132 = None
        view_219: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_83, [1, 512, 4096]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_86: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_83, view_219);  add_83 = view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_244: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_86, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_25: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_244, 2)
        mean_24: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_87: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_111: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_244, rsqrt_24);  convert_element_type_244 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_245: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_111, torch.float16);  mul_111 = None
        mul_112: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg111_1, convert_element_type_245);  arg111_1 = convert_element_type_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_220: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_112, [512, 4096])
        permute_133: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_84: "f16[512, 4096]" = torch.ops.aten.mm.default(view_220, permute_133);  view_220 = permute_133 = None
        view_221: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_84, [1, 512, 4096]);  mm_84 = None
        view_222: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_221, [1, 512, -1, 128]);  view_221 = None
        permute_134: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_35: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_113: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_134, unsqueeze_35)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_50: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_134, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_36: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_50);  slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_49: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_134, 3, 0, 64);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_24: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_36, slice_49], -1);  neg_36 = slice_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_36: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_114: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_24, unsqueeze_36);  cat_24 = None
        add_88: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_223: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_112, [512, 4096])
        permute_135: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_85: "f16[512, 4096]" = torch.ops.aten.mm.default(view_223, permute_135);  view_223 = permute_135 = None
        view_224: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_85, [1, 512, 4096]);  mm_85 = None
        view_225: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_224, [1, 512, -1, 128]);  view_224 = None
        permute_136: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_115: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_136, unsqueeze_35);  unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_52: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_136, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_37: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_52);  slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_51: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_136, 3, 0, 64);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_25: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_37, slice_51], -1);  neg_37 = slice_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_116: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_25, unsqueeze_36);  cat_25 = unsqueeze_36 = None
        add_89: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_226: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_112, [512, 4096]);  mul_112 = None
        permute_137: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_86: "f16[512, 4096]" = torch.ops.aten.mm.default(view_226, permute_137);  view_226 = permute_137 = None
        view_227: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_86, [1, 512, 4096]);  mm_86 = None
        view_228: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_227, [1, 512, -1, 128]);  view_227 = None
        permute_138: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_25: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_25, full_default_24);  full_default_25 = full_default_24 = None
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_88, add_89, permute_138, where_12, False, scale = 0.08838834764831845);  add_88 = where_12 = None
        getitem_108: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_12[0];  _scaled_dot_product_cudnn_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_139: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_229: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_139, [1, 512, -1]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_230: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_229, [512, 4096]);  view_229 = None
        permute_140: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_87: "f16[512, 4096]" = torch.ops.aten.mm.default(view_230, permute_140);  view_230 = permute_140 = None
        view_231: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_87, [1, 512, 4096]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_90: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_86, view_231);  add_86 = view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_254: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_90, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_26: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_254, 2)
        mean_25: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_91: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        mul_117: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_254, rsqrt_25);  convert_element_type_254 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_255: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_117, torch.float16);  mul_117 = None
        mul_118: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg116_1, convert_element_type_255);  arg116_1 = convert_element_type_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_232: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_118, [512, 4096])
        permute_141: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_88: "f16[512, 11008]" = torch.ops.aten.mm.default(view_232, permute_141);  view_232 = permute_141 = None
        view_233: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_88, [1, 512, 11008]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_258: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_233, torch.float32);  view_233 = None
        neg_38: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_258)
        exp_12: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_38);  neg_38 = None
        add_92: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_258, add_92);  convert_element_type_258 = add_92 = None
        convert_element_type_259: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_12, torch.float16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_234: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_118, [512, 4096]);  mul_118 = None
        permute_142: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_89: "f16[512, 11008]" = torch.ops.aten.mm.default(view_234, permute_142);  view_234 = permute_142 = None
        view_235: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_89, [1, 512, 11008]);  mm_89 = None
        mul_119: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_259, view_235);  convert_element_type_259 = view_235 = None
        view_236: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_119, [512, 11008]);  mul_119 = None
        permute_143: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_90: "f16[512, 4096]" = torch.ops.aten.mm.default(view_236, permute_143);  view_236 = permute_143 = None
        view_237: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_90, [1, 512, 4096]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_93: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_90, view_237);  add_90 = view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_264: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_93, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_27: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_264, 2)
        mean_26: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_94: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_120: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_264, rsqrt_26);  convert_element_type_264 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_265: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_120, torch.float16);  mul_120 = None
        mul_121: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg120_1, convert_element_type_265);  arg120_1 = convert_element_type_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_238: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_121, [512, 4096])
        permute_144: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_91: "f16[512, 4096]" = torch.ops.aten.mm.default(view_238, permute_144);  view_238 = permute_144 = None
        view_239: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_91, [1, 512, 4096]);  mm_91 = None
        view_240: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_239, [1, 512, -1, 128]);  view_239 = None
        permute_145: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_37: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_122: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_145, unsqueeze_37)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_54: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_145, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_39: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_54);  slice_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_53: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_145, 3, 0, 64);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_26: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_39, slice_53], -1);  neg_39 = slice_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_38: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_123: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_26, unsqueeze_38);  cat_26 = None
        add_95: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_241: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_121, [512, 4096])
        permute_146: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_92: "f16[512, 4096]" = torch.ops.aten.mm.default(view_241, permute_146);  view_241 = permute_146 = None
        view_242: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_92, [1, 512, 4096]);  mm_92 = None
        view_243: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_242, [1, 512, -1, 128]);  view_242 = None
        permute_147: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_124: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_147, unsqueeze_37);  unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_56: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_147, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_40: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_56);  slice_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_55: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_147, 3, 0, 64);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_27: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_40, slice_55], -1);  neg_40 = slice_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_125: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_27, unsqueeze_38);  cat_27 = unsqueeze_38 = None
        add_96: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_244: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_121, [512, 4096]);  mul_121 = None
        permute_148: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_93: "f16[512, 4096]" = torch.ops.aten.mm.default(view_244, permute_148);  view_244 = permute_148 = None
        view_245: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_93, [1, 512, 4096]);  mm_93 = None
        view_246: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_245, [1, 512, -1, 128]);  view_245 = None
        permute_149: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_27: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_27, full_default_26);  full_default_27 = full_default_26 = None
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_95, add_96, permute_149, where_13, False, scale = 0.08838834764831845);  add_95 = where_13 = None
        getitem_117: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_13[0];  _scaled_dot_product_cudnn_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_150: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_247: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_150, [1, 512, -1]);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_248: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_247, [512, 4096]);  view_247 = None
        permute_151: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        mm_94: "f16[512, 4096]" = torch.ops.aten.mm.default(view_248, permute_151);  view_248 = permute_151 = None
        view_249: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_94, [1, 512, 4096]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_97: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_93, view_249);  add_93 = view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_274: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_97, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_28: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_274, 2)
        mean_27: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_126: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_274, rsqrt_27);  convert_element_type_274 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_275: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_126, torch.float16);  mul_126 = None
        mul_127: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg125_1, convert_element_type_275);  arg125_1 = convert_element_type_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_250: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_127, [512, 4096])
        permute_152: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_95: "f16[512, 11008]" = torch.ops.aten.mm.default(view_250, permute_152);  view_250 = permute_152 = None
        view_251: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_95, [1, 512, 11008]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_278: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_251, torch.float32);  view_251 = None
        neg_41: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_278)
        exp_13: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_41);  neg_41 = None
        add_99: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_278, add_99);  convert_element_type_278 = add_99 = None
        convert_element_type_279: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_13, torch.float16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_252: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_127, [512, 4096]);  mul_127 = None
        permute_153: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_96: "f16[512, 11008]" = torch.ops.aten.mm.default(view_252, permute_153);  view_252 = permute_153 = None
        view_253: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_96, [1, 512, 11008]);  mm_96 = None
        mul_128: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_279, view_253);  convert_element_type_279 = view_253 = None
        view_254: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_128, [512, 11008]);  mul_128 = None
        permute_154: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        mm_97: "f16[512, 4096]" = torch.ops.aten.mm.default(view_254, permute_154);  view_254 = permute_154 = None
        view_255: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_97, [1, 512, 4096]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_100: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_97, view_255);  add_97 = view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_284: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_100, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_29: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_284, 2)
        mean_28: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_101: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        mul_129: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_284, rsqrt_28);  convert_element_type_284 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_285: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_129, torch.float16);  mul_129 = None
        mul_130: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg129_1, convert_element_type_285);  arg129_1 = convert_element_type_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_256: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_130, [512, 4096])
        permute_155: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        mm_98: "f16[512, 4096]" = torch.ops.aten.mm.default(view_256, permute_155);  view_256 = permute_155 = None
        view_257: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_98, [1, 512, 4096]);  mm_98 = None
        view_258: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_257, [1, 512, -1, 128]);  view_257 = None
        permute_156: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_39: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_131: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_156, unsqueeze_39)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_58: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_156, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_42: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_58);  slice_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_57: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_156, 3, 0, 64);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_28: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_42, slice_57], -1);  neg_42 = slice_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_40: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_132: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_28, unsqueeze_40);  cat_28 = None
        add_102: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_259: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_130, [512, 4096])
        permute_157: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_99: "f16[512, 4096]" = torch.ops.aten.mm.default(view_259, permute_157);  view_259 = permute_157 = None
        view_260: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_99, [1, 512, 4096]);  mm_99 = None
        view_261: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_260, [1, 512, -1, 128]);  view_260 = None
        permute_158: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_133: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_158, unsqueeze_39);  unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_60: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_158, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_43: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_60);  slice_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_59: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_158, 3, 0, 64);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_29: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_43, slice_59], -1);  neg_43 = slice_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_134: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_29, unsqueeze_40);  cat_29 = unsqueeze_40 = None
        add_103: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_262: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_130, [512, 4096]);  mul_130 = None
        permute_159: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        mm_100: "f16[512, 4096]" = torch.ops.aten.mm.default(view_262, permute_159);  view_262 = permute_159 = None
        view_263: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_100, [1, 512, 4096]);  mm_100 = None
        view_264: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_263, [1, 512, -1, 128]);  view_263 = None
        permute_160: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_29: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_29, full_default_28);  full_default_29 = full_default_28 = None
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_102, add_103, permute_160, where_14, False, scale = 0.08838834764831845);  add_102 = where_14 = None
        getitem_126: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_14[0];  _scaled_dot_product_cudnn_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_161: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_265: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_161, [1, 512, -1]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_266: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_265, [512, 4096]);  view_265 = None
        permute_162: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_101: "f16[512, 4096]" = torch.ops.aten.mm.default(view_266, permute_162);  view_266 = permute_162 = None
        view_267: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_101, [1, 512, 4096]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_104: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_100, view_267);  add_100 = view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_294: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_104, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_30: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_294, 2)
        mean_29: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_105: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        mul_135: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_294, rsqrt_29);  convert_element_type_294 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_295: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_135, torch.float16);  mul_135 = None
        mul_136: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg134_1, convert_element_type_295);  arg134_1 = convert_element_type_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_268: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_136, [512, 4096])
        permute_163: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_102: "f16[512, 11008]" = torch.ops.aten.mm.default(view_268, permute_163);  view_268 = permute_163 = None
        view_269: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_102, [1, 512, 11008]);  mm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_298: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_269, torch.float32);  view_269 = None
        neg_44: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_298)
        exp_14: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_44);  neg_44 = None
        add_106: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_298, add_106);  convert_element_type_298 = add_106 = None
        convert_element_type_299: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_14, torch.float16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_270: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_136, [512, 4096]);  mul_136 = None
        permute_164: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        mm_103: "f16[512, 11008]" = torch.ops.aten.mm.default(view_270, permute_164);  view_270 = permute_164 = None
        view_271: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_103, [1, 512, 11008]);  mm_103 = None
        mul_137: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_299, view_271);  convert_element_type_299 = view_271 = None
        view_272: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_137, [512, 11008]);  mul_137 = None
        permute_165: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_104: "f16[512, 4096]" = torch.ops.aten.mm.default(view_272, permute_165);  view_272 = permute_165 = None
        view_273: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_104, [1, 512, 4096]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_107: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_104, view_273);  add_104 = view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_304: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_31: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_304, 2)
        mean_30: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_108: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_138: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_304, rsqrt_30);  convert_element_type_304 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_305: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_138, torch.float16);  mul_138 = None
        mul_139: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg138_1, convert_element_type_305);  arg138_1 = convert_element_type_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_274: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_139, [512, 4096])
        permute_166: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        mm_105: "f16[512, 4096]" = torch.ops.aten.mm.default(view_274, permute_166);  view_274 = permute_166 = None
        view_275: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_105, [1, 512, 4096]);  mm_105 = None
        view_276: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_275, [1, 512, -1, 128]);  view_275 = None
        permute_167: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_276, [0, 2, 1, 3]);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_41: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_140: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_167, unsqueeze_41)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_62: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_167, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_45: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_62);  slice_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_61: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_167, 3, 0, 64);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_30: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_45, slice_61], -1);  neg_45 = slice_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_42: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_141: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_30, unsqueeze_42);  cat_30 = None
        add_109: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_277: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_139, [512, 4096])
        permute_168: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        mm_106: "f16[512, 4096]" = torch.ops.aten.mm.default(view_277, permute_168);  view_277 = permute_168 = None
        view_278: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_106, [1, 512, 4096]);  mm_106 = None
        view_279: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_278, [1, 512, -1, 128]);  view_278 = None
        permute_169: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_142: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_169, unsqueeze_41);  unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_64: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_169, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_46: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_64);  slice_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_63: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_169, 3, 0, 64);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_31: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_46, slice_63], -1);  neg_46 = slice_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_143: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_31, unsqueeze_42);  cat_31 = unsqueeze_42 = None
        add_110: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_280: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_139, [512, 4096]);  mul_139 = None
        permute_170: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        mm_107: "f16[512, 4096]" = torch.ops.aten.mm.default(view_280, permute_170);  view_280 = permute_170 = None
        view_281: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_107, [1, 512, 4096]);  mm_107 = None
        view_282: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_281, [1, 512, -1, 128]);  view_281 = None
        permute_171: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_31: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_31, full_default_30);  full_default_31 = full_default_30 = None
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_109, add_110, permute_171, where_15, False, scale = 0.08838834764831845);  add_109 = where_15 = None
        getitem_135: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_15[0];  _scaled_dot_product_cudnn_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_172: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_283: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_172, [1, 512, -1]);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_284: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_283, [512, 4096]);  view_283 = None
        permute_173: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        mm_108: "f16[512, 4096]" = torch.ops.aten.mm.default(view_284, permute_173);  view_284 = permute_173 = None
        view_285: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_108, [1, 512, 4096]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_111: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_107, view_285);  add_107 = view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_314: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_111, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_32: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_314, 2)
        mean_31: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_112: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_144: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_314, rsqrt_31);  convert_element_type_314 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_315: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_144, torch.float16);  mul_144 = None
        mul_145: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg143_1, convert_element_type_315);  arg143_1 = convert_element_type_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_286: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_145, [512, 4096])
        permute_174: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        mm_109: "f16[512, 11008]" = torch.ops.aten.mm.default(view_286, permute_174);  view_286 = permute_174 = None
        view_287: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_109, [1, 512, 11008]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_318: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_287, torch.float32);  view_287 = None
        neg_47: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_318)
        exp_15: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_47);  neg_47 = None
        add_113: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_318, add_113);  convert_element_type_318 = add_113 = None
        convert_element_type_319: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_15, torch.float16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_288: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_145, [512, 4096]);  mul_145 = None
        permute_175: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        mm_110: "f16[512, 11008]" = torch.ops.aten.mm.default(view_288, permute_175);  view_288 = permute_175 = None
        view_289: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_110, [1, 512, 11008]);  mm_110 = None
        mul_146: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_319, view_289);  convert_element_type_319 = view_289 = None
        view_290: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_146, [512, 11008]);  mul_146 = None
        permute_176: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        mm_111: "f16[512, 4096]" = torch.ops.aten.mm.default(view_290, permute_176);  view_290 = permute_176 = None
        view_291: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_111, [1, 512, 4096]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_114: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_111, view_291);  add_111 = view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_324: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_33: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_324, 2)
        mean_32: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_115: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        rsqrt_32: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_147: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_324, rsqrt_32);  convert_element_type_324 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_325: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_147, torch.float16);  mul_147 = None
        mul_148: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg147_1, convert_element_type_325);  arg147_1 = convert_element_type_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_292: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_148, [512, 4096])
        permute_177: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        mm_112: "f16[512, 4096]" = torch.ops.aten.mm.default(view_292, permute_177);  view_292 = permute_177 = None
        view_293: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_112, [1, 512, 4096]);  mm_112 = None
        view_294: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_293, [1, 512, -1, 128]);  view_293 = None
        permute_178: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_43: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_149: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_178, unsqueeze_43)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_66: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_178, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_48: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_66);  slice_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_65: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_178, 3, 0, 64);  permute_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_32: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_48, slice_65], -1);  neg_48 = slice_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_44: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_150: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_32, unsqueeze_44);  cat_32 = None
        add_116: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_295: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_148, [512, 4096])
        permute_179: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_113: "f16[512, 4096]" = torch.ops.aten.mm.default(view_295, permute_179);  view_295 = permute_179 = None
        view_296: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_113, [1, 512, 4096]);  mm_113 = None
        view_297: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_296, [1, 512, -1, 128]);  view_296 = None
        permute_180: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_151: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_180, unsqueeze_43);  unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_68: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_180, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_49: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_68);  slice_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_67: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_180, 3, 0, 64);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_33: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_49, slice_67], -1);  neg_49 = slice_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_152: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_33, unsqueeze_44);  cat_33 = unsqueeze_44 = None
        add_117: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_298: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_148, [512, 4096]);  mul_148 = None
        permute_181: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        mm_114: "f16[512, 4096]" = torch.ops.aten.mm.default(view_298, permute_181);  view_298 = permute_181 = None
        view_299: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_114, [1, 512, 4096]);  mm_114 = None
        view_300: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_299, [1, 512, -1, 128]);  view_299 = None
        permute_182: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_33: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_33, full_default_32);  full_default_33 = full_default_32 = None
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_116, add_117, permute_182, where_16, False, scale = 0.08838834764831845);  add_116 = where_16 = None
        getitem_144: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_16[0];  _scaled_dot_product_cudnn_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_183: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_301: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_183, [1, 512, -1]);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_302: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_301, [512, 4096]);  view_301 = None
        permute_184: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_115: "f16[512, 4096]" = torch.ops.aten.mm.default(view_302, permute_184);  view_302 = permute_184 = None
        view_303: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_115, [1, 512, 4096]);  mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_118: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_114, view_303);  add_114 = view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_334: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_118, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_34: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_334, 2)
        mean_33: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_119: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        rsqrt_33: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_153: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_334, rsqrt_33);  convert_element_type_334 = rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_335: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_153, torch.float16);  mul_153 = None
        mul_154: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg152_1, convert_element_type_335);  arg152_1 = convert_element_type_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_304: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_154, [512, 4096])
        permute_185: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        mm_116: "f16[512, 11008]" = torch.ops.aten.mm.default(view_304, permute_185);  view_304 = permute_185 = None
        view_305: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_116, [1, 512, 11008]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_338: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_305, torch.float32);  view_305 = None
        neg_50: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_338)
        exp_16: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_50);  neg_50 = None
        add_120: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_338, add_120);  convert_element_type_338 = add_120 = None
        convert_element_type_339: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_16, torch.float16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_306: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_154, [512, 4096]);  mul_154 = None
        permute_186: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        mm_117: "f16[512, 11008]" = torch.ops.aten.mm.default(view_306, permute_186);  view_306 = permute_186 = None
        view_307: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_117, [1, 512, 11008]);  mm_117 = None
        mul_155: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_339, view_307);  convert_element_type_339 = view_307 = None
        view_308: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_155, [512, 11008]);  mul_155 = None
        permute_187: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        mm_118: "f16[512, 4096]" = torch.ops.aten.mm.default(view_308, permute_187);  view_308 = permute_187 = None
        view_309: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_118, [1, 512, 4096]);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_121: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_118, view_309);  add_118 = view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_344: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_121, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_35: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_344, 2)
        mean_34: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_122: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        rsqrt_34: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_156: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_344, rsqrt_34);  convert_element_type_344 = rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_345: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_156, torch.float16);  mul_156 = None
        mul_157: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg156_1, convert_element_type_345);  arg156_1 = convert_element_type_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_310: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_157, [512, 4096])
        permute_188: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        mm_119: "f16[512, 4096]" = torch.ops.aten.mm.default(view_310, permute_188);  view_310 = permute_188 = None
        view_311: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_119, [1, 512, 4096]);  mm_119 = None
        view_312: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_311, [1, 512, -1, 128]);  view_311 = None
        permute_189: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_45: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_158: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_189, unsqueeze_45)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_70: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_189, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_51: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_70);  slice_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_69: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_189, 3, 0, 64);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_34: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_51, slice_69], -1);  neg_51 = slice_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_46: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_159: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_34, unsqueeze_46);  cat_34 = None
        add_123: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_313: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_157, [512, 4096])
        permute_190: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        mm_120: "f16[512, 4096]" = torch.ops.aten.mm.default(view_313, permute_190);  view_313 = permute_190 = None
        view_314: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_120, [1, 512, 4096]);  mm_120 = None
        view_315: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_314, [1, 512, -1, 128]);  view_314 = None
        permute_191: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_160: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_191, unsqueeze_45);  unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_72: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_191, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_52: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_72);  slice_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_71: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_191, 3, 0, 64);  permute_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_35: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_52, slice_71], -1);  neg_52 = slice_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_161: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_35, unsqueeze_46);  cat_35 = unsqueeze_46 = None
        add_124: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_316: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_157, [512, 4096]);  mul_157 = None
        permute_192: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_121: "f16[512, 4096]" = torch.ops.aten.mm.default(view_316, permute_192);  view_316 = permute_192 = None
        view_317: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_121, [1, 512, 4096]);  mm_121 = None
        view_318: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_317, [1, 512, -1, 128]);  view_317 = None
        permute_193: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_35: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_34: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_35, full_default_34);  full_default_35 = full_default_34 = None
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_123, add_124, permute_193, where_17, False, scale = 0.08838834764831845);  add_123 = where_17 = None
        getitem_153: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_17[0];  _scaled_dot_product_cudnn_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_194: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_319: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_194, [1, 512, -1]);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_320: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_319, [512, 4096]);  view_319 = None
        permute_195: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_122: "f16[512, 4096]" = torch.ops.aten.mm.default(view_320, permute_195);  view_320 = permute_195 = None
        view_321: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_122, [1, 512, 4096]);  mm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_125: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_121, view_321);  add_121 = view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_354: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_125, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_36: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_354, 2)
        mean_35: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_126: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        rsqrt_35: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_162: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_354, rsqrt_35);  convert_element_type_354 = rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_355: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_162, torch.float16);  mul_162 = None
        mul_163: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg161_1, convert_element_type_355);  arg161_1 = convert_element_type_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_322: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_163, [512, 4096])
        permute_196: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        mm_123: "f16[512, 11008]" = torch.ops.aten.mm.default(view_322, permute_196);  view_322 = permute_196 = None
        view_323: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_123, [1, 512, 11008]);  mm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_358: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_323, torch.float32);  view_323 = None
        neg_53: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_358)
        exp_17: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_53);  neg_53 = None
        add_127: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        div_17: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_358, add_127);  convert_element_type_358 = add_127 = None
        convert_element_type_359: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_17, torch.float16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_324: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_163, [512, 4096]);  mul_163 = None
        permute_197: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_124: "f16[512, 11008]" = torch.ops.aten.mm.default(view_324, permute_197);  view_324 = permute_197 = None
        view_325: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_124, [1, 512, 11008]);  mm_124 = None
        mul_164: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_359, view_325);  convert_element_type_359 = view_325 = None
        view_326: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_164, [512, 11008]);  mul_164 = None
        permute_198: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        mm_125: "f16[512, 4096]" = torch.ops.aten.mm.default(view_326, permute_198);  view_326 = permute_198 = None
        view_327: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_125, [1, 512, 4096]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_128: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_125, view_327);  add_125 = view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_364: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_128, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_37: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_364, 2)
        mean_36: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_37, [-1], True);  pow_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_129: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        rsqrt_36: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_165: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_364, rsqrt_36);  convert_element_type_364 = rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_365: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_165, torch.float16);  mul_165 = None
        mul_166: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg165_1, convert_element_type_365);  arg165_1 = convert_element_type_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_328: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_166, [512, 4096])
        permute_199: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        mm_126: "f16[512, 4096]" = torch.ops.aten.mm.default(view_328, permute_199);  view_328 = permute_199 = None
        view_329: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_126, [1, 512, 4096]);  mm_126 = None
        view_330: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_329, [1, 512, -1, 128]);  view_329 = None
        permute_200: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_47: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_167: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_200, unsqueeze_47)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_74: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_200, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_54: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_74);  slice_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_73: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_200, 3, 0, 64);  permute_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_36: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_54, slice_73], -1);  neg_54 = slice_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_48: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_168: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_36, unsqueeze_48);  cat_36 = None
        add_130: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_331: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_166, [512, 4096])
        permute_201: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        mm_127: "f16[512, 4096]" = torch.ops.aten.mm.default(view_331, permute_201);  view_331 = permute_201 = None
        view_332: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_127, [1, 512, 4096]);  mm_127 = None
        view_333: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_332, [1, 512, -1, 128]);  view_332 = None
        permute_202: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_169: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_202, unsqueeze_47);  unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_76: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_202, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_55: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_76);  slice_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_75: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_202, 3, 0, 64);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_37: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_55, slice_75], -1);  neg_55 = slice_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_170: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_37, unsqueeze_48);  cat_37 = unsqueeze_48 = None
        add_131: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_334: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_166, [512, 4096]);  mul_166 = None
        permute_203: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        mm_128: "f16[512, 4096]" = torch.ops.aten.mm.default(view_334, permute_203);  view_334 = permute_203 = None
        view_335: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_128, [1, 512, 4096]);  mm_128 = None
        view_336: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_335, [1, 512, -1, 128]);  view_335 = None
        permute_204: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_37: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_37, full_default_36);  full_default_37 = full_default_36 = None
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_130, add_131, permute_204, where_18, False, scale = 0.08838834764831845);  add_130 = where_18 = None
        getitem_162: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_18[0];  _scaled_dot_product_cudnn_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_205: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_337: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_205, [1, 512, -1]);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_338: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_337, [512, 4096]);  view_337 = None
        permute_206: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_129: "f16[512, 4096]" = torch.ops.aten.mm.default(view_338, permute_206);  view_338 = permute_206 = None
        view_339: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_129, [1, 512, 4096]);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_132: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_128, view_339);  add_128 = view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_374: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_132, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_38: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_374, 2)
        mean_37: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_133: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        rsqrt_37: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_171: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_374, rsqrt_37);  convert_element_type_374 = rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_375: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_171, torch.float16);  mul_171 = None
        mul_172: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg170_1, convert_element_type_375);  arg170_1 = convert_element_type_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_340: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_172, [512, 4096])
        permute_207: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        mm_130: "f16[512, 11008]" = torch.ops.aten.mm.default(view_340, permute_207);  view_340 = permute_207 = None
        view_341: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_130, [1, 512, 11008]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_378: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_341, torch.float32);  view_341 = None
        neg_56: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_378)
        exp_18: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_56);  neg_56 = None
        add_134: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        div_18: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_378, add_134);  convert_element_type_378 = add_134 = None
        convert_element_type_379: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_18, torch.float16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_342: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_172, [512, 4096]);  mul_172 = None
        permute_208: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        mm_131: "f16[512, 11008]" = torch.ops.aten.mm.default(view_342, permute_208);  view_342 = permute_208 = None
        view_343: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_131, [1, 512, 11008]);  mm_131 = None
        mul_173: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_379, view_343);  convert_element_type_379 = view_343 = None
        view_344: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_173, [512, 11008]);  mul_173 = None
        permute_209: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_132: "f16[512, 4096]" = torch.ops.aten.mm.default(view_344, permute_209);  view_344 = permute_209 = None
        view_345: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_132, [1, 512, 4096]);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_135: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_132, view_345);  add_132 = view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_384: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_135, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_39: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_384, 2)
        mean_38: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_136: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        rsqrt_38: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_174: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_384, rsqrt_38);  convert_element_type_384 = rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_385: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_174, torch.float16);  mul_174 = None
        mul_175: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg174_1, convert_element_type_385);  arg174_1 = convert_element_type_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_346: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_175, [512, 4096])
        permute_210: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        mm_133: "f16[512, 4096]" = torch.ops.aten.mm.default(view_346, permute_210);  view_346 = permute_210 = None
        view_347: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_133, [1, 512, 4096]);  mm_133 = None
        view_348: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_347, [1, 512, -1, 128]);  view_347 = None
        permute_211: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_348, [0, 2, 1, 3]);  view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_49: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_176: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_211, unsqueeze_49)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_78: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_211, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_57: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_78);  slice_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_77: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_211, 3, 0, 64);  permute_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_38: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_57, slice_77], -1);  neg_57 = slice_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_50: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_177: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_38, unsqueeze_50);  cat_38 = None
        add_137: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_349: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_175, [512, 4096])
        permute_212: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        mm_134: "f16[512, 4096]" = torch.ops.aten.mm.default(view_349, permute_212);  view_349 = permute_212 = None
        view_350: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_134, [1, 512, 4096]);  mm_134 = None
        view_351: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_350, [1, 512, -1, 128]);  view_350 = None
        permute_213: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_178: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_213, unsqueeze_49);  unsqueeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_80: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_213, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_58: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_80);  slice_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_79: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_213, 3, 0, 64);  permute_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_39: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_58, slice_79], -1);  neg_58 = slice_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_179: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_39, unsqueeze_50);  cat_39 = unsqueeze_50 = None
        add_138: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_352: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_175, [512, 4096]);  mul_175 = None
        permute_214: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        mm_135: "f16[512, 4096]" = torch.ops.aten.mm.default(view_352, permute_214);  view_352 = permute_214 = None
        view_353: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_135, [1, 512, 4096]);  mm_135 = None
        view_354: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_353, [1, 512, -1, 128]);  view_353 = None
        permute_215: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_39: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_39, full_default_38);  full_default_39 = full_default_38 = None
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_137, add_138, permute_215, where_19, False, scale = 0.08838834764831845);  add_137 = where_19 = None
        getitem_171: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_19[0];  _scaled_dot_product_cudnn_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_216: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_355: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_216, [1, 512, -1]);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_356: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_355, [512, 4096]);  view_355 = None
        permute_217: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        mm_136: "f16[512, 4096]" = torch.ops.aten.mm.default(view_356, permute_217);  view_356 = permute_217 = None
        view_357: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_136, [1, 512, 4096]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_139: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_135, view_357);  add_135 = view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_394: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_139, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_40: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_394, 2)
        mean_39: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_140: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        rsqrt_39: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_180: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_394, rsqrt_39);  convert_element_type_394 = rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_395: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_180, torch.float16);  mul_180 = None
        mul_181: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg179_1, convert_element_type_395);  arg179_1 = convert_element_type_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_358: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_181, [512, 4096])
        permute_218: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        mm_137: "f16[512, 11008]" = torch.ops.aten.mm.default(view_358, permute_218);  view_358 = permute_218 = None
        view_359: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_137, [1, 512, 11008]);  mm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_398: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_359, torch.float32);  view_359 = None
        neg_59: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_398)
        exp_19: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_59);  neg_59 = None
        add_141: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_398, add_141);  convert_element_type_398 = add_141 = None
        convert_element_type_399: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_19, torch.float16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_360: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_181, [512, 4096]);  mul_181 = None
        permute_219: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        mm_138: "f16[512, 11008]" = torch.ops.aten.mm.default(view_360, permute_219);  view_360 = permute_219 = None
        view_361: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_138, [1, 512, 11008]);  mm_138 = None
        mul_182: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_399, view_361);  convert_element_type_399 = view_361 = None
        view_362: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_182, [512, 11008]);  mul_182 = None
        permute_220: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        mm_139: "f16[512, 4096]" = torch.ops.aten.mm.default(view_362, permute_220);  view_362 = permute_220 = None
        view_363: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_139, [1, 512, 4096]);  mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_142: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_139, view_363);  add_139 = view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_404: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_142, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_41: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_404, 2)
        mean_40: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_41, [-1], True);  pow_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_143: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        rsqrt_40: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        mul_183: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_404, rsqrt_40);  convert_element_type_404 = rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_405: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_183, torch.float16);  mul_183 = None
        mul_184: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg183_1, convert_element_type_405);  arg183_1 = convert_element_type_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_364: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_184, [512, 4096])
        permute_221: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        mm_140: "f16[512, 4096]" = torch.ops.aten.mm.default(view_364, permute_221);  view_364 = permute_221 = None
        view_365: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_140, [1, 512, 4096]);  mm_140 = None
        view_366: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_365, [1, 512, -1, 128]);  view_365 = None
        permute_222: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_51: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_185: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_222, unsqueeze_51)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_82: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_222, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_60: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_82);  slice_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_81: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_222, 3, 0, 64);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_40: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_60, slice_81], -1);  neg_60 = slice_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_52: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_186: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_40, unsqueeze_52);  cat_40 = None
        add_144: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_367: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_184, [512, 4096])
        permute_223: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_141: "f16[512, 4096]" = torch.ops.aten.mm.default(view_367, permute_223);  view_367 = permute_223 = None
        view_368: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_141, [1, 512, 4096]);  mm_141 = None
        view_369: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_368, [1, 512, -1, 128]);  view_368 = None
        permute_224: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_187: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_224, unsqueeze_51);  unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_84: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_224, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_61: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_84);  slice_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_83: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_224, 3, 0, 64);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_41: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_61, slice_83], -1);  neg_61 = slice_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_188: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_41, unsqueeze_52);  cat_41 = unsqueeze_52 = None
        add_145: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_370: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_184, [512, 4096]);  mul_184 = None
        permute_225: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        mm_142: "f16[512, 4096]" = torch.ops.aten.mm.default(view_370, permute_225);  view_370 = permute_225 = None
        view_371: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_142, [1, 512, 4096]);  mm_142 = None
        view_372: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_371, [1, 512, -1, 128]);  view_371 = None
        permute_226: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_41: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_41, full_default_40);  full_default_41 = full_default_40 = None
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_144, add_145, permute_226, where_20, False, scale = 0.08838834764831845);  add_144 = where_20 = None
        getitem_180: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_20[0];  _scaled_dot_product_cudnn_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_227: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_373: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_227, [1, 512, -1]);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_374: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_373, [512, 4096]);  view_373 = None
        permute_228: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        mm_143: "f16[512, 4096]" = torch.ops.aten.mm.default(view_374, permute_228);  view_374 = permute_228 = None
        view_375: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_143, [1, 512, 4096]);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_146: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_142, view_375);  add_142 = view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_414: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_146, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_42: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_414, 2)
        mean_41: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_147: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        rsqrt_41: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        mul_189: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_414, rsqrt_41);  convert_element_type_414 = rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_415: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_189, torch.float16);  mul_189 = None
        mul_190: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg188_1, convert_element_type_415);  arg188_1 = convert_element_type_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_376: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_190, [512, 4096])
        permute_229: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        mm_144: "f16[512, 11008]" = torch.ops.aten.mm.default(view_376, permute_229);  view_376 = permute_229 = None
        view_377: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_144, [1, 512, 11008]);  mm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_418: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_377, torch.float32);  view_377 = None
        neg_62: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_418)
        exp_20: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_62);  neg_62 = None
        add_148: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        div_20: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_418, add_148);  convert_element_type_418 = add_148 = None
        convert_element_type_419: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_20, torch.float16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_378: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_190, [512, 4096]);  mul_190 = None
        permute_230: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        mm_145: "f16[512, 11008]" = torch.ops.aten.mm.default(view_378, permute_230);  view_378 = permute_230 = None
        view_379: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_145, [1, 512, 11008]);  mm_145 = None
        mul_191: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_419, view_379);  convert_element_type_419 = view_379 = None
        view_380: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_191, [512, 11008]);  mul_191 = None
        permute_231: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        mm_146: "f16[512, 4096]" = torch.ops.aten.mm.default(view_380, permute_231);  view_380 = permute_231 = None
        view_381: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_146, [1, 512, 4096]);  mm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_149: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_146, view_381);  add_146 = view_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_424: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_149, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_43: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_424, 2)
        mean_42: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_150: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_42, 1e-06);  mean_42 = None
        rsqrt_42: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        mul_192: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_424, rsqrt_42);  convert_element_type_424 = rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_425: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_192, torch.float16);  mul_192 = None
        mul_193: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg192_1, convert_element_type_425);  arg192_1 = convert_element_type_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_382: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_193, [512, 4096])
        permute_232: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        mm_147: "f16[512, 4096]" = torch.ops.aten.mm.default(view_382, permute_232);  view_382 = permute_232 = None
        view_383: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_147, [1, 512, 4096]);  mm_147 = None
        view_384: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_383, [1, 512, -1, 128]);  view_383 = None
        permute_233: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_53: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_194: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_233, unsqueeze_53)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_86: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_233, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_63: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_86);  slice_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_85: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_233, 3, 0, 64);  permute_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_42: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_63, slice_85], -1);  neg_63 = slice_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_54: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_195: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_42, unsqueeze_54);  cat_42 = None
        add_151: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_385: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_193, [512, 4096])
        permute_234: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        mm_148: "f16[512, 4096]" = torch.ops.aten.mm.default(view_385, permute_234);  view_385 = permute_234 = None
        view_386: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_148, [1, 512, 4096]);  mm_148 = None
        view_387: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_386, [1, 512, -1, 128]);  view_386 = None
        permute_235: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_196: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_235, unsqueeze_53);  unsqueeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_88: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_235, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_64: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_88);  slice_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_87: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_235, 3, 0, 64);  permute_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_43: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_64, slice_87], -1);  neg_64 = slice_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_197: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_43, unsqueeze_54);  cat_43 = unsqueeze_54 = None
        add_152: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_388: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_193, [512, 4096]);  mul_193 = None
        permute_236: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        mm_149: "f16[512, 4096]" = torch.ops.aten.mm.default(view_388, permute_236);  view_388 = permute_236 = None
        view_389: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_149, [1, 512, 4096]);  mm_149 = None
        view_390: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_389, [1, 512, -1, 128]);  view_389 = None
        permute_237: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_43: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_43, full_default_42);  full_default_43 = full_default_42 = None
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_151, add_152, permute_237, where_21, False, scale = 0.08838834764831845);  add_151 = where_21 = None
        getitem_189: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_21[0];  _scaled_dot_product_cudnn_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_238: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_391: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_238, [1, 512, -1]);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_392: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_391, [512, 4096]);  view_391 = None
        permute_239: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        mm_150: "f16[512, 4096]" = torch.ops.aten.mm.default(view_392, permute_239);  view_392 = permute_239 = None
        view_393: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_150, [1, 512, 4096]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_153: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_149, view_393);  add_149 = view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_434: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_153, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_44: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_434, 2)
        mean_43: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_154: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_43, 1e-06);  mean_43 = None
        rsqrt_43: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        mul_198: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_434, rsqrt_43);  convert_element_type_434 = rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_435: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_198, torch.float16);  mul_198 = None
        mul_199: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg197_1, convert_element_type_435);  arg197_1 = convert_element_type_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_394: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_199, [512, 4096])
        permute_240: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        mm_151: "f16[512, 11008]" = torch.ops.aten.mm.default(view_394, permute_240);  view_394 = permute_240 = None
        view_395: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_151, [1, 512, 11008]);  mm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_438: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_395, torch.float32);  view_395 = None
        neg_65: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_438)
        exp_21: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_65);  neg_65 = None
        add_155: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        div_21: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_438, add_155);  convert_element_type_438 = add_155 = None
        convert_element_type_439: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_21, torch.float16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_396: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_199, [512, 4096]);  mul_199 = None
        permute_241: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        mm_152: "f16[512, 11008]" = torch.ops.aten.mm.default(view_396, permute_241);  view_396 = permute_241 = None
        view_397: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_152, [1, 512, 11008]);  mm_152 = None
        mul_200: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_439, view_397);  convert_element_type_439 = view_397 = None
        view_398: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_200, [512, 11008]);  mul_200 = None
        permute_242: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        mm_153: "f16[512, 4096]" = torch.ops.aten.mm.default(view_398, permute_242);  view_398 = permute_242 = None
        view_399: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_153, [1, 512, 4096]);  mm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_156: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_153, view_399);  add_153 = view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_444: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_156, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_45: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_444, 2)
        mean_44: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_45, [-1], True);  pow_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_157: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_44, 1e-06);  mean_44 = None
        rsqrt_44: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        mul_201: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_444, rsqrt_44);  convert_element_type_444 = rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_445: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_201, torch.float16);  mul_201 = None
        mul_202: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg201_1, convert_element_type_445);  arg201_1 = convert_element_type_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_400: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_202, [512, 4096])
        permute_243: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        mm_154: "f16[512, 4096]" = torch.ops.aten.mm.default(view_400, permute_243);  view_400 = permute_243 = None
        view_401: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_154, [1, 512, 4096]);  mm_154 = None
        view_402: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_401, [1, 512, -1, 128]);  view_401 = None
        permute_244: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_55: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_203: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_244, unsqueeze_55)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_90: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_244, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_66: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_90);  slice_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_89: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_244, 3, 0, 64);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_44: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_66, slice_89], -1);  neg_66 = slice_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_56: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_204: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_44, unsqueeze_56);  cat_44 = None
        add_158: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_403: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_202, [512, 4096])
        permute_245: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        mm_155: "f16[512, 4096]" = torch.ops.aten.mm.default(view_403, permute_245);  view_403 = permute_245 = None
        view_404: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_155, [1, 512, 4096]);  mm_155 = None
        view_405: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_404, [1, 512, -1, 128]);  view_404 = None
        permute_246: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_205: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_246, unsqueeze_55);  unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_92: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_246, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_67: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_92);  slice_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_91: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_246, 3, 0, 64);  permute_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_45: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_67, slice_91], -1);  neg_67 = slice_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_206: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_45, unsqueeze_56);  cat_45 = unsqueeze_56 = None
        add_159: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_406: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_202, [512, 4096]);  mul_202 = None
        permute_247: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        mm_156: "f16[512, 4096]" = torch.ops.aten.mm.default(view_406, permute_247);  view_406 = permute_247 = None
        view_407: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_156, [1, 512, 4096]);  mm_156 = None
        view_408: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_407, [1, 512, -1, 128]);  view_407 = None
        permute_248: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_45: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_45, full_default_44);  full_default_45 = full_default_44 = None
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_158, add_159, permute_248, where_22, False, scale = 0.08838834764831845);  add_158 = where_22 = None
        getitem_198: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_22[0];  _scaled_dot_product_cudnn_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_249: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3]);  getitem_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_409: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_249, [1, 512, -1]);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_410: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_409, [512, 4096]);  view_409 = None
        permute_250: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        mm_157: "f16[512, 4096]" = torch.ops.aten.mm.default(view_410, permute_250);  view_410 = permute_250 = None
        view_411: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_157, [1, 512, 4096]);  mm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_160: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_156, view_411);  add_156 = view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_454: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_160, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_46: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_454, 2)
        mean_45: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_161: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_45, 1e-06);  mean_45 = None
        rsqrt_45: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        mul_207: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_454, rsqrt_45);  convert_element_type_454 = rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_455: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_207, torch.float16);  mul_207 = None
        mul_208: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg206_1, convert_element_type_455);  arg206_1 = convert_element_type_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_412: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_208, [512, 4096])
        permute_251: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        mm_158: "f16[512, 11008]" = torch.ops.aten.mm.default(view_412, permute_251);  view_412 = permute_251 = None
        view_413: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_158, [1, 512, 11008]);  mm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_458: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_413, torch.float32);  view_413 = None
        neg_68: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_458)
        exp_22: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_68);  neg_68 = None
        add_162: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_458, add_162);  convert_element_type_458 = add_162 = None
        convert_element_type_459: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_22, torch.float16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_414: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_208, [512, 4096]);  mul_208 = None
        permute_252: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        mm_159: "f16[512, 11008]" = torch.ops.aten.mm.default(view_414, permute_252);  view_414 = permute_252 = None
        view_415: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_159, [1, 512, 11008]);  mm_159 = None
        mul_209: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_459, view_415);  convert_element_type_459 = view_415 = None
        view_416: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_209, [512, 11008]);  mul_209 = None
        permute_253: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        mm_160: "f16[512, 4096]" = torch.ops.aten.mm.default(view_416, permute_253);  view_416 = permute_253 = None
        view_417: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_160, [1, 512, 4096]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_163: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_160, view_417);  add_160 = view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_464: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_163, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_47: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_464, 2)
        mean_46: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_164: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_46, 1e-06);  mean_46 = None
        rsqrt_46: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        mul_210: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_464, rsqrt_46);  convert_element_type_464 = rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_465: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_210, torch.float16);  mul_210 = None
        mul_211: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg210_1, convert_element_type_465);  arg210_1 = convert_element_type_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_418: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_211, [512, 4096])
        permute_254: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        mm_161: "f16[512, 4096]" = torch.ops.aten.mm.default(view_418, permute_254);  view_418 = permute_254 = None
        view_419: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_161, [1, 512, 4096]);  mm_161 = None
        view_420: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_419, [1, 512, -1, 128]);  view_419 = None
        permute_255: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_57: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_212: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_255, unsqueeze_57)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_94: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_255, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_69: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_94);  slice_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_93: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_255, 3, 0, 64);  permute_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_46: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_69, slice_93], -1);  neg_69 = slice_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_58: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_213: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_46, unsqueeze_58);  cat_46 = None
        add_165: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_421: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_211, [512, 4096])
        permute_256: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        mm_162: "f16[512, 4096]" = torch.ops.aten.mm.default(view_421, permute_256);  view_421 = permute_256 = None
        view_422: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_162, [1, 512, 4096]);  mm_162 = None
        view_423: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_422, [1, 512, -1, 128]);  view_422 = None
        permute_257: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_214: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_257, unsqueeze_57);  unsqueeze_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_96: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_257, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_70: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_96);  slice_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_95: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_257, 3, 0, 64);  permute_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_47: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_70, slice_95], -1);  neg_70 = slice_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_215: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_47, unsqueeze_58);  cat_47 = unsqueeze_58 = None
        add_166: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_424: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_211, [512, 4096]);  mul_211 = None
        permute_258: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        mm_163: "f16[512, 4096]" = torch.ops.aten.mm.default(view_424, permute_258);  view_424 = permute_258 = None
        view_425: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_163, [1, 512, 4096]);  mm_163 = None
        view_426: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_425, [1, 512, -1, 128]);  view_425 = None
        permute_259: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_47: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_47, full_default_46);  full_default_47 = full_default_46 = None
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_165, add_166, permute_259, where_23, False, scale = 0.08838834764831845);  add_165 = where_23 = None
        getitem_207: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_23[0];  _scaled_dot_product_cudnn_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_260: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_427: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_260, [1, 512, -1]);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_428: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_427, [512, 4096]);  view_427 = None
        permute_261: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        mm_164: "f16[512, 4096]" = torch.ops.aten.mm.default(view_428, permute_261);  view_428 = permute_261 = None
        view_429: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_164, [1, 512, 4096]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_167: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_163, view_429);  add_163 = view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_474: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_167, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_48: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_474, 2)
        mean_47: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_168: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_47, 1e-06);  mean_47 = None
        rsqrt_47: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_216: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_474, rsqrt_47);  convert_element_type_474 = rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_475: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_216, torch.float16);  mul_216 = None
        mul_217: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg215_1, convert_element_type_475);  arg215_1 = convert_element_type_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_430: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_217, [512, 4096])
        permute_262: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        mm_165: "f16[512, 11008]" = torch.ops.aten.mm.default(view_430, permute_262);  view_430 = permute_262 = None
        view_431: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_165, [1, 512, 11008]);  mm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_478: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_431, torch.float32);  view_431 = None
        neg_71: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_478)
        exp_23: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_71);  neg_71 = None
        add_169: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        div_23: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_478, add_169);  convert_element_type_478 = add_169 = None
        convert_element_type_479: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_23, torch.float16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_432: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_217, [512, 4096]);  mul_217 = None
        permute_263: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        mm_166: "f16[512, 11008]" = torch.ops.aten.mm.default(view_432, permute_263);  view_432 = permute_263 = None
        view_433: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_166, [1, 512, 11008]);  mm_166 = None
        mul_218: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_479, view_433);  convert_element_type_479 = view_433 = None
        view_434: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_218, [512, 11008]);  mul_218 = None
        permute_264: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        mm_167: "f16[512, 4096]" = torch.ops.aten.mm.default(view_434, permute_264);  view_434 = permute_264 = None
        view_435: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_167, [1, 512, 4096]);  mm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_170: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_167, view_435);  add_167 = view_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_484: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_170, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_49: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_484, 2)
        mean_48: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_49, [-1], True);  pow_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_171: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_48, 1e-06);  mean_48 = None
        rsqrt_48: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        mul_219: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_484, rsqrt_48);  convert_element_type_484 = rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_485: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_219, torch.float16);  mul_219 = None
        mul_220: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg219_1, convert_element_type_485);  arg219_1 = convert_element_type_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_436: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_220, [512, 4096])
        permute_265: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        mm_168: "f16[512, 4096]" = torch.ops.aten.mm.default(view_436, permute_265);  view_436 = permute_265 = None
        view_437: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_168, [1, 512, 4096]);  mm_168 = None
        view_438: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_437, [1, 512, -1, 128]);  view_437 = None
        permute_266: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_438, [0, 2, 1, 3]);  view_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_59: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_221: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_266, unsqueeze_59)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_98: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_266, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_72: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_98);  slice_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_97: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_266, 3, 0, 64);  permute_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_48: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_72, slice_97], -1);  neg_72 = slice_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_60: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_222: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_48, unsqueeze_60);  cat_48 = None
        add_172: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_439: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_220, [512, 4096])
        permute_267: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        mm_169: "f16[512, 4096]" = torch.ops.aten.mm.default(view_439, permute_267);  view_439 = permute_267 = None
        view_440: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_169, [1, 512, 4096]);  mm_169 = None
        view_441: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_440, [1, 512, -1, 128]);  view_440 = None
        permute_268: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_223: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_268, unsqueeze_59);  unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_100: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_268, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_73: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_100);  slice_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_99: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_268, 3, 0, 64);  permute_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_49: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_73, slice_99], -1);  neg_73 = slice_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_224: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_49, unsqueeze_60);  cat_49 = unsqueeze_60 = None
        add_173: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_442: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_220, [512, 4096]);  mul_220 = None
        permute_269: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        mm_170: "f16[512, 4096]" = torch.ops.aten.mm.default(view_442, permute_269);  view_442 = permute_269 = None
        view_443: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_170, [1, 512, 4096]);  mm_170 = None
        view_444: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_443, [1, 512, -1, 128]);  view_443 = None
        permute_270: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_49: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_49, full_default_48);  full_default_49 = full_default_48 = None
        _scaled_dot_product_cudnn_attention_24 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_172, add_173, permute_270, where_24, False, scale = 0.08838834764831845);  add_172 = where_24 = None
        getitem_216: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_24[0];  _scaled_dot_product_cudnn_attention_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_271: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_445: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_271, [1, 512, -1]);  permute_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_446: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_445, [512, 4096]);  view_445 = None
        permute_272: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        mm_171: "f16[512, 4096]" = torch.ops.aten.mm.default(view_446, permute_272);  view_446 = permute_272 = None
        view_447: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_171, [1, 512, 4096]);  mm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_174: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_170, view_447);  add_170 = view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_494: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_174, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_50: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_494, 2)
        mean_49: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_175: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_49, 1e-06);  mean_49 = None
        rsqrt_49: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        mul_225: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_494, rsqrt_49);  convert_element_type_494 = rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_495: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_225, torch.float16);  mul_225 = None
        mul_226: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg224_1, convert_element_type_495);  arg224_1 = convert_element_type_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_448: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_226, [512, 4096])
        permute_273: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        mm_172: "f16[512, 11008]" = torch.ops.aten.mm.default(view_448, permute_273);  view_448 = permute_273 = None
        view_449: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_172, [1, 512, 11008]);  mm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_498: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_449, torch.float32);  view_449 = None
        neg_74: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_498)
        exp_24: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_74);  neg_74 = None
        add_176: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        div_24: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_498, add_176);  convert_element_type_498 = add_176 = None
        convert_element_type_499: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_24, torch.float16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_450: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_226, [512, 4096]);  mul_226 = None
        permute_274: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        mm_173: "f16[512, 11008]" = torch.ops.aten.mm.default(view_450, permute_274);  view_450 = permute_274 = None
        view_451: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_173, [1, 512, 11008]);  mm_173 = None
        mul_227: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_499, view_451);  convert_element_type_499 = view_451 = None
        view_452: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_227, [512, 11008]);  mul_227 = None
        permute_275: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        mm_174: "f16[512, 4096]" = torch.ops.aten.mm.default(view_452, permute_275);  view_452 = permute_275 = None
        view_453: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_174, [1, 512, 4096]);  mm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_177: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_174, view_453);  add_174 = view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_504: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_177, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_51: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_504, 2)
        mean_50: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_178: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_50, 1e-06);  mean_50 = None
        rsqrt_50: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_228: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_504, rsqrt_50);  convert_element_type_504 = rsqrt_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_505: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_228, torch.float16);  mul_228 = None
        mul_229: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg228_1, convert_element_type_505);  arg228_1 = convert_element_type_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_454: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_229, [512, 4096])
        permute_276: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        mm_175: "f16[512, 4096]" = torch.ops.aten.mm.default(view_454, permute_276);  view_454 = permute_276 = None
        view_455: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_175, [1, 512, 4096]);  mm_175 = None
        view_456: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_455, [1, 512, -1, 128]);  view_455 = None
        permute_277: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_61: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_230: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_277, unsqueeze_61)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_102: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_277, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_75: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_102);  slice_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_101: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_277, 3, 0, 64);  permute_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_50: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_75, slice_101], -1);  neg_75 = slice_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_62: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_231: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_50, unsqueeze_62);  cat_50 = None
        add_179: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_457: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_229, [512, 4096])
        permute_278: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
        mm_176: "f16[512, 4096]" = torch.ops.aten.mm.default(view_457, permute_278);  view_457 = permute_278 = None
        view_458: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_176, [1, 512, 4096]);  mm_176 = None
        view_459: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_458, [1, 512, -1, 128]);  view_458 = None
        permute_279: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_232: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_279, unsqueeze_61);  unsqueeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_104: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_279, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_76: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_104);  slice_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_103: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_279, 3, 0, 64);  permute_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_51: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_76, slice_103], -1);  neg_76 = slice_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_233: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_51, unsqueeze_62);  cat_51 = unsqueeze_62 = None
        add_180: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_460: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_229, [512, 4096]);  mul_229 = None
        permute_280: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        mm_177: "f16[512, 4096]" = torch.ops.aten.mm.default(view_460, permute_280);  view_460 = permute_280 = None
        view_461: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_177, [1, 512, 4096]);  mm_177 = None
        view_462: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_461, [1, 512, -1, 128]);  view_461 = None
        permute_281: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_51: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_50: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_51, full_default_50);  full_default_51 = full_default_50 = None
        _scaled_dot_product_cudnn_attention_25 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_179, add_180, permute_281, where_25, False, scale = 0.08838834764831845);  add_179 = where_25 = None
        getitem_225: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_25[0];  _scaled_dot_product_cudnn_attention_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_282: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3]);  getitem_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_463: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_282, [1, 512, -1]);  permute_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_464: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_463, [512, 4096]);  view_463 = None
        permute_283: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        mm_178: "f16[512, 4096]" = torch.ops.aten.mm.default(view_464, permute_283);  view_464 = permute_283 = None
        view_465: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_178, [1, 512, 4096]);  mm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_181: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_177, view_465);  add_177 = view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_514: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_181, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_52: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_514, 2)
        mean_51: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_182: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_51, 1e-06);  mean_51 = None
        rsqrt_51: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        mul_234: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_514, rsqrt_51);  convert_element_type_514 = rsqrt_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_515: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_234, torch.float16);  mul_234 = None
        mul_235: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg233_1, convert_element_type_515);  arg233_1 = convert_element_type_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_466: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_235, [512, 4096])
        permute_284: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        mm_179: "f16[512, 11008]" = torch.ops.aten.mm.default(view_466, permute_284);  view_466 = permute_284 = None
        view_467: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_179, [1, 512, 11008]);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_518: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_467, torch.float32);  view_467 = None
        neg_77: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_518)
        exp_25: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_77);  neg_77 = None
        add_183: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_518, add_183);  convert_element_type_518 = add_183 = None
        convert_element_type_519: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_25, torch.float16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_468: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_235, [512, 4096]);  mul_235 = None
        permute_285: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        mm_180: "f16[512, 11008]" = torch.ops.aten.mm.default(view_468, permute_285);  view_468 = permute_285 = None
        view_469: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_180, [1, 512, 11008]);  mm_180 = None
        mul_236: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_519, view_469);  convert_element_type_519 = view_469 = None
        view_470: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_236, [512, 11008]);  mul_236 = None
        permute_286: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        mm_181: "f16[512, 4096]" = torch.ops.aten.mm.default(view_470, permute_286);  view_470 = permute_286 = None
        view_471: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_181, [1, 512, 4096]);  mm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_184: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_181, view_471);  add_181 = view_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_524: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_184, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_53: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_524, 2)
        mean_52: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_53, [-1], True);  pow_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_185: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_52, 1e-06);  mean_52 = None
        rsqrt_52: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        mul_237: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_524, rsqrt_52);  convert_element_type_524 = rsqrt_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_525: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_237, torch.float16);  mul_237 = None
        mul_238: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg237_1, convert_element_type_525);  arg237_1 = convert_element_type_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_472: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_238, [512, 4096])
        permute_287: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        mm_182: "f16[512, 4096]" = torch.ops.aten.mm.default(view_472, permute_287);  view_472 = permute_287 = None
        view_473: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_182, [1, 512, 4096]);  mm_182 = None
        view_474: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_473, [1, 512, -1, 128]);  view_473 = None
        permute_288: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_474, [0, 2, 1, 3]);  view_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_63: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_239: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_288, unsqueeze_63)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_106: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_288, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_78: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_106);  slice_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_105: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_288, 3, 0, 64);  permute_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_52: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_78, slice_105], -1);  neg_78 = slice_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_64: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_240: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_52, unsqueeze_64);  cat_52 = None
        add_186: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_475: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_238, [512, 4096])
        permute_289: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        mm_183: "f16[512, 4096]" = torch.ops.aten.mm.default(view_475, permute_289);  view_475 = permute_289 = None
        view_476: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_183, [1, 512, 4096]);  mm_183 = None
        view_477: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_476, [1, 512, -1, 128]);  view_476 = None
        permute_290: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_241: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_290, unsqueeze_63);  unsqueeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_108: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_290, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_79: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_108);  slice_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_107: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_290, 3, 0, 64);  permute_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_53: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_79, slice_107], -1);  neg_79 = slice_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_242: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_53, unsqueeze_64);  cat_53 = unsqueeze_64 = None
        add_187: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_478: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_238, [512, 4096]);  mul_238 = None
        permute_291: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        mm_184: "f16[512, 4096]" = torch.ops.aten.mm.default(view_478, permute_291);  view_478 = permute_291 = None
        view_479: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_184, [1, 512, 4096]);  mm_184 = None
        view_480: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_479, [1, 512, -1, 128]);  view_479 = None
        permute_292: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_53: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_53, full_default_52);  full_default_53 = full_default_52 = None
        _scaled_dot_product_cudnn_attention_26 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_186, add_187, permute_292, where_26, False, scale = 0.08838834764831845);  add_186 = where_26 = None
        getitem_234: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_26[0];  _scaled_dot_product_cudnn_attention_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_293: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_234, [0, 2, 1, 3]);  getitem_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_481: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_293, [1, 512, -1]);  permute_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_482: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_481, [512, 4096]);  view_481 = None
        permute_294: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        mm_185: "f16[512, 4096]" = torch.ops.aten.mm.default(view_482, permute_294);  view_482 = permute_294 = None
        view_483: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_185, [1, 512, 4096]);  mm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_188: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_184, view_483);  add_184 = view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_534: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_188, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_54: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_534, 2)
        mean_53: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_189: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_53, 1e-06);  mean_53 = None
        rsqrt_53: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
        mul_243: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_534, rsqrt_53);  convert_element_type_534 = rsqrt_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_535: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_243, torch.float16);  mul_243 = None
        mul_244: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg242_1, convert_element_type_535);  arg242_1 = convert_element_type_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_484: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_244, [512, 4096])
        permute_295: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        mm_186: "f16[512, 11008]" = torch.ops.aten.mm.default(view_484, permute_295);  view_484 = permute_295 = None
        view_485: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_186, [1, 512, 11008]);  mm_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_538: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_485, torch.float32);  view_485 = None
        neg_80: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_538)
        exp_26: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_80);  neg_80 = None
        add_190: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        div_26: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_538, add_190);  convert_element_type_538 = add_190 = None
        convert_element_type_539: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_26, torch.float16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_486: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_244, [512, 4096]);  mul_244 = None
        permute_296: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        mm_187: "f16[512, 11008]" = torch.ops.aten.mm.default(view_486, permute_296);  view_486 = permute_296 = None
        view_487: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_187, [1, 512, 11008]);  mm_187 = None
        mul_245: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_539, view_487);  convert_element_type_539 = view_487 = None
        view_488: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_245, [512, 11008]);  mul_245 = None
        permute_297: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        mm_188: "f16[512, 4096]" = torch.ops.aten.mm.default(view_488, permute_297);  view_488 = permute_297 = None
        view_489: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_188, [1, 512, 4096]);  mm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_191: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_188, view_489);  add_188 = view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_544: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_191, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_55: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_544, 2)
        mean_54: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_192: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_54, 1e-06);  mean_54 = None
        rsqrt_54: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_246: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_544, rsqrt_54);  convert_element_type_544 = rsqrt_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_545: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_246, torch.float16);  mul_246 = None
        mul_247: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg246_1, convert_element_type_545);  arg246_1 = convert_element_type_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_490: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_247, [512, 4096])
        permute_298: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        mm_189: "f16[512, 4096]" = torch.ops.aten.mm.default(view_490, permute_298);  view_490 = permute_298 = None
        view_491: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_189, [1, 512, 4096]);  mm_189 = None
        view_492: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_491, [1, 512, -1, 128]);  view_491 = None
        permute_299: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_65: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_248: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_299, unsqueeze_65)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_110: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_299, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_81: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_110);  slice_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_109: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_299, 3, 0, 64);  permute_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_54: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_81, slice_109], -1);  neg_81 = slice_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_66: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_249: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_54, unsqueeze_66);  cat_54 = None
        add_193: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_493: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_247, [512, 4096])
        permute_300: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        mm_190: "f16[512, 4096]" = torch.ops.aten.mm.default(view_493, permute_300);  view_493 = permute_300 = None
        view_494: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_190, [1, 512, 4096]);  mm_190 = None
        view_495: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_494, [1, 512, -1, 128]);  view_494 = None
        permute_301: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_495, [0, 2, 1, 3]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_250: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_301, unsqueeze_65);  unsqueeze_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_112: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_301, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_82: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_112);  slice_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_111: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_301, 3, 0, 64);  permute_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_55: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_82, slice_111], -1);  neg_82 = slice_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_251: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_55, unsqueeze_66);  cat_55 = unsqueeze_66 = None
        add_194: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_496: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_247, [512, 4096]);  mul_247 = None
        permute_302: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        mm_191: "f16[512, 4096]" = torch.ops.aten.mm.default(view_496, permute_302);  view_496 = permute_302 = None
        view_497: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_191, [1, 512, 4096]);  mm_191 = None
        view_498: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_497, [1, 512, -1, 128]);  view_497 = None
        permute_303: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_55: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_55, full_default_54);  full_default_55 = full_default_54 = None
        _scaled_dot_product_cudnn_attention_27 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_193, add_194, permute_303, where_27, False, scale = 0.08838834764831845);  add_193 = where_27 = None
        getitem_243: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_27[0];  _scaled_dot_product_cudnn_attention_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_304: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_243, [0, 2, 1, 3]);  getitem_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_499: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_304, [1, 512, -1]);  permute_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_500: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_499, [512, 4096]);  view_499 = None
        permute_305: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        mm_192: "f16[512, 4096]" = torch.ops.aten.mm.default(view_500, permute_305);  view_500 = permute_305 = None
        view_501: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_192, [1, 512, 4096]);  mm_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_195: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_191, view_501);  add_191 = view_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_554: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_195, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_56: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_554, 2)
        mean_55: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_196: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_55, 1e-06);  mean_55 = None
        rsqrt_55: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        mul_252: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_554, rsqrt_55);  convert_element_type_554 = rsqrt_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_555: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_252, torch.float16);  mul_252 = None
        mul_253: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg251_1, convert_element_type_555);  arg251_1 = convert_element_type_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_502: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_253, [512, 4096])
        permute_306: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        mm_193: "f16[512, 11008]" = torch.ops.aten.mm.default(view_502, permute_306);  view_502 = permute_306 = None
        view_503: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_193, [1, 512, 11008]);  mm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_558: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None
        neg_83: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_558)
        exp_27: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_83);  neg_83 = None
        add_197: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        div_27: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_558, add_197);  convert_element_type_558 = add_197 = None
        convert_element_type_559: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_27, torch.float16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_504: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_253, [512, 4096]);  mul_253 = None
        permute_307: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        mm_194: "f16[512, 11008]" = torch.ops.aten.mm.default(view_504, permute_307);  view_504 = permute_307 = None
        view_505: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_194, [1, 512, 11008]);  mm_194 = None
        mul_254: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_559, view_505);  convert_element_type_559 = view_505 = None
        view_506: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_254, [512, 11008]);  mul_254 = None
        permute_308: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        mm_195: "f16[512, 4096]" = torch.ops.aten.mm.default(view_506, permute_308);  view_506 = permute_308 = None
        view_507: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_195, [1, 512, 4096]);  mm_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_198: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_195, view_507);  add_195 = view_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_564: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_198, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_57: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_564, 2)
        mean_56: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_57, [-1], True);  pow_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_199: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_56, 1e-06);  mean_56 = None
        rsqrt_56: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        mul_255: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_564, rsqrt_56);  convert_element_type_564 = rsqrt_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_565: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_255, torch.float16);  mul_255 = None
        mul_256: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg255_1, convert_element_type_565);  arg255_1 = convert_element_type_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_508: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_256, [512, 4096])
        permute_309: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        mm_196: "f16[512, 4096]" = torch.ops.aten.mm.default(view_508, permute_309);  view_508 = permute_309 = None
        view_509: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_196, [1, 512, 4096]);  mm_196 = None
        view_510: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_509, [1, 512, -1, 128]);  view_509 = None
        permute_310: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_67: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_257: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_310, unsqueeze_67)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_114: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_310, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_84: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_114);  slice_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_113: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_310, 3, 0, 64);  permute_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_56: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_84, slice_113], -1);  neg_84 = slice_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_68: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_258: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_56, unsqueeze_68);  cat_56 = None
        add_200: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_511: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_256, [512, 4096])
        permute_311: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        mm_197: "f16[512, 4096]" = torch.ops.aten.mm.default(view_511, permute_311);  view_511 = permute_311 = None
        view_512: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_197, [1, 512, 4096]);  mm_197 = None
        view_513: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_512, [1, 512, -1, 128]);  view_512 = None
        permute_312: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_259: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_312, unsqueeze_67);  unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_116: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_312, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_85: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_116);  slice_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_115: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_312, 3, 0, 64);  permute_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_57: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_85, slice_115], -1);  neg_85 = slice_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_260: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_57, unsqueeze_68);  cat_57 = unsqueeze_68 = None
        add_201: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_514: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_256, [512, 4096]);  mul_256 = None
        permute_313: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        mm_198: "f16[512, 4096]" = torch.ops.aten.mm.default(view_514, permute_313);  view_514 = permute_313 = None
        view_515: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_198, [1, 512, 4096]);  mm_198 = None
        view_516: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_515, [1, 512, -1, 128]);  view_515 = None
        permute_314: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_57: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_57, full_default_56);  full_default_57 = full_default_56 = None
        _scaled_dot_product_cudnn_attention_28 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_200, add_201, permute_314, where_28, False, scale = 0.08838834764831845);  add_200 = where_28 = None
        getitem_252: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_28[0];  _scaled_dot_product_cudnn_attention_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_315: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_252, [0, 2, 1, 3]);  getitem_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_517: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_315, [1, 512, -1]);  permute_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_518: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_517, [512, 4096]);  view_517 = None
        permute_316: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        mm_199: "f16[512, 4096]" = torch.ops.aten.mm.default(view_518, permute_316);  view_518 = permute_316 = None
        view_519: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_199, [1, 512, 4096]);  mm_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_202: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_198, view_519);  add_198 = view_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_574: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_202, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_58: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_574, 2)
        mean_57: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_203: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_57, 1e-06);  mean_57 = None
        rsqrt_57: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        mul_261: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_574, rsqrt_57);  convert_element_type_574 = rsqrt_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_575: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_261, torch.float16);  mul_261 = None
        mul_262: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg260_1, convert_element_type_575);  arg260_1 = convert_element_type_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_520: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_262, [512, 4096])
        permute_317: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        mm_200: "f16[512, 11008]" = torch.ops.aten.mm.default(view_520, permute_317);  view_520 = permute_317 = None
        view_521: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_200, [1, 512, 11008]);  mm_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_578: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_521, torch.float32);  view_521 = None
        neg_86: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_578)
        exp_28: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_86);  neg_86 = None
        add_204: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        div_28: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_578, add_204);  convert_element_type_578 = add_204 = None
        convert_element_type_579: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_28, torch.float16);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_522: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_262, [512, 4096]);  mul_262 = None
        permute_318: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        mm_201: "f16[512, 11008]" = torch.ops.aten.mm.default(view_522, permute_318);  view_522 = permute_318 = None
        view_523: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_201, [1, 512, 11008]);  mm_201 = None
        mul_263: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_579, view_523);  convert_element_type_579 = view_523 = None
        view_524: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_263, [512, 11008]);  mul_263 = None
        permute_319: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        mm_202: "f16[512, 4096]" = torch.ops.aten.mm.default(view_524, permute_319);  view_524 = permute_319 = None
        view_525: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_202, [1, 512, 4096]);  mm_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_205: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_202, view_525);  add_202 = view_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_584: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_205, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_59: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_584, 2)
        mean_58: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_59, [-1], True);  pow_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_206: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_58, 1e-06);  mean_58 = None
        rsqrt_58: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        mul_264: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_584, rsqrt_58);  convert_element_type_584 = rsqrt_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_585: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_264, torch.float16);  mul_264 = None
        mul_265: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg264_1, convert_element_type_585);  arg264_1 = convert_element_type_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_526: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_265, [512, 4096])
        permute_320: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        mm_203: "f16[512, 4096]" = torch.ops.aten.mm.default(view_526, permute_320);  view_526 = permute_320 = None
        view_527: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_203, [1, 512, 4096]);  mm_203 = None
        view_528: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_527, [1, 512, -1, 128]);  view_527 = None
        permute_321: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_69: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_266: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_321, unsqueeze_69)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_118: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_321, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_87: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_118);  slice_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_117: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_321, 3, 0, 64);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_58: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_87, slice_117], -1);  neg_87 = slice_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_70: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_267: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_58, unsqueeze_70);  cat_58 = None
        add_207: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_529: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_265, [512, 4096])
        permute_322: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        mm_204: "f16[512, 4096]" = torch.ops.aten.mm.default(view_529, permute_322);  view_529 = permute_322 = None
        view_530: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_204, [1, 512, 4096]);  mm_204 = None
        view_531: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_530, [1, 512, -1, 128]);  view_530 = None
        permute_323: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_268: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_323, unsqueeze_69);  unsqueeze_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_120: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_323, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_88: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_120);  slice_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_119: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_323, 3, 0, 64);  permute_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_59: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_88, slice_119], -1);  neg_88 = slice_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_269: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_59, unsqueeze_70);  cat_59 = unsqueeze_70 = None
        add_208: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_532: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_265, [512, 4096]);  mul_265 = None
        permute_324: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        mm_205: "f16[512, 4096]" = torch.ops.aten.mm.default(view_532, permute_324);  view_532 = permute_324 = None
        view_533: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_205, [1, 512, 4096]);  mm_205 = None
        view_534: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_533, [1, 512, -1, 128]);  view_533 = None
        permute_325: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_59: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_29: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_59, full_default_58);  full_default_59 = full_default_58 = None
        _scaled_dot_product_cudnn_attention_29 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_207, add_208, permute_325, where_29, False, scale = 0.08838834764831845);  add_207 = where_29 = None
        getitem_261: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_29[0];  _scaled_dot_product_cudnn_attention_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_326: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_261, [0, 2, 1, 3]);  getitem_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_535: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_326, [1, 512, -1]);  permute_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_536: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_535, [512, 4096]);  view_535 = None
        permute_327: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        mm_206: "f16[512, 4096]" = torch.ops.aten.mm.default(view_536, permute_327);  view_536 = permute_327 = None
        view_537: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_206, [1, 512, 4096]);  mm_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_209: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_205, view_537);  add_205 = view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_594: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_209, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_60: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_594, 2)
        mean_59: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_60, [-1], True);  pow_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_210: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_59, 1e-06);  mean_59 = None
        rsqrt_59: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        mul_270: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_594, rsqrt_59);  convert_element_type_594 = rsqrt_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_595: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_270, torch.float16);  mul_270 = None
        mul_271: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg269_1, convert_element_type_595);  arg269_1 = convert_element_type_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_538: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_271, [512, 4096])
        permute_328: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        mm_207: "f16[512, 11008]" = torch.ops.aten.mm.default(view_538, permute_328);  view_538 = permute_328 = None
        view_539: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_207, [1, 512, 11008]);  mm_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_598: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_539, torch.float32);  view_539 = None
        neg_89: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_598)
        exp_29: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_89);  neg_89 = None
        add_211: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        div_29: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_598, add_211);  convert_element_type_598 = add_211 = None
        convert_element_type_599: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_29, torch.float16);  div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_540: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_271, [512, 4096]);  mul_271 = None
        permute_329: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        mm_208: "f16[512, 11008]" = torch.ops.aten.mm.default(view_540, permute_329);  view_540 = permute_329 = None
        view_541: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_208, [1, 512, 11008]);  mm_208 = None
        mul_272: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_599, view_541);  convert_element_type_599 = view_541 = None
        view_542: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_272, [512, 11008]);  mul_272 = None
        permute_330: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        mm_209: "f16[512, 4096]" = torch.ops.aten.mm.default(view_542, permute_330);  view_542 = permute_330 = None
        view_543: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_209, [1, 512, 4096]);  mm_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_212: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_209, view_543);  add_209 = view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_604: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_212, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_61: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_604, 2)
        mean_60: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_61, [-1], True);  pow_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_213: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_60, 1e-06);  mean_60 = None
        rsqrt_60: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        mul_273: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_604, rsqrt_60);  convert_element_type_604 = rsqrt_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_605: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_273, torch.float16);  mul_273 = None
        mul_274: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg273_1, convert_element_type_605);  arg273_1 = convert_element_type_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_544: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_274, [512, 4096])
        permute_331: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        mm_210: "f16[512, 4096]" = torch.ops.aten.mm.default(view_544, permute_331);  view_544 = permute_331 = None
        view_545: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_210, [1, 512, 4096]);  mm_210 = None
        view_546: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_545, [1, 512, -1, 128]);  view_545 = None
        permute_332: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_71: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_275: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_332, unsqueeze_71)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_122: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_332, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_90: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_122);  slice_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_121: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_332, 3, 0, 64);  permute_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_60: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_90, slice_121], -1);  neg_90 = slice_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_72: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_276: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_60, unsqueeze_72);  cat_60 = None
        add_214: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_547: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_274, [512, 4096])
        permute_333: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        mm_211: "f16[512, 4096]" = torch.ops.aten.mm.default(view_547, permute_333);  view_547 = permute_333 = None
        view_548: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_211, [1, 512, 4096]);  mm_211 = None
        view_549: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_548, [1, 512, -1, 128]);  view_548 = None
        permute_334: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_549, [0, 2, 1, 3]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_277: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_334, unsqueeze_71);  unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_124: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_334, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_91: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_124);  slice_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_123: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_334, 3, 0, 64);  permute_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_61: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_91, slice_123], -1);  neg_91 = slice_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_278: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_61, unsqueeze_72);  cat_61 = unsqueeze_72 = None
        add_215: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_550: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_274, [512, 4096]);  mul_274 = None
        permute_335: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
        mm_212: "f16[512, 4096]" = torch.ops.aten.mm.default(view_550, permute_335);  view_550 = permute_335 = None
        view_551: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_212, [1, 512, 4096]);  mm_212 = None
        view_552: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_551, [1, 512, -1, 128]);  view_551 = None
        permute_336: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_61: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_30: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_61, full_default_60);  full_default_61 = full_default_60 = None
        _scaled_dot_product_cudnn_attention_30 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_214, add_215, permute_336, where_30, False, scale = 0.08838834764831845);  add_214 = where_30 = None
        getitem_270: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_30[0];  _scaled_dot_product_cudnn_attention_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_337: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_270, [0, 2, 1, 3]);  getitem_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_553: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_337, [1, 512, -1]);  permute_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_554: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_553, [512, 4096]);  view_553 = None
        permute_338: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        mm_213: "f16[512, 4096]" = torch.ops.aten.mm.default(view_554, permute_338);  view_554 = permute_338 = None
        view_555: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_213, [1, 512, 4096]);  mm_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_216: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_212, view_555);  add_212 = view_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_614: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_216, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_62: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_614, 2)
        mean_61: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_62, [-1], True);  pow_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_217: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_61, 1e-06);  mean_61 = None
        rsqrt_61: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        mul_279: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_614, rsqrt_61);  convert_element_type_614 = rsqrt_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_615: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_279, torch.float16);  mul_279 = None
        mul_280: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg278_1, convert_element_type_615);  arg278_1 = convert_element_type_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_556: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_280, [512, 4096])
        permute_339: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        mm_214: "f16[512, 11008]" = torch.ops.aten.mm.default(view_556, permute_339);  view_556 = permute_339 = None
        view_557: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_214, [1, 512, 11008]);  mm_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_618: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_557, torch.float32);  view_557 = None
        neg_92: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_618)
        exp_30: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_92);  neg_92 = None
        add_218: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        div_30: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_618, add_218);  convert_element_type_618 = add_218 = None
        convert_element_type_619: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_30, torch.float16);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_558: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_280, [512, 4096]);  mul_280 = None
        permute_340: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        mm_215: "f16[512, 11008]" = torch.ops.aten.mm.default(view_558, permute_340);  view_558 = permute_340 = None
        view_559: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_215, [1, 512, 11008]);  mm_215 = None
        mul_281: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_619, view_559);  convert_element_type_619 = view_559 = None
        view_560: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_281, [512, 11008]);  mul_281 = None
        permute_341: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        mm_216: "f16[512, 4096]" = torch.ops.aten.mm.default(view_560, permute_341);  view_560 = permute_341 = None
        view_561: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_216, [1, 512, 4096]);  mm_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_219: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_216, view_561);  add_216 = view_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_624: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_219, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_63: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_624, 2)
        mean_62: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_63, [-1], True);  pow_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_220: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_62, 1e-06);  mean_62 = None
        rsqrt_62: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        mul_282: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_624, rsqrt_62);  convert_element_type_624 = rsqrt_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_625: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_282, torch.float16);  mul_282 = None
        mul_283: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg282_1, convert_element_type_625);  arg282_1 = convert_element_type_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_562: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_283, [512, 4096])
        permute_342: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        mm_217: "f16[512, 4096]" = torch.ops.aten.mm.default(view_562, permute_342);  view_562 = permute_342 = None
        view_563: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_217, [1, 512, 4096]);  mm_217 = None
        view_564: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_563, [1, 512, -1, 128]);  view_563 = None
        permute_343: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_564, [0, 2, 1, 3]);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_73: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_284: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_343, unsqueeze_73)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_126: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_343, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_93: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_126);  slice_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_125: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_343, 3, 0, 64);  permute_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_62: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_93, slice_125], -1);  neg_93 = slice_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_74: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_285: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_62, unsqueeze_74);  cat_62 = None
        add_221: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_565: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_283, [512, 4096])
        permute_344: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        mm_218: "f16[512, 4096]" = torch.ops.aten.mm.default(view_565, permute_344);  view_565 = permute_344 = None
        view_566: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_218, [1, 512, 4096]);  mm_218 = None
        view_567: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_566, [1, 512, -1, 128]);  view_566 = None
        permute_345: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_567, [0, 2, 1, 3]);  view_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_286: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_345, unsqueeze_73);  unsqueeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_128: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_345, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_94: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_128);  slice_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_127: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_345, 3, 0, 64);  permute_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_63: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_94, slice_127], -1);  neg_94 = slice_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_287: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_63, unsqueeze_74);  cat_63 = unsqueeze_74 = None
        add_222: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_568: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_283, [512, 4096]);  mul_283 = None
        permute_346: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        mm_219: "f16[512, 4096]" = torch.ops.aten.mm.default(view_568, permute_346);  view_568 = permute_346 = None
        view_569: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_219, [1, 512, 4096]);  mm_219 = None
        view_570: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(view_569, [1, 512, -1, 128]);  view_569 = None
        permute_347: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_63: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_62: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_63, full_default_62);  expand = full_default_63 = full_default_62 = None
        _scaled_dot_product_cudnn_attention_31 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(add_221, add_222, permute_347, where_31, False, scale = 0.08838834764831845);  add_221 = where_31 = None
        getitem_279: "f16[1, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_31[0];  _scaled_dot_product_cudnn_attention_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_348: "f16[1, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_279, [0, 2, 1, 3]);  getitem_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:287 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_571: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(permute_348, [1, 512, -1]);  permute_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        view_572: "f16[512, 4096]" = torch.ops.aten.reshape.default(view_571, [512, 4096]);  view_571 = None
        permute_349: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        mm_220: "f16[512, 4096]" = torch.ops.aten.mm.default(view_572, permute_349);  view_572 = permute_349 = None
        view_573: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_220, [1, 512, 4096]);  mm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_223: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_219, view_573);  add_219 = view_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_634: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_223, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_64: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_634, 2)
        mean_63: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_64, [-1], True);  pow_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_224: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_63, 1e-06);  mean_63 = None
        rsqrt_63: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        mul_288: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_634, rsqrt_63);  convert_element_type_634 = rsqrt_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_635: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_288, torch.float16);  mul_288 = None
        mul_289: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg287_1, convert_element_type_635);  arg287_1 = convert_element_type_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_574: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_289, [512, 4096])
        permute_350: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        mm_221: "f16[512, 11008]" = torch.ops.aten.mm.default(view_574, permute_350);  view_574 = permute_350 = None
        view_575: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_221, [1, 512, 11008]);  mm_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_638: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_575, torch.float32);  view_575 = None
        neg_95: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_638)
        exp_31: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_95);  neg_95 = None
        add_225: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        div_31: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_638, add_225);  convert_element_type_638 = add_225 = None
        convert_element_type_639: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_31, torch.float16);  div_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_576: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_289, [512, 4096]);  mul_289 = None
        permute_351: "f16[4096, 11008]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        mm_222: "f16[512, 11008]" = torch.ops.aten.mm.default(view_576, permute_351);  view_576 = permute_351 = None
        view_577: "f16[1, 512, 11008]" = torch.ops.aten.reshape.default(mm_222, [1, 512, 11008]);  mm_222 = None
        mul_290: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_639, view_577);  convert_element_type_639 = view_577 = None
        view_578: "f16[512, 11008]" = torch.ops.aten.reshape.default(mul_290, [512, 11008]);  mul_290 = None
        permute_352: "f16[11008, 4096]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        mm_223: "f16[512, 4096]" = torch.ops.aten.mm.default(view_578, permute_352);  view_578 = permute_352 = None
        view_579: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_223, [1, 512, 4096]);  mm_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:331 in forward, code: hidden_states = residual + hidden_states
        add_226: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_223, view_579);  add_223 = view_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_644: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_226, torch.float32);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_65: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_644, 2)
        mean_64: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_65, [-1], True);  pow_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_227: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_64, 1e-06);  mean_64 = None
        rsqrt_64: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
        mul_291: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_644, rsqrt_64);  convert_element_type_644 = rsqrt_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_645: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_291, torch.float16);  mul_291 = None
        mul_292: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg291_1, convert_element_type_645);  arg291_1 = convert_element_type_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llava/modeling_llava.py:373 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_580: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_292, [512, 4096]);  mul_292 = None
        permute_353: "f16[4096, 32000]" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
        mm_224: "f16[512, 32000]" = torch.ops.aten.mm.default(view_580, permute_353);  view_580 = permute_353 = None
        view_581: "f16[1, 512, 32000]" = torch.ops.aten.reshape.default(mm_224, [1, 512, 32000]);  mm_224 = None
        return (permute_6, add_5, permute_17, add_12, permute_28, add_19, permute_39, add_26, permute_50, add_33, permute_61, add_40, permute_72, add_47, permute_83, add_54, permute_94, add_61, permute_105, add_68, permute_116, add_75, permute_127, add_82, permute_138, add_89, permute_149, add_96, permute_160, add_103, permute_171, add_110, permute_182, add_117, permute_193, add_124, permute_204, add_131, permute_215, add_138, permute_226, add_145, permute_237, add_152, permute_248, add_159, permute_259, add_166, permute_270, add_173, permute_281, add_180, permute_292, add_187, permute_303, add_194, permute_314, add_201, permute_325, add_208, permute_336, add_215, permute_347, add_222, view_581)
