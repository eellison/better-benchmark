import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "f16[51200, 2048]", arg2_1: "f16[16]", arg3_1: "f16[2048]", arg4_1: "f16[2048]", arg5_1: "f16[2048, 2048]", arg6_1: "f16[2048]", arg7_1: "f16[2048, 2048]", arg8_1: "f16[2048]", arg9_1: "f16[2048, 2048]", arg10_1: "f16[2048]", arg11_1: "f16[2048, 2048]", arg12_1: "f16[2048]", arg13_1: "f16[8192, 2048]", arg14_1: "f16[8192]", arg15_1: "f16[2048, 8192]", arg16_1: "f16[2048]", arg17_1: "f16[2048]", arg18_1: "f16[2048]", arg19_1: "f16[2048, 2048]", arg20_1: "f16[2048]", arg21_1: "f16[2048, 2048]", arg22_1: "f16[2048]", arg23_1: "f16[2048, 2048]", arg24_1: "f16[2048]", arg25_1: "f16[2048, 2048]", arg26_1: "f16[2048]", arg27_1: "f16[8192, 2048]", arg28_1: "f16[8192]", arg29_1: "f16[2048, 8192]", arg30_1: "f16[2048]", arg31_1: "f16[2048]", arg32_1: "f16[2048]", arg33_1: "f16[2048, 2048]", arg34_1: "f16[2048]", arg35_1: "f16[2048, 2048]", arg36_1: "f16[2048]", arg37_1: "f16[2048, 2048]", arg38_1: "f16[2048]", arg39_1: "f16[2048, 2048]", arg40_1: "f16[2048]", arg41_1: "f16[8192, 2048]", arg42_1: "f16[8192]", arg43_1: "f16[2048, 8192]", arg44_1: "f16[2048]", arg45_1: "f16[2048]", arg46_1: "f16[2048]", arg47_1: "f16[2048, 2048]", arg48_1: "f16[2048]", arg49_1: "f16[2048, 2048]", arg50_1: "f16[2048]", arg51_1: "f16[2048, 2048]", arg52_1: "f16[2048]", arg53_1: "f16[2048, 2048]", arg54_1: "f16[2048]", arg55_1: "f16[8192, 2048]", arg56_1: "f16[8192]", arg57_1: "f16[2048, 8192]", arg58_1: "f16[2048]", arg59_1: "f16[2048]", arg60_1: "f16[2048]", arg61_1: "f16[2048, 2048]", arg62_1: "f16[2048]", arg63_1: "f16[2048, 2048]", arg64_1: "f16[2048]", arg65_1: "f16[2048, 2048]", arg66_1: "f16[2048]", arg67_1: "f16[2048, 2048]", arg68_1: "f16[2048]", arg69_1: "f16[8192, 2048]", arg70_1: "f16[8192]", arg71_1: "f16[2048, 8192]", arg72_1: "f16[2048]", arg73_1: "f16[2048]", arg74_1: "f16[2048]", arg75_1: "f16[2048, 2048]", arg76_1: "f16[2048]", arg77_1: "f16[2048, 2048]", arg78_1: "f16[2048]", arg79_1: "f16[2048, 2048]", arg80_1: "f16[2048]", arg81_1: "f16[2048, 2048]", arg82_1: "f16[2048]", arg83_1: "f16[8192, 2048]", arg84_1: "f16[8192]", arg85_1: "f16[2048, 8192]", arg86_1: "f16[2048]", arg87_1: "f16[2048]", arg88_1: "f16[2048]", arg89_1: "f16[2048, 2048]", arg90_1: "f16[2048]", arg91_1: "f16[2048, 2048]", arg92_1: "f16[2048]", arg93_1: "f16[2048, 2048]", arg94_1: "f16[2048]", arg95_1: "f16[2048, 2048]", arg96_1: "f16[2048]", arg97_1: "f16[8192, 2048]", arg98_1: "f16[8192]", arg99_1: "f16[2048, 8192]", arg100_1: "f16[2048]", arg101_1: "f16[2048]", arg102_1: "f16[2048]", arg103_1: "f16[2048, 2048]", arg104_1: "f16[2048]", arg105_1: "f16[2048, 2048]", arg106_1: "f16[2048]", arg107_1: "f16[2048, 2048]", arg108_1: "f16[2048]", arg109_1: "f16[2048, 2048]", arg110_1: "f16[2048]", arg111_1: "f16[8192, 2048]", arg112_1: "f16[8192]", arg113_1: "f16[2048, 8192]", arg114_1: "f16[2048]", arg115_1: "f16[2048]", arg116_1: "f16[2048]", arg117_1: "f16[2048, 2048]", arg118_1: "f16[2048]", arg119_1: "f16[2048, 2048]", arg120_1: "f16[2048]", arg121_1: "f16[2048, 2048]", arg122_1: "f16[2048]", arg123_1: "f16[2048, 2048]", arg124_1: "f16[2048]", arg125_1: "f16[8192, 2048]", arg126_1: "f16[8192]", arg127_1: "f16[2048, 8192]", arg128_1: "f16[2048]", arg129_1: "f16[2048]", arg130_1: "f16[2048]", arg131_1: "f16[2048, 2048]", arg132_1: "f16[2048]", arg133_1: "f16[2048, 2048]", arg134_1: "f16[2048]", arg135_1: "f16[2048, 2048]", arg136_1: "f16[2048]", arg137_1: "f16[2048, 2048]", arg138_1: "f16[2048]", arg139_1: "f16[8192, 2048]", arg140_1: "f16[8192]", arg141_1: "f16[2048, 8192]", arg142_1: "f16[2048]", arg143_1: "f16[2048]", arg144_1: "f16[2048]", arg145_1: "f16[2048, 2048]", arg146_1: "f16[2048]", arg147_1: "f16[2048, 2048]", arg148_1: "f16[2048]", arg149_1: "f16[2048, 2048]", arg150_1: "f16[2048]", arg151_1: "f16[2048, 2048]", arg152_1: "f16[2048]", arg153_1: "f16[8192, 2048]", arg154_1: "f16[8192]", arg155_1: "f16[2048, 8192]", arg156_1: "f16[2048]", arg157_1: "f16[2048]", arg158_1: "f16[2048]", arg159_1: "f16[2048, 2048]", arg160_1: "f16[2048]", arg161_1: "f16[2048, 2048]", arg162_1: "f16[2048]", arg163_1: "f16[2048, 2048]", arg164_1: "f16[2048]", arg165_1: "f16[2048, 2048]", arg166_1: "f16[2048]", arg167_1: "f16[8192, 2048]", arg168_1: "f16[8192]", arg169_1: "f16[2048, 8192]", arg170_1: "f16[2048]", arg171_1: "f16[2048]", arg172_1: "f16[2048]", arg173_1: "f16[2048, 2048]", arg174_1: "f16[2048]", arg175_1: "f16[2048, 2048]", arg176_1: "f16[2048]", arg177_1: "f16[2048, 2048]", arg178_1: "f16[2048]", arg179_1: "f16[2048, 2048]", arg180_1: "f16[2048]", arg181_1: "f16[8192, 2048]", arg182_1: "f16[8192]", arg183_1: "f16[2048, 8192]", arg184_1: "f16[2048]", arg185_1: "f16[2048]", arg186_1: "f16[2048]", arg187_1: "f16[2048, 2048]", arg188_1: "f16[2048]", arg189_1: "f16[2048, 2048]", arg190_1: "f16[2048]", arg191_1: "f16[2048, 2048]", arg192_1: "f16[2048]", arg193_1: "f16[2048, 2048]", arg194_1: "f16[2048]", arg195_1: "f16[8192, 2048]", arg196_1: "f16[8192]", arg197_1: "f16[2048, 8192]", arg198_1: "f16[2048]", arg199_1: "f16[2048]", arg200_1: "f16[2048]", arg201_1: "f16[2048, 2048]", arg202_1: "f16[2048]", arg203_1: "f16[2048, 2048]", arg204_1: "f16[2048]", arg205_1: "f16[2048, 2048]", arg206_1: "f16[2048]", arg207_1: "f16[2048, 2048]", arg208_1: "f16[2048]", arg209_1: "f16[8192, 2048]", arg210_1: "f16[8192]", arg211_1: "f16[2048, 8192]", arg212_1: "f16[2048]", arg213_1: "f16[2048]", arg214_1: "f16[2048]", arg215_1: "f16[2048, 2048]", arg216_1: "f16[2048]", arg217_1: "f16[2048, 2048]", arg218_1: "f16[2048]", arg219_1: "f16[2048, 2048]", arg220_1: "f16[2048]", arg221_1: "f16[2048, 2048]", arg222_1: "f16[2048]", arg223_1: "f16[8192, 2048]", arg224_1: "f16[8192]", arg225_1: "f16[2048, 8192]", arg226_1: "f16[2048]", arg227_1: "f16[2048]", arg228_1: "f16[2048]", arg229_1: "f16[2048, 2048]", arg230_1: "f16[2048]", arg231_1: "f16[2048, 2048]", arg232_1: "f16[2048]", arg233_1: "f16[2048, 2048]", arg234_1: "f16[2048]", arg235_1: "f16[2048, 2048]", arg236_1: "f16[2048]", arg237_1: "f16[8192, 2048]", arg238_1: "f16[8192]", arg239_1: "f16[2048, 8192]", arg240_1: "f16[2048]", arg241_1: "f16[2048]", arg242_1: "f16[2048]", arg243_1: "f16[2048, 2048]", arg244_1: "f16[2048]", arg245_1: "f16[2048, 2048]", arg246_1: "f16[2048]", arg247_1: "f16[2048, 2048]", arg248_1: "f16[2048]", arg249_1: "f16[2048, 2048]", arg250_1: "f16[2048]", arg251_1: "f16[8192, 2048]", arg252_1: "f16[8192]", arg253_1: "f16[2048, 8192]", arg254_1: "f16[2048]", arg255_1: "f16[2048]", arg256_1: "f16[2048]", arg257_1: "f16[2048, 2048]", arg258_1: "f16[2048]", arg259_1: "f16[2048, 2048]", arg260_1: "f16[2048]", arg261_1: "f16[2048, 2048]", arg262_1: "f16[2048]", arg263_1: "f16[2048, 2048]", arg264_1: "f16[2048]", arg265_1: "f16[8192, 2048]", arg266_1: "f16[8192]", arg267_1: "f16[2048, 8192]", arg268_1: "f16[2048]", arg269_1: "f16[2048]", arg270_1: "f16[2048]", arg271_1: "f16[2048, 2048]", arg272_1: "f16[2048]", arg273_1: "f16[2048, 2048]", arg274_1: "f16[2048]", arg275_1: "f16[2048, 2048]", arg276_1: "f16[2048]", arg277_1: "f16[2048, 2048]", arg278_1: "f16[2048]", arg279_1: "f16[8192, 2048]", arg280_1: "f16[8192]", arg281_1: "f16[2048, 8192]", arg282_1: "f16[2048]", arg283_1: "f16[2048]", arg284_1: "f16[2048]", arg285_1: "f16[2048, 2048]", arg286_1: "f16[2048]", arg287_1: "f16[2048, 2048]", arg288_1: "f16[2048]", arg289_1: "f16[2048, 2048]", arg290_1: "f16[2048]", arg291_1: "f16[2048, 2048]", arg292_1: "f16[2048]", arg293_1: "f16[8192, 2048]", arg294_1: "f16[8192]", arg295_1: "f16[2048, 8192]", arg296_1: "f16[2048]", arg297_1: "f16[2048]", arg298_1: "f16[2048]", arg299_1: "f16[2048, 2048]", arg300_1: "f16[2048]", arg301_1: "f16[2048, 2048]", arg302_1: "f16[2048]", arg303_1: "f16[2048, 2048]", arg304_1: "f16[2048]", arg305_1: "f16[2048, 2048]", arg306_1: "f16[2048]", arg307_1: "f16[8192, 2048]", arg308_1: "f16[8192]", arg309_1: "f16[2048, 8192]", arg310_1: "f16[2048]", arg311_1: "f16[2048]", arg312_1: "f16[2048]", arg313_1: "f16[2048, 2048]", arg314_1: "f16[2048]", arg315_1: "f16[2048, 2048]", arg316_1: "f16[2048]", arg317_1: "f16[2048, 2048]", arg318_1: "f16[2048]", arg319_1: "f16[2048, 2048]", arg320_1: "f16[2048]", arg321_1: "f16[8192, 2048]", arg322_1: "f16[8192]", arg323_1: "f16[2048, 8192]", arg324_1: "f16[2048]", arg325_1: "f16[2048]", arg326_1: "f16[2048]", arg327_1: "f16[2048, 2048]", arg328_1: "f16[2048]", arg329_1: "f16[2048, 2048]", arg330_1: "f16[2048]", arg331_1: "f16[2048, 2048]", arg332_1: "f16[2048]", arg333_1: "f16[2048, 2048]", arg334_1: "f16[2048]", arg335_1: "f16[8192, 2048]", arg336_1: "f16[8192]", arg337_1: "f16[2048, 8192]", arg338_1: "f16[2048]", arg339_1: "f16[2048]", arg340_1: "f16[2048]", arg341_1: "f16[51200, 2048]", arg342_1: "f16[51200]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:365 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "f16[1, 512, 2048]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_4: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean[0]
        getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
        sub_2: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_1);  convert_element_type_4 = getitem_1 = None
        add_3: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_3: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_4: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_3, arg3_1);  mul_3 = arg3_1 = None
        add_4: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_4, arg4_1);  mul_4 = arg4_1 = None
        convert_element_type_5: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_4, torch.float16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_4: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_5, [512, 2048])
        permute_1: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg6_1, view_4, permute_1);  arg6_1 = view_4 = permute_1 = None
        view_5: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm, [1, 512, 2048]);  addmm = None
        view_6: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_5, [1, 512, -1, 64]);  view_5 = None
        permute_2: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_4: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_2, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:87 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_10: "f16[1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_11: "f16[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        convert_element_type: "f32[1, 16, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_11, torch.float32);  unsqueeze_11 = None
        expand_1: "f32[1, 16, 1]" = torch.ops.aten.expand.default(convert_element_type, [1, -1, 1]);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:92 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_2: "f32[1, 16, 1]" = torch.ops.aten.expand.default(expand_1, [1, 16, 1]);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:372 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:373 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:88 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_12: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        convert_element_type_1: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_12, torch.float32);  unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:92 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_3: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_1, [1, 1, 512]);  convert_element_type_1 = None
        mul: "f32[1, 16, 512]" = torch.ops.aten.mul.Tensor(expand_2, expand_3);  expand_2 = expand_3 = None
        permute: "f32[1, 512, 16]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:93 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_13: "f32[1, 512, 1, 16]" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_4: "f32[1, 512, 2, 16]" = torch.ops.aten.expand.default(unsqueeze_13, [1, 512, 2, 16]);  unsqueeze_13 = None
        clone_1: "f32[1, 512, 2, 16]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_3: "f32[1, 512, 32]" = torch.ops.aten.reshape.default(clone_1, [1, 512, 32]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:94 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 512, 32]" = torch.ops.aten.cos.default(view_3)
        mul_1: "f32[1, 512, 32]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:97 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "f16[1, 512, 32]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float16);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_14: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_5: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_4, unsqueeze_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_9: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_4, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_9);  slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_8: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 16);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_1: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg, slice_8], -1);  neg = slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:95 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 512, 32]" = torch.ops.aten.sin.default(view_3);  view_3 = None
        mul_2: "f32[1, 512, 32]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:97 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_3: "f16[1, 512, 32]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_15: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_6: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_15);  cat_1 = None
        add_5: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_5: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_2, 3, 32, 9223372036854775807);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_3: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_5, slice_5], -1);  add_5 = slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_7: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_5, [512, 2048])
        permute_3: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg8_1, view_7, permute_3);  arg8_1 = view_7 = permute_3 = None
        view_8: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 2048]);  addmm_1 = None
        view_9: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_8, [1, 512, -1, 64]);  view_8 = None
        permute_4: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_6: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_4, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_7: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_6, unsqueeze_14);  unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_11: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_6, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_11);  slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_10: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 16);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_2: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_1, slice_10], -1);  neg_1 = slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_8: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_15);  cat_2 = unsqueeze_15 = None
        add_6: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_7: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_4, 3, 32, 9223372036854775807);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_4: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_6, slice_7], -1);  add_6 = slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_10: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_5, [512, 2048])
        permute_5: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg10_1, view_10, permute_5);  arg10_1 = view_10 = permute_5 = None
        view_11: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_2, [1, 512, 2048]);  addmm_2 = None
        view_12: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_11, [1, 512, -1, 64]);  view_11 = None
        permute_6: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None

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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[1, 513]" = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = unsqueeze = None
        slice_3: "i64[1, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 513)
        slice_2: "i64[1, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 512);  cat = None
        sub_1: "i64[1, 512]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[1, 512]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[1, 512]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[1, 1, 512, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[1, 1, 1, 512]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[1, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_1, [1, -1, 512, 512]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_3, cat_4, permute_6, where, False, scale = 0.125);  cat_3 = cat_4 = permute_6 = where = None
        getitem_2: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        clone_3: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_13: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_3, [1, 512, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_14: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_13, [512, 2048]);  view_13 = None
        permute_8: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg12_1, view_14, permute_8);  arg12_1 = view_14 = permute_8 = None
        view_15: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 2048]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_16: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_5, [512, 2048]);  convert_element_type_5 = None
        permute_9: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_4: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg14_1, view_16, permute_9);  arg14_1 = view_16 = permute_9 = None
        view_17: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 8192]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_9: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
        pow_1: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 3.0)
        mul_10: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_17, mul_10);  view_17 = mul_10 = None
        mul_11: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_11);  mul_11 = None
        add_8: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_12: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_9, add_8);  mul_9 = add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_18: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_12, [512, 8192]);  mul_12 = None
        permute_10: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_5: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg16_1, view_18, permute_10);  arg16_1 = view_18 = permute_10 = None
        view_19: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 2048]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_9: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_15, view_19);  view_15 = view_19 = None
        add_10: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_9, embedding);  add_9 = embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_24: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_11: "f32[1, 512, 1]" = var_mean_1[0]
        getitem_12: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_3: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_12);  convert_element_type_24 = getitem_12 = None
        add_11: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_13: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_14: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_13, arg17_1);  mul_13 = arg17_1 = None
        add_12: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_14, arg18_1);  mul_14 = arg18_1 = None
        convert_element_type_25: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_12, torch.float16);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_20: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_25, [512, 2048])
        permute_11: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_6: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg20_1, view_20, permute_11);  arg20_1 = view_20 = permute_11 = None
        view_21: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_6, [1, 512, 2048]);  addmm_6 = None
        view_22: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_21, [1, 512, -1, 64]);  view_21 = None
        permute_12: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_12: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_12, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_16: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_15: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_12, unsqueeze_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_17: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_12, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_2: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_17);  slice_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_16: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_12, 3, 0, 16);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_5: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_2, slice_16], -1);  neg_2 = slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_17: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_16: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_17);  cat_5 = None
        add_13: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_13: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_12, 3, 32, 9223372036854775807);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_7: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_13, slice_13], -1);  add_13 = slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_23: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_25, [512, 2048])
        permute_13: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_7: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg22_1, view_23, permute_13);  arg22_1 = view_23 = permute_13 = None
        view_24: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 2048]);  addmm_7 = None
        view_25: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_24, [1, 512, -1, 64]);  view_24 = None
        permute_14: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_14: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_14, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_17: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_14, unsqueeze_16);  unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_19: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_14, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_3: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_19);  slice_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_18: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 16);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_6: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_3, slice_18], -1);  neg_3 = slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_18: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_17);  cat_6 = unsqueeze_17 = None
        add_14: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_15: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_14, 3, 32, 9223372036854775807);  permute_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_8: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_14, slice_15], -1);  add_14 = slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_26: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_25, [512, 2048])
        permute_15: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_8: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg24_1, view_26, permute_15);  arg24_1 = view_26 = permute_15 = None
        view_27: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_8, [1, 512, 2048]);  addmm_8 = None
        view_28: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_27, [1, 512, -1, 64]);  view_27 = None
        permute_16: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_4: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_7, cat_8, permute_16, where_1, False, scale = 0.125);  cat_7 = cat_8 = permute_16 = where_1 = None
        getitem_13: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_17: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_13, [0, 2, 1, 3]);  getitem_13 = None
        clone_6: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_29: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_6, [1, 512, -1]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_30: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_29, [512, 2048]);  view_29 = None
        permute_18: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_9: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg26_1, view_30, permute_18);  arg26_1 = view_30 = permute_18 = None
        view_31: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_9, [1, 512, 2048]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_32: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_25, [512, 2048]);  convert_element_type_25 = None
        permute_19: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_10: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg28_1, view_32, permute_19);  arg28_1 = view_32 = permute_19 = None
        view_33: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 8192]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_19: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
        pow_2: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_33, 3.0)
        mul_20: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_33, mul_20);  view_33 = mul_20 = None
        mul_21: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_21);  mul_21 = None
        add_16: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_22: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_19, add_16);  mul_19 = add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_34: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_22, [512, 8192]);  mul_22 = None
        permute_20: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_11: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg30_1, view_34, permute_20);  arg30_1 = view_34 = permute_20 = None
        view_35: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 2048]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_17: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_31, view_35);  view_31 = view_35 = None
        add_18: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_17, add_10);  add_17 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_44: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_18, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_44, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 512, 1]" = var_mean_2[0]
        getitem_23: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_4: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_44, getitem_23);  convert_element_type_44 = getitem_23 = None
        add_19: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_23: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_24: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_23, arg31_1);  mul_23 = arg31_1 = None
        add_20: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_24, arg32_1);  mul_24 = arg32_1 = None
        convert_element_type_45: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_20, torch.float16);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_36: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_45, [512, 2048])
        permute_21: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_12: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg34_1, view_36, permute_21);  arg34_1 = view_36 = permute_21 = None
        view_37: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_12, [1, 512, 2048]);  addmm_12 = None
        view_38: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_37, [1, 512, -1, 64]);  view_37 = None
        permute_22: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_20: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_22, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_18: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_25: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_20, unsqueeze_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_25: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_20, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_4: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_25);  slice_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_24: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 16);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_9: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_4, slice_24], -1);  neg_4 = slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_19: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_26: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_19);  cat_9 = None
        add_21: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_21: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_22, 3, 32, 9223372036854775807);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_11: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_21, slice_21], -1);  add_21 = slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_39: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_45, [512, 2048])
        permute_23: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_13: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg36_1, view_39, permute_23);  arg36_1 = view_39 = permute_23 = None
        view_40: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 2048]);  addmm_13 = None
        view_41: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_40, [1, 512, -1, 64]);  view_40 = None
        permute_24: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_22: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_24, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_27: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_22, unsqueeze_18);  unsqueeze_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_27: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_22, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_5: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_27);  slice_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_26: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 16);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_10: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_5, slice_26], -1);  neg_5 = slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_28: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_19);  cat_10 = unsqueeze_19 = None
        add_22: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_27, mul_28);  mul_27 = mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_23: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_24, 3, 32, 9223372036854775807);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_12: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_22, slice_23], -1);  add_22 = slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_42: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_45, [512, 2048])
        permute_25: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_14: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg38_1, view_42, permute_25);  arg38_1 = view_42 = permute_25 = None
        view_43: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_14, [1, 512, 2048]);  addmm_14 = None
        view_44: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_43, [1, 512, -1, 64]);  view_43 = None
        permute_26: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_6: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_11, cat_12, permute_26, where_2, False, scale = 0.125);  cat_11 = cat_12 = permute_26 = where_2 = None
        getitem_24: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_24, [0, 2, 1, 3]);  getitem_24 = None
        clone_9: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_45: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_9, [1, 512, -1]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_46: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_45, [512, 2048]);  view_45 = None
        permute_28: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_15: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg40_1, view_46, permute_28);  arg40_1 = view_46 = permute_28 = None
        view_47: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_15, [1, 512, 2048]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_48: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_45, [512, 2048]);  convert_element_type_45 = None
        permute_29: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_16: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg42_1, view_48, permute_29);  arg42_1 = view_48 = permute_29 = None
        view_49: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 8192]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_29: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_49, 0.5)
        pow_3: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_49, 3.0)
        mul_30: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_49, mul_30);  view_49 = mul_30 = None
        mul_31: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_31);  mul_31 = None
        add_24: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_32: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_29, add_24);  mul_29 = add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_50: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_32, [512, 8192]);  mul_32 = None
        permute_30: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_17: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg44_1, view_50, permute_30);  arg44_1 = view_50 = permute_30 = None
        view_51: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 2048]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_25: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_47, view_51);  view_47 = view_51 = None
        add_26: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_25, add_18);  add_25 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_64: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_64, [2], correction = 0, keepdim = True)
        getitem_33: "f32[1, 512, 1]" = var_mean_3[0]
        getitem_34: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_64, getitem_34);  convert_element_type_64 = getitem_34 = None
        add_27: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_33: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_34: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_33, arg45_1);  mul_33 = arg45_1 = None
        add_28: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_34, arg46_1);  mul_34 = arg46_1 = None
        convert_element_type_65: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_28, torch.float16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_52: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_65, [512, 2048])
        permute_31: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_18: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg48_1, view_52, permute_31);  arg48_1 = view_52 = permute_31 = None
        view_53: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_18, [1, 512, 2048]);  addmm_18 = None
        view_54: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_53, [1, 512, -1, 64]);  view_53 = None
        permute_32: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_28: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_32, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_20: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_35: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_28, unsqueeze_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_33: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_28, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_6: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_33);  slice_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_32: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_28, 3, 0, 16);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_13: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_6, slice_32], -1);  neg_6 = slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_21: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_36: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_21);  cat_13 = None
        add_29: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_29: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_32, 3, 32, 9223372036854775807);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_15: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_29, slice_29], -1);  add_29 = slice_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_55: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_65, [512, 2048])
        permute_33: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_19: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg50_1, view_55, permute_33);  arg50_1 = view_55 = permute_33 = None
        view_56: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 2048]);  addmm_19 = None
        view_57: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_56, [1, 512, -1, 64]);  view_56 = None
        permute_34: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_30: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_34, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_37: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_30, unsqueeze_20);  unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_35: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_30, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_7: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_35);  slice_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_34: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_30, 3, 0, 16);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_14: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_7, slice_34], -1);  neg_7 = slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_38: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_14, unsqueeze_21);  cat_14 = unsqueeze_21 = None
        add_30: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_31: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_34, 3, 32, 9223372036854775807);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_16: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_30, slice_31], -1);  add_30 = slice_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_58: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_65, [512, 2048])
        permute_35: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_20: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg52_1, view_58, permute_35);  arg52_1 = view_58 = permute_35 = None
        view_59: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_20, [1, 512, 2048]);  addmm_20 = None
        view_60: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_59, [1, 512, -1, 64]);  view_59 = None
        permute_36: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_8: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_15, cat_16, permute_36, where_3, False, scale = 0.125);  cat_15 = cat_16 = permute_36 = where_3 = None
        getitem_35: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_37: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_35, [0, 2, 1, 3]);  getitem_35 = None
        clone_12: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_61: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_12, [1, 512, -1]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_62: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_61, [512, 2048]);  view_61 = None
        permute_38: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_21: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg54_1, view_62, permute_38);  arg54_1 = view_62 = permute_38 = None
        view_63: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 2048]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_64: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_65, [512, 2048]);  convert_element_type_65 = None
        permute_39: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_22: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg56_1, view_64, permute_39);  arg56_1 = view_64 = permute_39 = None
        view_65: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 8192]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_39: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
        pow_4: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
        mul_40: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_65, mul_40);  view_65 = mul_40 = None
        mul_41: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_41);  mul_41 = None
        add_32: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_42: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_39, add_32);  mul_39 = add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_66: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_42, [512, 8192]);  mul_42 = None
        permute_40: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_23: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg58_1, view_66, permute_40);  arg58_1 = view_66 = permute_40 = None
        view_67: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 2048]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_33: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_63, view_67);  view_63 = view_67 = None
        add_34: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_33, add_26);  add_33 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_84: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_84, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 512, 1]" = var_mean_4[0]
        getitem_45: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_84, getitem_45);  convert_element_type_84 = getitem_45 = None
        add_35: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_43: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_44: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_43, arg59_1);  mul_43 = arg59_1 = None
        add_36: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_44, arg60_1);  mul_44 = arg60_1 = None
        convert_element_type_85: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_36, torch.float16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_68: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_85, [512, 2048])
        permute_41: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_24: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg62_1, view_68, permute_41);  arg62_1 = view_68 = permute_41 = None
        view_69: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_24, [1, 512, 2048]);  addmm_24 = None
        view_70: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_69, [1, 512, -1, 64]);  view_69 = None
        permute_42: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_36: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_42, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_22: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_45: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_36, unsqueeze_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_41: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_36, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_8: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_41);  slice_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_40: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_36, 3, 0, 16);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_17: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_8, slice_40], -1);  neg_8 = slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_23: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_46: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_17, unsqueeze_23);  cat_17 = None
        add_37: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_45, mul_46);  mul_45 = mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_37: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_42, 3, 32, 9223372036854775807);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_19: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_37, slice_37], -1);  add_37 = slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_71: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_85, [512, 2048])
        permute_43: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_25: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg64_1, view_71, permute_43);  arg64_1 = view_71 = permute_43 = None
        view_72: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_25, [1, 512, 2048]);  addmm_25 = None
        view_73: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_72, [1, 512, -1, 64]);  view_72 = None
        permute_44: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_38: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_44, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_47: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_38, unsqueeze_22);  unsqueeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_43: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_38, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_9: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_43);  slice_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_42: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 16);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_18: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_9, slice_42], -1);  neg_9 = slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_48: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_18, unsqueeze_23);  cat_18 = unsqueeze_23 = None
        add_38: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_39: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_44, 3, 32, 9223372036854775807);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_20: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_38, slice_39], -1);  add_38 = slice_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_74: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_85, [512, 2048])
        permute_45: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_26: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg66_1, view_74, permute_45);  arg66_1 = view_74 = permute_45 = None
        view_75: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_26, [1, 512, 2048]);  addmm_26 = None
        view_76: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_75, [1, 512, -1, 64]);  view_75 = None
        permute_46: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_10: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_19, cat_20, permute_46, where_4, False, scale = 0.125);  cat_19 = cat_20 = permute_46 = where_4 = None
        getitem_46: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_46, [0, 2, 1, 3]);  getitem_46 = None
        clone_15: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_77: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_15, [1, 512, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_78: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_77, [512, 2048]);  view_77 = None
        permute_48: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_27: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg68_1, view_78, permute_48);  arg68_1 = view_78 = permute_48 = None
        view_79: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_27, [1, 512, 2048]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_80: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_85, [512, 2048]);  convert_element_type_85 = None
        permute_49: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_28: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg70_1, view_80, permute_49);  arg70_1 = view_80 = permute_49 = None
        view_81: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 8192]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_49: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_81, 0.5)
        pow_5: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_81, 3.0)
        mul_50: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_81, mul_50);  view_81 = mul_50 = None
        mul_51: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_51);  mul_51 = None
        add_40: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_52: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_49, add_40);  mul_49 = add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_82: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_52, [512, 8192]);  mul_52 = None
        permute_50: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_29: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg72_1, view_82, permute_50);  arg72_1 = view_82 = permute_50 = None
        view_83: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_29, [1, 512, 2048]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_41: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_79, view_83);  view_79 = view_83 = None
        add_42: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_41, add_34);  add_41 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_104: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_42, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_104, [2], correction = 0, keepdim = True)
        getitem_55: "f32[1, 512, 1]" = var_mean_5[0]
        getitem_56: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_7: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_104, getitem_56);  convert_element_type_104 = getitem_56 = None
        add_43: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_53: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_54: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_53, arg73_1);  mul_53 = arg73_1 = None
        add_44: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_54, arg74_1);  mul_54 = arg74_1 = None
        convert_element_type_105: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_44, torch.float16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_84: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_105, [512, 2048])
        permute_51: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_30: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg76_1, view_84, permute_51);  arg76_1 = view_84 = permute_51 = None
        view_85: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_30, [1, 512, 2048]);  addmm_30 = None
        view_86: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_85, [1, 512, -1, 64]);  view_85 = None
        permute_52: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_44: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_52, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_24: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_55: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_44, unsqueeze_24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_49: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_44, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_10: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_49);  slice_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_48: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_44, 3, 0, 16);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_21: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_10, slice_48], -1);  neg_10 = slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_25: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_56: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_21, unsqueeze_25);  cat_21 = None
        add_45: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_55, mul_56);  mul_55 = mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_45: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_52, 3, 32, 9223372036854775807);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_23: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_45, slice_45], -1);  add_45 = slice_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_87: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_105, [512, 2048])
        permute_53: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_31: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg78_1, view_87, permute_53);  arg78_1 = view_87 = permute_53 = None
        view_88: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_31, [1, 512, 2048]);  addmm_31 = None
        view_89: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_88, [1, 512, -1, 64]);  view_88 = None
        permute_54: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_46: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_54, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_57: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_46, unsqueeze_24);  unsqueeze_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_51: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_46, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_11: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_51);  slice_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_50: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 16);  slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_22: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_11, slice_50], -1);  neg_11 = slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_58: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_22, unsqueeze_25);  cat_22 = unsqueeze_25 = None
        add_46: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_47: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_54, 3, 32, 9223372036854775807);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_24: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_46, slice_47], -1);  add_46 = slice_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_90: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_105, [512, 2048])
        permute_55: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_32: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg80_1, view_90, permute_55);  arg80_1 = view_90 = permute_55 = None
        view_91: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_32, [1, 512, 2048]);  addmm_32 = None
        view_92: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_91, [1, 512, -1, 64]);  view_91 = None
        permute_56: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_12: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_23, cat_24, permute_56, where_5, False, scale = 0.125);  cat_23 = cat_24 = permute_56 = where_5 = None
        getitem_57: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_57: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
        clone_18: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_93: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_18, [1, 512, -1]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_94: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_93, [512, 2048]);  view_93 = None
        permute_58: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_33: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg82_1, view_94, permute_58);  arg82_1 = view_94 = permute_58 = None
        view_95: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_33, [1, 512, 2048]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_96: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_105, [512, 2048]);  convert_element_type_105 = None
        permute_59: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_34: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg84_1, view_96, permute_59);  arg84_1 = view_96 = permute_59 = None
        view_97: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 8192]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_59: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_97, 0.5)
        pow_6: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_97, 3.0)
        mul_60: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_97, mul_60);  view_97 = mul_60 = None
        mul_61: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_61);  mul_61 = None
        add_48: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_62: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_59, add_48);  mul_59 = add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_98: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_62, [512, 8192]);  mul_62 = None
        permute_60: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_35: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg86_1, view_98, permute_60);  arg86_1 = view_98 = permute_60 = None
        view_99: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_35, [1, 512, 2048]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_49: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_95, view_99);  view_95 = view_99 = None
        add_50: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_49, add_42);  add_49 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_124: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_50, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_124, [2], correction = 0, keepdim = True)
        getitem_66: "f32[1, 512, 1]" = var_mean_6[0]
        getitem_67: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_8: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_124, getitem_67);  convert_element_type_124 = getitem_67 = None
        add_51: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_63: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_64: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_63, arg87_1);  mul_63 = arg87_1 = None
        add_52: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_64, arg88_1);  mul_64 = arg88_1 = None
        convert_element_type_125: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_52, torch.float16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_100: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_125, [512, 2048])
        permute_61: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_36: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg90_1, view_100, permute_61);  arg90_1 = view_100 = permute_61 = None
        view_101: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_36, [1, 512, 2048]);  addmm_36 = None
        view_102: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_101, [1, 512, -1, 64]);  view_101 = None
        permute_62: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_52: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_62, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_26: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_65: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_52, unsqueeze_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_57: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_52, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_12: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_57);  slice_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_56: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_52, 3, 0, 16);  slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_25: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_12, slice_56], -1);  neg_12 = slice_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_27: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_66: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_25, unsqueeze_27);  cat_25 = None
        add_53: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_53: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_62, 3, 32, 9223372036854775807);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_27: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_53, slice_53], -1);  add_53 = slice_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_103: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_125, [512, 2048])
        permute_63: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_37: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg92_1, view_103, permute_63);  arg92_1 = view_103 = permute_63 = None
        view_104: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_37, [1, 512, 2048]);  addmm_37 = None
        view_105: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_104, [1, 512, -1, 64]);  view_104 = None
        permute_64: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_54: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_64, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_67: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_54, unsqueeze_26);  unsqueeze_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_59: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_54, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_13: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_59);  slice_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_58: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_54, 3, 0, 16);  slice_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_26: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_13, slice_58], -1);  neg_13 = slice_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_68: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_26, unsqueeze_27);  cat_26 = unsqueeze_27 = None
        add_54: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_55: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_64, 3, 32, 9223372036854775807);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_28: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_54, slice_55], -1);  add_54 = slice_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_106: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_125, [512, 2048])
        permute_65: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_38: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg94_1, view_106, permute_65);  arg94_1 = view_106 = permute_65 = None
        view_107: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_38, [1, 512, 2048]);  addmm_38 = None
        view_108: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_107, [1, 512, -1, 64]);  view_107 = None
        permute_66: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_14: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_27, cat_28, permute_66, where_6, False, scale = 0.125);  cat_27 = cat_28 = permute_66 = where_6 = None
        getitem_68: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_67: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_68, [0, 2, 1, 3]);  getitem_68 = None
        clone_21: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_109: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_21, [1, 512, -1]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_110: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_109, [512, 2048]);  view_109 = None
        permute_68: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_39: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg96_1, view_110, permute_68);  arg96_1 = view_110 = permute_68 = None
        view_111: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_39, [1, 512, 2048]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_112: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_125, [512, 2048]);  convert_element_type_125 = None
        permute_69: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_40: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg98_1, view_112, permute_69);  arg98_1 = view_112 = permute_69 = None
        view_113: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_40, [1, 512, 8192]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_69: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_113, 0.5)
        pow_7: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_113, 3.0)
        mul_70: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_113, mul_70);  view_113 = mul_70 = None
        mul_71: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_71);  mul_71 = None
        add_56: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_72: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_69, add_56);  mul_69 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_114: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_72, [512, 8192]);  mul_72 = None
        permute_70: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_41: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg100_1, view_114, permute_70);  arg100_1 = view_114 = permute_70 = None
        view_115: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_41, [1, 512, 2048]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_57: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_111, view_115);  view_111 = view_115 = None
        add_58: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_57, add_50);  add_57 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_144: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_77: "f32[1, 512, 1]" = var_mean_7[0]
        getitem_78: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_9: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_78);  convert_element_type_144 = getitem_78 = None
        add_59: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_73: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = rsqrt_7 = None
        mul_74: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_73, arg101_1);  mul_73 = arg101_1 = None
        add_60: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_74, arg102_1);  mul_74 = arg102_1 = None
        convert_element_type_145: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_60, torch.float16);  add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_116: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_145, [512, 2048])
        permute_71: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_42: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg104_1, view_116, permute_71);  arg104_1 = view_116 = permute_71 = None
        view_117: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_42, [1, 512, 2048]);  addmm_42 = None
        view_118: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_117, [1, 512, -1, 64]);  view_117 = None
        permute_72: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_60: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_72, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_28: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_75: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_60, unsqueeze_28)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_65: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_60, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_14: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_65);  slice_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_64: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_60, 3, 0, 16);  slice_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_29: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_14, slice_64], -1);  neg_14 = slice_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_29: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_76: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_29, unsqueeze_29);  cat_29 = None
        add_61: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_61: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_72, 3, 32, 9223372036854775807);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_31: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_61, slice_61], -1);  add_61 = slice_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_119: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_145, [512, 2048])
        permute_73: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_43: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg106_1, view_119, permute_73);  arg106_1 = view_119 = permute_73 = None
        view_120: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_43, [1, 512, 2048]);  addmm_43 = None
        view_121: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_120, [1, 512, -1, 64]);  view_120 = None
        permute_74: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_62: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_74, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_77: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_62, unsqueeze_28);  unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_67: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_62, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_15: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_67);  slice_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_66: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_62, 3, 0, 16);  slice_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_30: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_15, slice_66], -1);  neg_15 = slice_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_78: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_30, unsqueeze_29);  cat_30 = unsqueeze_29 = None
        add_62: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_63: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_74, 3, 32, 9223372036854775807);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_32: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_62, slice_63], -1);  add_62 = slice_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_122: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_145, [512, 2048])
        permute_75: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_44: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg108_1, view_122, permute_75);  arg108_1 = view_122 = permute_75 = None
        view_123: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_44, [1, 512, 2048]);  addmm_44 = None
        view_124: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_123, [1, 512, -1, 64]);  view_123 = None
        permute_76: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_16: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_31, cat_32, permute_76, where_7, False, scale = 0.125);  cat_31 = cat_32 = permute_76 = where_7 = None
        getitem_79: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_77: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_79, [0, 2, 1, 3]);  getitem_79 = None
        clone_24: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_125: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_24, [1, 512, -1]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_126: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_125, [512, 2048]);  view_125 = None
        permute_78: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_45: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg110_1, view_126, permute_78);  arg110_1 = view_126 = permute_78 = None
        view_127: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_45, [1, 512, 2048]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_128: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_145, [512, 2048]);  convert_element_type_145 = None
        permute_79: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_46: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg112_1, view_128, permute_79);  arg112_1 = view_128 = permute_79 = None
        view_129: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_46, [1, 512, 8192]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_79: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        pow_8: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_129, 3.0)
        mul_80: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_129, mul_80);  view_129 = mul_80 = None
        mul_81: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_81);  mul_81 = None
        add_64: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_82: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_79, add_64);  mul_79 = add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_130: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_82, [512, 8192]);  mul_82 = None
        permute_80: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_47: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg114_1, view_130, permute_80);  arg114_1 = view_130 = permute_80 = None
        view_131: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_47, [1, 512, 2048]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_65: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_127, view_131);  view_127 = view_131 = None
        add_66: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_65, add_58);  add_65 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_164: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_66, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_164, [2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 512, 1]" = var_mean_8[0]
        getitem_89: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_10: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_164, getitem_89);  convert_element_type_164 = getitem_89 = None
        add_67: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_83: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = rsqrt_8 = None
        mul_84: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_83, arg115_1);  mul_83 = arg115_1 = None
        add_68: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_84, arg116_1);  mul_84 = arg116_1 = None
        convert_element_type_165: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_68, torch.float16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_132: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_165, [512, 2048])
        permute_81: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_48: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg118_1, view_132, permute_81);  arg118_1 = view_132 = permute_81 = None
        view_133: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_48, [1, 512, 2048]);  addmm_48 = None
        view_134: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_133, [1, 512, -1, 64]);  view_133 = None
        permute_82: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_68: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_82, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_30: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_85: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_68, unsqueeze_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_73: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_68, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_16: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_73);  slice_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_72: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_68, 3, 0, 16);  slice_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_33: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_16, slice_72], -1);  neg_16 = slice_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_31: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_86: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_33, unsqueeze_31);  cat_33 = None
        add_69: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_69: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_82, 3, 32, 9223372036854775807);  permute_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_35: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_69, slice_69], -1);  add_69 = slice_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_135: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_165, [512, 2048])
        permute_83: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_49: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg120_1, view_135, permute_83);  arg120_1 = view_135 = permute_83 = None
        view_136: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_49, [1, 512, 2048]);  addmm_49 = None
        view_137: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_136, [1, 512, -1, 64]);  view_136 = None
        permute_84: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_70: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_84, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_87: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_70, unsqueeze_30);  unsqueeze_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_75: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_70, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_17: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_75);  slice_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_74: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_70, 3, 0, 16);  slice_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_34: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_17, slice_74], -1);  neg_17 = slice_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_88: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_34, unsqueeze_31);  cat_34 = unsqueeze_31 = None
        add_70: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_71: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_84, 3, 32, 9223372036854775807);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_36: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_70, slice_71], -1);  add_70 = slice_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_138: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_165, [512, 2048])
        permute_85: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_50: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg122_1, view_138, permute_85);  arg122_1 = view_138 = permute_85 = None
        view_139: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_50, [1, 512, 2048]);  addmm_50 = None
        view_140: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_139, [1, 512, -1, 64]);  view_139 = None
        permute_86: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_18: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_18, full_default_17);  full_default_18 = full_default_17 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_35, cat_36, permute_86, where_8, False, scale = 0.125);  cat_35 = cat_36 = permute_86 = where_8 = None
        getitem_90: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_87: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        clone_27: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_141: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_27, [1, 512, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_142: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_141, [512, 2048]);  view_141 = None
        permute_88: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_51: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg124_1, view_142, permute_88);  arg124_1 = view_142 = permute_88 = None
        view_143: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_51, [1, 512, 2048]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_144: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_165, [512, 2048]);  convert_element_type_165 = None
        permute_89: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_52: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg126_1, view_144, permute_89);  arg126_1 = view_144 = permute_89 = None
        view_145: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_52, [1, 512, 8192]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_89: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_145, 0.5)
        pow_9: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_145, 3.0)
        mul_90: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_145, mul_90);  view_145 = mul_90 = None
        mul_91: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_91);  mul_91 = None
        add_72: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_92: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_89, add_72);  mul_89 = add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_146: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_92, [512, 8192]);  mul_92 = None
        permute_90: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_53: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg128_1, view_146, permute_90);  arg128_1 = view_146 = permute_90 = None
        view_147: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_53, [1, 512, 2048]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_73: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_143, view_147);  view_143 = view_147 = None
        add_74: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_73, add_66);  add_73 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_184: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_74, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_184, [2], correction = 0, keepdim = True)
        getitem_99: "f32[1, 512, 1]" = var_mean_9[0]
        getitem_100: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_11: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_184, getitem_100);  convert_element_type_184 = getitem_100 = None
        add_75: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_93: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = rsqrt_9 = None
        mul_94: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_93, arg129_1);  mul_93 = arg129_1 = None
        add_76: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_94, arg130_1);  mul_94 = arg130_1 = None
        convert_element_type_185: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_76, torch.float16);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_148: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_185, [512, 2048])
        permute_91: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_54: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg132_1, view_148, permute_91);  arg132_1 = view_148 = permute_91 = None
        view_149: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_54, [1, 512, 2048]);  addmm_54 = None
        view_150: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_149, [1, 512, -1, 64]);  view_149 = None
        permute_92: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_76: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_92, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_32: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_95: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_76, unsqueeze_32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_81: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_76, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_18: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_81);  slice_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_80: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_76, 3, 0, 16);  slice_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_37: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_18, slice_80], -1);  neg_18 = slice_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_33: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_96: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_37, unsqueeze_33);  cat_37 = None
        add_77: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_77: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_92, 3, 32, 9223372036854775807);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_39: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_77, slice_77], -1);  add_77 = slice_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_151: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_185, [512, 2048])
        permute_93: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_55: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg134_1, view_151, permute_93);  arg134_1 = view_151 = permute_93 = None
        view_152: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_55, [1, 512, 2048]);  addmm_55 = None
        view_153: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_152, [1, 512, -1, 64]);  view_152 = None
        permute_94: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_78: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_94, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_97: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_78, unsqueeze_32);  unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_83: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_78, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_19: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_83);  slice_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_82: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_78, 3, 0, 16);  slice_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_38: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_19, slice_82], -1);  neg_19 = slice_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_98: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_38, unsqueeze_33);  cat_38 = unsqueeze_33 = None
        add_78: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_79: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_94, 3, 32, 9223372036854775807);  permute_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_40: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_78, slice_79], -1);  add_78 = slice_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_154: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_185, [512, 2048])
        permute_95: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_56: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg136_1, view_154, permute_95);  arg136_1 = view_154 = permute_95 = None
        view_155: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_56, [1, 512, 2048]);  addmm_56 = None
        view_156: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_155, [1, 512, -1, 64]);  view_155 = None
        permute_96: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_20: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_20, full_default_19);  full_default_20 = full_default_19 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_39, cat_40, permute_96, where_9, False, scale = 0.125);  cat_39 = cat_40 = permute_96 = where_9 = None
        getitem_101: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_97: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
        clone_30: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_157: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_30, [1, 512, -1]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_158: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_157, [512, 2048]);  view_157 = None
        permute_98: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_57: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg138_1, view_158, permute_98);  arg138_1 = view_158 = permute_98 = None
        view_159: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_57, [1, 512, 2048]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_160: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_185, [512, 2048]);  convert_element_type_185 = None
        permute_99: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_58: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg140_1, view_160, permute_99);  arg140_1 = view_160 = permute_99 = None
        view_161: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_58, [1, 512, 8192]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_99: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
        pow_10: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_161, 3.0)
        mul_100: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_161, mul_100);  view_161 = mul_100 = None
        mul_101: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_101);  mul_101 = None
        add_80: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_102: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_99, add_80);  mul_99 = add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_162: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_102, [512, 8192]);  mul_102 = None
        permute_100: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_59: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg142_1, view_162, permute_100);  arg142_1 = view_162 = permute_100 = None
        view_163: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_59, [1, 512, 2048]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_81: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_159, view_163);  view_159 = view_163 = None
        add_82: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_81, add_74);  add_81 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_204: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_82, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_204, [2], correction = 0, keepdim = True)
        getitem_110: "f32[1, 512, 1]" = var_mean_10[0]
        getitem_111: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_12: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_204, getitem_111);  convert_element_type_204 = getitem_111 = None
        add_83: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_103: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = rsqrt_10 = None
        mul_104: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_103, arg143_1);  mul_103 = arg143_1 = None
        add_84: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_104, arg144_1);  mul_104 = arg144_1 = None
        convert_element_type_205: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_84, torch.float16);  add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_164: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_205, [512, 2048])
        permute_101: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_60: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg146_1, view_164, permute_101);  arg146_1 = view_164 = permute_101 = None
        view_165: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_60, [1, 512, 2048]);  addmm_60 = None
        view_166: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_165, [1, 512, -1, 64]);  view_165 = None
        permute_102: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_84: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_102, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_34: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_105: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_84, unsqueeze_34)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_89: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_84, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_20: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_89);  slice_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_88: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_84, 3, 0, 16);  slice_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_41: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_20, slice_88], -1);  neg_20 = slice_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_35: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_106: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_41, unsqueeze_35);  cat_41 = None
        add_85: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_85: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_102, 3, 32, 9223372036854775807);  permute_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_43: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_85, slice_85], -1);  add_85 = slice_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_167: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_205, [512, 2048])
        permute_103: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_61: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg148_1, view_167, permute_103);  arg148_1 = view_167 = permute_103 = None
        view_168: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_61, [1, 512, 2048]);  addmm_61 = None
        view_169: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_168, [1, 512, -1, 64]);  view_168 = None
        permute_104: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_86: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_104, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_107: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_86, unsqueeze_34);  unsqueeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_91: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_86, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_21: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_91);  slice_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_90: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_86, 3, 0, 16);  slice_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_42: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_21, slice_90], -1);  neg_21 = slice_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_108: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_42, unsqueeze_35);  cat_42 = unsqueeze_35 = None
        add_86: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_107, mul_108);  mul_107 = mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_87: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_104, 3, 32, 9223372036854775807);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_44: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_86, slice_87], -1);  add_86 = slice_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_170: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_205, [512, 2048])
        permute_105: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_62: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg150_1, view_170, permute_105);  arg150_1 = view_170 = permute_105 = None
        view_171: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_62, [1, 512, 2048]);  addmm_62 = None
        view_172: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_171, [1, 512, -1, 64]);  view_171 = None
        permute_106: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_22: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_43, cat_44, permute_106, where_10, False, scale = 0.125);  cat_43 = cat_44 = permute_106 = where_10 = None
        getitem_112: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_112, [0, 2, 1, 3]);  getitem_112 = None
        clone_33: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_173: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_33, [1, 512, -1]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_174: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_173, [512, 2048]);  view_173 = None
        permute_108: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_63: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg152_1, view_174, permute_108);  arg152_1 = view_174 = permute_108 = None
        view_175: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_63, [1, 512, 2048]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_176: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_205, [512, 2048]);  convert_element_type_205 = None
        permute_109: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_64: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg154_1, view_176, permute_109);  arg154_1 = view_176 = permute_109 = None
        view_177: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_64, [1, 512, 8192]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_109: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_177, 0.5)
        pow_11: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_177, 3.0)
        mul_110: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_177, mul_110);  view_177 = mul_110 = None
        mul_111: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_111);  mul_111 = None
        add_88: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_112: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_109, add_88);  mul_109 = add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_178: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_112, [512, 8192]);  mul_112 = None
        permute_110: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_65: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg156_1, view_178, permute_110);  arg156_1 = view_178 = permute_110 = None
        view_179: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_65, [1, 512, 2048]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_89: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_175, view_179);  view_175 = view_179 = None
        add_90: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_89, add_82);  add_89 = add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_224: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_90, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_224, [2], correction = 0, keepdim = True)
        getitem_121: "f32[1, 512, 1]" = var_mean_11[0]
        getitem_122: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_13: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_224, getitem_122);  convert_element_type_224 = getitem_122 = None
        add_91: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        mul_113: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = rsqrt_11 = None
        mul_114: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_113, arg157_1);  mul_113 = arg157_1 = None
        add_92: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_114, arg158_1);  mul_114 = arg158_1 = None
        convert_element_type_225: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_92, torch.float16);  add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_180: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_225, [512, 2048])
        permute_111: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_66: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg160_1, view_180, permute_111);  arg160_1 = view_180 = permute_111 = None
        view_181: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_66, [1, 512, 2048]);  addmm_66 = None
        view_182: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_181, [1, 512, -1, 64]);  view_181 = None
        permute_112: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_92: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_112, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_36: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_115: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_92, unsqueeze_36)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_97: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_92, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_22: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_97);  slice_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_96: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_92, 3, 0, 16);  slice_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_45: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_22, slice_96], -1);  neg_22 = slice_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_37: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_116: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_45, unsqueeze_37);  cat_45 = None
        add_93: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_93: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_112, 3, 32, 9223372036854775807);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_47: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_93, slice_93], -1);  add_93 = slice_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_183: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_225, [512, 2048])
        permute_113: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_67: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg162_1, view_183, permute_113);  arg162_1 = view_183 = permute_113 = None
        view_184: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_67, [1, 512, 2048]);  addmm_67 = None
        view_185: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_184, [1, 512, -1, 64]);  view_184 = None
        permute_114: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_94: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_114, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_117: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_94, unsqueeze_36);  unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_99: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_94, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_23: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_99);  slice_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_98: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_94, 3, 0, 16);  slice_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_46: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_23, slice_98], -1);  neg_23 = slice_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_118: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_46, unsqueeze_37);  cat_46 = unsqueeze_37 = None
        add_94: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_95: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_114, 3, 32, 9223372036854775807);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_48: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_94, slice_95], -1);  add_94 = slice_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_186: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_225, [512, 2048])
        permute_115: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_68: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg164_1, view_186, permute_115);  arg164_1 = view_186 = permute_115 = None
        view_187: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_68, [1, 512, 2048]);  addmm_68 = None
        view_188: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_187, [1, 512, -1, 64]);  view_187 = None
        permute_116: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_24: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_24, full_default_23);  full_default_24 = full_default_23 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_47, cat_48, permute_116, where_11, False, scale = 0.125);  cat_47 = cat_48 = permute_116 = where_11 = None
        getitem_123: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_123, [0, 2, 1, 3]);  getitem_123 = None
        clone_36: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_189: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_36, [1, 512, -1]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_190: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_189, [512, 2048]);  view_189 = None
        permute_118: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_69: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg166_1, view_190, permute_118);  arg166_1 = view_190 = permute_118 = None
        view_191: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_69, [1, 512, 2048]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_192: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_225, [512, 2048]);  convert_element_type_225 = None
        permute_119: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_70: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg168_1, view_192, permute_119);  arg168_1 = view_192 = permute_119 = None
        view_193: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_70, [1, 512, 8192]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_119: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_193, 0.5)
        pow_12: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_193, 3.0)
        mul_120: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_193, mul_120);  view_193 = mul_120 = None
        mul_121: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_121);  mul_121 = None
        add_96: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_122: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_119, add_96);  mul_119 = add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_194: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_122, [512, 8192]);  mul_122 = None
        permute_120: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_71: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg170_1, view_194, permute_120);  arg170_1 = view_194 = permute_120 = None
        view_195: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_71, [1, 512, 2048]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_97: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_191, view_195);  view_191 = view_195 = None
        add_98: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_97, add_90);  add_97 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_244: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_98, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_244, [2], correction = 0, keepdim = True)
        getitem_132: "f32[1, 512, 1]" = var_mean_12[0]
        getitem_133: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_14: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_244, getitem_133);  convert_element_type_244 = getitem_133 = None
        add_99: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_123: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = rsqrt_12 = None
        mul_124: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_123, arg171_1);  mul_123 = arg171_1 = None
        add_100: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_124, arg172_1);  mul_124 = arg172_1 = None
        convert_element_type_245: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_100, torch.float16);  add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_196: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_245, [512, 2048])
        permute_121: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_72: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg174_1, view_196, permute_121);  arg174_1 = view_196 = permute_121 = None
        view_197: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_72, [1, 512, 2048]);  addmm_72 = None
        view_198: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_197, [1, 512, -1, 64]);  view_197 = None
        permute_122: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_100: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_122, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_38: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_125: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_100, unsqueeze_38)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_105: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_100, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_24: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_105);  slice_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_104: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_100, 3, 0, 16);  slice_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_49: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_24, slice_104], -1);  neg_24 = slice_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_39: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_126: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_49, unsqueeze_39);  cat_49 = None
        add_101: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_101: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_122, 3, 32, 9223372036854775807);  permute_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_51: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_101, slice_101], -1);  add_101 = slice_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_199: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_245, [512, 2048])
        permute_123: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_73: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg176_1, view_199, permute_123);  arg176_1 = view_199 = permute_123 = None
        view_200: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_73, [1, 512, 2048]);  addmm_73 = None
        view_201: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_200, [1, 512, -1, 64]);  view_200 = None
        permute_124: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_102: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_124, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_127: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_102, unsqueeze_38);  unsqueeze_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_107: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_102, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_25: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_107);  slice_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_106: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_102, 3, 0, 16);  slice_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_50: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_25, slice_106], -1);  neg_25 = slice_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_128: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_50, unsqueeze_39);  cat_50 = unsqueeze_39 = None
        add_102: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_103: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_124, 3, 32, 9223372036854775807);  permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_52: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_102, slice_103], -1);  add_102 = slice_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_202: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_245, [512, 2048])
        permute_125: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_74: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg178_1, view_202, permute_125);  arg178_1 = view_202 = permute_125 = None
        view_203: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_74, [1, 512, 2048]);  addmm_74 = None
        view_204: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_203, [1, 512, -1, 64]);  view_203 = None
        permute_126: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_26: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_26, full_default_25);  full_default_26 = full_default_25 = None
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_51, cat_52, permute_126, where_12, False, scale = 0.125);  cat_51 = cat_52 = permute_126 = where_12 = None
        getitem_134: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_12[0];  _scaled_dot_product_cudnn_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_127: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        clone_39: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_205: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_39, [1, 512, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_206: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_205, [512, 2048]);  view_205 = None
        permute_128: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_75: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg180_1, view_206, permute_128);  arg180_1 = view_206 = permute_128 = None
        view_207: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_75, [1, 512, 2048]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_208: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_245, [512, 2048]);  convert_element_type_245 = None
        permute_129: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_76: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg182_1, view_208, permute_129);  arg182_1 = view_208 = permute_129 = None
        view_209: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_76, [1, 512, 8192]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_129: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_209, 0.5)
        pow_13: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_209, 3.0)
        mul_130: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_103: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_209, mul_130);  view_209 = mul_130 = None
        mul_131: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_103, 0.7978845608028654);  add_103 = None
        tanh_12: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_131);  mul_131 = None
        add_104: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_132: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_129, add_104);  mul_129 = add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_210: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_132, [512, 8192]);  mul_132 = None
        permute_130: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_77: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg184_1, view_210, permute_130);  arg184_1 = view_210 = permute_130 = None
        view_211: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_77, [1, 512, 2048]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_105: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_207, view_211);  view_207 = view_211 = None
        add_106: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_105, add_98);  add_105 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_264: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_106, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_264, [2], correction = 0, keepdim = True)
        getitem_143: "f32[1, 512, 1]" = var_mean_13[0]
        getitem_144: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_15: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_264, getitem_144);  convert_element_type_264 = getitem_144 = None
        add_107: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_133: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = rsqrt_13 = None
        mul_134: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_133, arg185_1);  mul_133 = arg185_1 = None
        add_108: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_134, arg186_1);  mul_134 = arg186_1 = None
        convert_element_type_265: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_108, torch.float16);  add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_212: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_265, [512, 2048])
        permute_131: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_78: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg188_1, view_212, permute_131);  arg188_1 = view_212 = permute_131 = None
        view_213: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_78, [1, 512, 2048]);  addmm_78 = None
        view_214: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_213, [1, 512, -1, 64]);  view_213 = None
        permute_132: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_108: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_132, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_40: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_135: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_108, unsqueeze_40)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_113: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_108, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_26: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_113);  slice_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_112: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_108, 3, 0, 16);  slice_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_53: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_26, slice_112], -1);  neg_26 = slice_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_41: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_136: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_53, unsqueeze_41);  cat_53 = None
        add_109: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_109: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_132, 3, 32, 9223372036854775807);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_55: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_109, slice_109], -1);  add_109 = slice_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_215: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_265, [512, 2048])
        permute_133: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_79: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg190_1, view_215, permute_133);  arg190_1 = view_215 = permute_133 = None
        view_216: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_79, [1, 512, 2048]);  addmm_79 = None
        view_217: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_216, [1, 512, -1, 64]);  view_216 = None
        permute_134: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_110: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_134, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_137: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_110, unsqueeze_40);  unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_115: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_110, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_27: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_115);  slice_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_114: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_110, 3, 0, 16);  slice_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_54: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_27, slice_114], -1);  neg_27 = slice_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_138: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_54, unsqueeze_41);  cat_54 = unsqueeze_41 = None
        add_110: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_111: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_134, 3, 32, 9223372036854775807);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_56: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_110, slice_111], -1);  add_110 = slice_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_218: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_265, [512, 2048])
        permute_135: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_80: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg192_1, view_218, permute_135);  arg192_1 = view_218 = permute_135 = None
        view_219: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_80, [1, 512, 2048]);  addmm_80 = None
        view_220: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_219, [1, 512, -1, 64]);  view_219 = None
        permute_136: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_28: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_28, full_default_27);  full_default_28 = full_default_27 = None
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_55, cat_56, permute_136, where_13, False, scale = 0.125);  cat_55 = cat_56 = permute_136 = where_13 = None
        getitem_145: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_13[0];  _scaled_dot_product_cudnn_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_137: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None
        clone_42: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_221: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_42, [1, 512, -1]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_222: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_221, [512, 2048]);  view_221 = None
        permute_138: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_81: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg194_1, view_222, permute_138);  arg194_1 = view_222 = permute_138 = None
        view_223: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_81, [1, 512, 2048]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_224: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_265, [512, 2048]);  convert_element_type_265 = None
        permute_139: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_82: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg196_1, view_224, permute_139);  arg196_1 = view_224 = permute_139 = None
        view_225: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_82, [1, 512, 8192]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_139: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_225, 0.5)
        pow_14: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_225, 3.0)
        mul_140: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_111: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_225, mul_140);  view_225 = mul_140 = None
        mul_141: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_111, 0.7978845608028654);  add_111 = None
        tanh_13: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_141);  mul_141 = None
        add_112: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_142: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_139, add_112);  mul_139 = add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_226: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_142, [512, 8192]);  mul_142 = None
        permute_140: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_83: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg198_1, view_226, permute_140);  arg198_1 = view_226 = permute_140 = None
        view_227: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_83, [1, 512, 2048]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_113: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_223, view_227);  view_223 = view_227 = None
        add_114: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_113, add_106);  add_113 = add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_284: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_284, [2], correction = 0, keepdim = True)
        getitem_154: "f32[1, 512, 1]" = var_mean_14[0]
        getitem_155: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_16: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_284, getitem_155);  convert_element_type_284 = getitem_155 = None
        add_115: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_143: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = rsqrt_14 = None
        mul_144: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_143, arg199_1);  mul_143 = arg199_1 = None
        add_116: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_144, arg200_1);  mul_144 = arg200_1 = None
        convert_element_type_285: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_116, torch.float16);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_228: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_285, [512, 2048])
        permute_141: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_84: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg202_1, view_228, permute_141);  arg202_1 = view_228 = permute_141 = None
        view_229: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_84, [1, 512, 2048]);  addmm_84 = None
        view_230: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_229, [1, 512, -1, 64]);  view_229 = None
        permute_142: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_116: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_142, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_42: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_145: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_116, unsqueeze_42)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_121: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_116, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_28: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_121);  slice_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_120: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_116, 3, 0, 16);  slice_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_57: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_28, slice_120], -1);  neg_28 = slice_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_43: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_146: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_57, unsqueeze_43);  cat_57 = None
        add_117: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_117: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_142, 3, 32, 9223372036854775807);  permute_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_59: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_117, slice_117], -1);  add_117 = slice_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_231: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_285, [512, 2048])
        permute_143: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_85: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg204_1, view_231, permute_143);  arg204_1 = view_231 = permute_143 = None
        view_232: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_85, [1, 512, 2048]);  addmm_85 = None
        view_233: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_232, [1, 512, -1, 64]);  view_232 = None
        permute_144: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_118: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_144, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_147: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_118, unsqueeze_42);  unsqueeze_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_123: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_118, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_29: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_123);  slice_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_122: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_118, 3, 0, 16);  slice_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_58: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_29, slice_122], -1);  neg_29 = slice_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_148: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_58, unsqueeze_43);  cat_58 = unsqueeze_43 = None
        add_118: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_119: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_144, 3, 32, 9223372036854775807);  permute_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_60: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_118, slice_119], -1);  add_118 = slice_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_234: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_285, [512, 2048])
        permute_145: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_86: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg206_1, view_234, permute_145);  arg206_1 = view_234 = permute_145 = None
        view_235: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_86, [1, 512, 2048]);  addmm_86 = None
        view_236: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_235, [1, 512, -1, 64]);  view_235 = None
        permute_146: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_30: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_30, full_default_29);  full_default_30 = full_default_29 = None
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_59, cat_60, permute_146, where_14, False, scale = 0.125);  cat_59 = cat_60 = permute_146 = where_14 = None
        getitem_156: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_14[0];  _scaled_dot_product_cudnn_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_147: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3]);  getitem_156 = None
        clone_45: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_237: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_45, [1, 512, -1]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_238: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_237, [512, 2048]);  view_237 = None
        permute_148: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_87: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg208_1, view_238, permute_148);  arg208_1 = view_238 = permute_148 = None
        view_239: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_87, [1, 512, 2048]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_240: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_285, [512, 2048]);  convert_element_type_285 = None
        permute_149: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_88: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg210_1, view_240, permute_149);  arg210_1 = view_240 = permute_149 = None
        view_241: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_88, [1, 512, 8192]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_149: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
        pow_15: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 3.0)
        mul_150: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_119: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_241, mul_150);  view_241 = mul_150 = None
        mul_151: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_119, 0.7978845608028654);  add_119 = None
        tanh_14: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_151);  mul_151 = None
        add_120: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_152: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_149, add_120);  mul_149 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_242: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_152, [512, 8192]);  mul_152 = None
        permute_150: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_89: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg212_1, view_242, permute_150);  arg212_1 = view_242 = permute_150 = None
        view_243: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_89, [1, 512, 2048]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_121: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_239, view_243);  view_239 = view_243 = None
        add_122: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_121, add_114);  add_121 = add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_304: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_122, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_304, [2], correction = 0, keepdim = True)
        getitem_165: "f32[1, 512, 1]" = var_mean_15[0]
        getitem_166: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_17: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_304, getitem_166);  convert_element_type_304 = getitem_166 = None
        add_123: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_165, 1e-05);  getitem_165 = None
        rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        mul_153: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_15);  sub_17 = rsqrt_15 = None
        mul_154: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_153, arg213_1);  mul_153 = arg213_1 = None
        add_124: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_154, arg214_1);  mul_154 = arg214_1 = None
        convert_element_type_305: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_124, torch.float16);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_244: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_305, [512, 2048])
        permute_151: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_90: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg216_1, view_244, permute_151);  arg216_1 = view_244 = permute_151 = None
        view_245: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_90, [1, 512, 2048]);  addmm_90 = None
        view_246: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_245, [1, 512, -1, 64]);  view_245 = None
        permute_152: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_124: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_152, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_44: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_155: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_124, unsqueeze_44)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_129: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_124, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_30: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_129);  slice_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_128: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_124, 3, 0, 16);  slice_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_61: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_30, slice_128], -1);  neg_30 = slice_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_45: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_156: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_61, unsqueeze_45);  cat_61 = None
        add_125: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_125: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_152, 3, 32, 9223372036854775807);  permute_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_63: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_125, slice_125], -1);  add_125 = slice_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_247: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_305, [512, 2048])
        permute_153: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_91: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg218_1, view_247, permute_153);  arg218_1 = view_247 = permute_153 = None
        view_248: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_91, [1, 512, 2048]);  addmm_91 = None
        view_249: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_248, [1, 512, -1, 64]);  view_248 = None
        permute_154: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_126: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_154, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_157: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_126, unsqueeze_44);  unsqueeze_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_131: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_126, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_31: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_131);  slice_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_130: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_126, 3, 0, 16);  slice_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_62: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_31, slice_130], -1);  neg_31 = slice_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_158: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_62, unsqueeze_45);  cat_62 = unsqueeze_45 = None
        add_126: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_127: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_154, 3, 32, 9223372036854775807);  permute_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_64: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_126, slice_127], -1);  add_126 = slice_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_250: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_305, [512, 2048])
        permute_155: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_92: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg220_1, view_250, permute_155);  arg220_1 = view_250 = permute_155 = None
        view_251: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_92, [1, 512, 2048]);  addmm_92 = None
        view_252: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_251, [1, 512, -1, 64]);  view_251 = None
        permute_156: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_32: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_32, full_default_31);  full_default_32 = full_default_31 = None
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_63, cat_64, permute_156, where_15, False, scale = 0.125);  cat_63 = cat_64 = permute_156 = where_15 = None
        getitem_167: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_15[0];  _scaled_dot_product_cudnn_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_157: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None
        clone_48: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_253: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_48, [1, 512, -1]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_254: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_253, [512, 2048]);  view_253 = None
        permute_158: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        addmm_93: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg222_1, view_254, permute_158);  arg222_1 = view_254 = permute_158 = None
        view_255: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_93, [1, 512, 2048]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_256: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_305, [512, 2048]);  convert_element_type_305 = None
        permute_159: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_94: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg224_1, view_256, permute_159);  arg224_1 = view_256 = permute_159 = None
        view_257: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_94, [1, 512, 8192]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_159: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_257, 0.5)
        pow_16: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_257, 3.0)
        mul_160: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_127: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_257, mul_160);  view_257 = mul_160 = None
        mul_161: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_127, 0.7978845608028654);  add_127 = None
        tanh_15: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_161);  mul_161 = None
        add_128: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_162: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_159, add_128);  mul_159 = add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_258: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_162, [512, 8192]);  mul_162 = None
        permute_160: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_95: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg226_1, view_258, permute_160);  arg226_1 = view_258 = permute_160 = None
        view_259: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_95, [1, 512, 2048]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_129: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_255, view_259);  view_255 = view_259 = None
        add_130: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_129, add_122);  add_129 = add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_324: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_130, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_324, [2], correction = 0, keepdim = True)
        getitem_176: "f32[1, 512, 1]" = var_mean_16[0]
        getitem_177: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_18: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_324, getitem_177);  convert_element_type_324 = getitem_177 = None
        add_131: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        mul_163: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_16);  sub_18 = rsqrt_16 = None
        mul_164: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_163, arg227_1);  mul_163 = arg227_1 = None
        add_132: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_164, arg228_1);  mul_164 = arg228_1 = None
        convert_element_type_325: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_132, torch.float16);  add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_260: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_325, [512, 2048])
        permute_161: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_96: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg230_1, view_260, permute_161);  arg230_1 = view_260 = permute_161 = None
        view_261: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_96, [1, 512, 2048]);  addmm_96 = None
        view_262: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_261, [1, 512, -1, 64]);  view_261 = None
        permute_162: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_132: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_162, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_46: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_165: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_132, unsqueeze_46)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_137: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_132, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_32: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_137);  slice_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_136: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_132, 3, 0, 16);  slice_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_65: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_32, slice_136], -1);  neg_32 = slice_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_47: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_166: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_65, unsqueeze_47);  cat_65 = None
        add_133: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_133: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_162, 3, 32, 9223372036854775807);  permute_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_67: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_133, slice_133], -1);  add_133 = slice_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_263: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_325, [512, 2048])
        permute_163: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_97: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg232_1, view_263, permute_163);  arg232_1 = view_263 = permute_163 = None
        view_264: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_97, [1, 512, 2048]);  addmm_97 = None
        view_265: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_264, [1, 512, -1, 64]);  view_264 = None
        permute_164: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_134: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_164, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_167: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_134, unsqueeze_46);  unsqueeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_139: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_134, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_33: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_139);  slice_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_138: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_134, 3, 0, 16);  slice_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_66: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_33, slice_138], -1);  neg_33 = slice_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_168: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_66, unsqueeze_47);  cat_66 = unsqueeze_47 = None
        add_134: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_135: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_164, 3, 32, 9223372036854775807);  permute_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_68: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_134, slice_135], -1);  add_134 = slice_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_266: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_325, [512, 2048])
        permute_165: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_98: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg234_1, view_266, permute_165);  arg234_1 = view_266 = permute_165 = None
        view_267: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_98, [1, 512, 2048]);  addmm_98 = None
        view_268: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_267, [1, 512, -1, 64]);  view_267 = None
        permute_166: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_34: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_34, full_default_33);  full_default_34 = full_default_33 = None
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_67, cat_68, permute_166, where_16, False, scale = 0.125);  cat_67 = cat_68 = permute_166 = where_16 = None
        getitem_178: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_16[0];  _scaled_dot_product_cudnn_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_167: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_178, [0, 2, 1, 3]);  getitem_178 = None
        clone_51: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_269: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_51, [1, 512, -1]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_270: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_269, [512, 2048]);  view_269 = None
        permute_168: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_99: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg236_1, view_270, permute_168);  arg236_1 = view_270 = permute_168 = None
        view_271: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_99, [1, 512, 2048]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_272: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_325, [512, 2048]);  convert_element_type_325 = None
        permute_169: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_100: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg238_1, view_272, permute_169);  arg238_1 = view_272 = permute_169 = None
        view_273: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_100, [1, 512, 8192]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_169: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_273, 0.5)
        pow_17: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_273, 3.0)
        mul_170: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_135: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_273, mul_170);  view_273 = mul_170 = None
        mul_171: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_135, 0.7978845608028654);  add_135 = None
        tanh_16: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_171);  mul_171 = None
        add_136: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
        mul_172: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_169, add_136);  mul_169 = add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_274: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_172, [512, 8192]);  mul_172 = None
        permute_170: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_101: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg240_1, view_274, permute_170);  arg240_1 = view_274 = permute_170 = None
        view_275: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_101, [1, 512, 2048]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_137: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_271, view_275);  view_271 = view_275 = None
        add_138: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_137, add_130);  add_137 = add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_344: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_138, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_344, [2], correction = 0, keepdim = True)
        getitem_187: "f32[1, 512, 1]" = var_mean_17[0]
        getitem_188: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_19: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_344, getitem_188);  convert_element_type_344 = getitem_188 = None
        add_139: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_187, 1e-05);  getitem_187 = None
        rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_173: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_17);  sub_19 = rsqrt_17 = None
        mul_174: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_173, arg241_1);  mul_173 = arg241_1 = None
        add_140: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_174, arg242_1);  mul_174 = arg242_1 = None
        convert_element_type_345: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_140, torch.float16);  add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_276: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_345, [512, 2048])
        permute_171: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_102: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg244_1, view_276, permute_171);  arg244_1 = view_276 = permute_171 = None
        view_277: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_102, [1, 512, 2048]);  addmm_102 = None
        view_278: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_277, [1, 512, -1, 64]);  view_277 = None
        permute_172: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_140: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_172, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_48: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_175: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_140, unsqueeze_48)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_145: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_140, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_34: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_145);  slice_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_144: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_140, 3, 0, 16);  slice_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_69: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_34, slice_144], -1);  neg_34 = slice_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_49: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_176: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_69, unsqueeze_49);  cat_69 = None
        add_141: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_141: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_172, 3, 32, 9223372036854775807);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_71: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_141, slice_141], -1);  add_141 = slice_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_279: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_345, [512, 2048])
        permute_173: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_103: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg246_1, view_279, permute_173);  arg246_1 = view_279 = permute_173 = None
        view_280: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_103, [1, 512, 2048]);  addmm_103 = None
        view_281: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_280, [1, 512, -1, 64]);  view_280 = None
        permute_174: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_142: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_174, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_177: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_142, unsqueeze_48);  unsqueeze_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_147: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_142, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_35: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_147);  slice_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_146: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_142, 3, 0, 16);  slice_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_70: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_35, slice_146], -1);  neg_35 = slice_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_178: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_70, unsqueeze_49);  cat_70 = unsqueeze_49 = None
        add_142: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_143: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_174, 3, 32, 9223372036854775807);  permute_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_72: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_142, slice_143], -1);  add_142 = slice_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_282: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_345, [512, 2048])
        permute_175: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_104: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg248_1, view_282, permute_175);  arg248_1 = view_282 = permute_175 = None
        view_283: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_104, [1, 512, 2048]);  addmm_104 = None
        view_284: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_283, [1, 512, -1, 64]);  view_283 = None
        permute_176: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_36: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_36, full_default_35);  full_default_36 = full_default_35 = None
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_71, cat_72, permute_176, where_17, False, scale = 0.125);  cat_71 = cat_72 = permute_176 = where_17 = None
        getitem_189: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_17[0];  _scaled_dot_product_cudnn_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_177: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None
        clone_54: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_285: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_54, [1, 512, -1]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_286: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_285, [512, 2048]);  view_285 = None
        permute_178: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_105: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg250_1, view_286, permute_178);  arg250_1 = view_286 = permute_178 = None
        view_287: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_105, [1, 512, 2048]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_288: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_345, [512, 2048]);  convert_element_type_345 = None
        permute_179: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_106: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg252_1, view_288, permute_179);  arg252_1 = view_288 = permute_179 = None
        view_289: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_106, [1, 512, 8192]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_179: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_289, 0.5)
        pow_18: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_289, 3.0)
        mul_180: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_143: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_289, mul_180);  view_289 = mul_180 = None
        mul_181: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_143, 0.7978845608028654);  add_143 = None
        tanh_17: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_181);  mul_181 = None
        add_144: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
        mul_182: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_179, add_144);  mul_179 = add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_290: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_182, [512, 8192]);  mul_182 = None
        permute_180: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_107: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg254_1, view_290, permute_180);  arg254_1 = view_290 = permute_180 = None
        view_291: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_107, [1, 512, 2048]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_145: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_287, view_291);  view_287 = view_291 = None
        add_146: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_145, add_138);  add_145 = add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_364: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_146, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_364, [2], correction = 0, keepdim = True)
        getitem_198: "f32[1, 512, 1]" = var_mean_18[0]
        getitem_199: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_20: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_364, getitem_199);  convert_element_type_364 = getitem_199 = None
        add_147: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        mul_183: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_18);  sub_20 = rsqrt_18 = None
        mul_184: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_183, arg255_1);  mul_183 = arg255_1 = None
        add_148: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_184, arg256_1);  mul_184 = arg256_1 = None
        convert_element_type_365: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_148, torch.float16);  add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_292: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_365, [512, 2048])
        permute_181: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_108: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg258_1, view_292, permute_181);  arg258_1 = view_292 = permute_181 = None
        view_293: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_108, [1, 512, 2048]);  addmm_108 = None
        view_294: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_293, [1, 512, -1, 64]);  view_293 = None
        permute_182: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_148: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_182, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_50: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_185: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_148, unsqueeze_50)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_153: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_148, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_36: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_153);  slice_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_152: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_148, 3, 0, 16);  slice_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_73: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_36, slice_152], -1);  neg_36 = slice_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_51: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_186: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_73, unsqueeze_51);  cat_73 = None
        add_149: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_149: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_182, 3, 32, 9223372036854775807);  permute_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_75: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_149, slice_149], -1);  add_149 = slice_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_295: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_365, [512, 2048])
        permute_183: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_109: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg260_1, view_295, permute_183);  arg260_1 = view_295 = permute_183 = None
        view_296: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_109, [1, 512, 2048]);  addmm_109 = None
        view_297: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_296, [1, 512, -1, 64]);  view_296 = None
        permute_184: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_150: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_184, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_187: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_150, unsqueeze_50);  unsqueeze_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_155: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_150, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_37: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_155);  slice_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_154: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_150, 3, 0, 16);  slice_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_74: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_37, slice_154], -1);  neg_37 = slice_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_188: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_74, unsqueeze_51);  cat_74 = unsqueeze_51 = None
        add_150: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_151: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_184, 3, 32, 9223372036854775807);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_76: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_150, slice_151], -1);  add_150 = slice_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_298: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_365, [512, 2048])
        permute_185: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_110: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg262_1, view_298, permute_185);  arg262_1 = view_298 = permute_185 = None
        view_299: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_110, [1, 512, 2048]);  addmm_110 = None
        view_300: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_299, [1, 512, -1, 64]);  view_299 = None
        permute_186: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_38: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_38, full_default_37);  full_default_38 = full_default_37 = None
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_75, cat_76, permute_186, where_18, False, scale = 0.125);  cat_75 = cat_76 = permute_186 = where_18 = None
        getitem_200: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_18[0];  _scaled_dot_product_cudnn_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_187: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_200, [0, 2, 1, 3]);  getitem_200 = None
        clone_57: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_301: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_57, [1, 512, -1]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_302: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_301, [512, 2048]);  view_301 = None
        permute_188: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_111: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg264_1, view_302, permute_188);  arg264_1 = view_302 = permute_188 = None
        view_303: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_111, [1, 512, 2048]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_304: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_365, [512, 2048]);  convert_element_type_365 = None
        permute_189: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_112: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg266_1, view_304, permute_189);  arg266_1 = view_304 = permute_189 = None
        view_305: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_112, [1, 512, 8192]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_189: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        pow_19: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_305, 3.0)
        mul_190: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_151: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_305, mul_190);  view_305 = mul_190 = None
        mul_191: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_151, 0.7978845608028654);  add_151 = None
        tanh_18: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_191);  mul_191 = None
        add_152: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
        mul_192: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_189, add_152);  mul_189 = add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_306: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_192, [512, 8192]);  mul_192 = None
        permute_190: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_113: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg268_1, view_306, permute_190);  arg268_1 = view_306 = permute_190 = None
        view_307: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_113, [1, 512, 2048]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_153: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_303, view_307);  view_303 = view_307 = None
        add_154: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_153, add_146);  add_153 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_384: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_154, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_384, [2], correction = 0, keepdim = True)
        getitem_209: "f32[1, 512, 1]" = var_mean_19[0]
        getitem_210: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_21: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_384, getitem_210);  convert_element_type_384 = getitem_210 = None
        add_155: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_209, 1e-05);  getitem_209 = None
        rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_193: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_19);  sub_21 = rsqrt_19 = None
        mul_194: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_193, arg269_1);  mul_193 = arg269_1 = None
        add_156: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_194, arg270_1);  mul_194 = arg270_1 = None
        convert_element_type_385: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_156, torch.float16);  add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_308: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_385, [512, 2048])
        permute_191: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_114: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg272_1, view_308, permute_191);  arg272_1 = view_308 = permute_191 = None
        view_309: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_114, [1, 512, 2048]);  addmm_114 = None
        view_310: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_309, [1, 512, -1, 64]);  view_309 = None
        permute_192: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_156: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_192, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_52: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_195: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_156, unsqueeze_52)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_161: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_156, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_38: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_161);  slice_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_160: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_156, 3, 0, 16);  slice_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_77: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_38, slice_160], -1);  neg_38 = slice_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_53: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_196: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_77, unsqueeze_53);  cat_77 = None
        add_157: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_157: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_192, 3, 32, 9223372036854775807);  permute_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_79: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_157, slice_157], -1);  add_157 = slice_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_311: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_385, [512, 2048])
        permute_193: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_115: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg274_1, view_311, permute_193);  arg274_1 = view_311 = permute_193 = None
        view_312: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_115, [1, 512, 2048]);  addmm_115 = None
        view_313: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_312, [1, 512, -1, 64]);  view_312 = None
        permute_194: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_158: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_194, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_197: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_158, unsqueeze_52);  unsqueeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_163: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_158, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_39: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_163);  slice_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_162: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_158, 3, 0, 16);  slice_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_78: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_39, slice_162], -1);  neg_39 = slice_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_198: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_78, unsqueeze_53);  cat_78 = unsqueeze_53 = None
        add_158: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_159: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_194, 3, 32, 9223372036854775807);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_80: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_158, slice_159], -1);  add_158 = slice_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_314: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_385, [512, 2048])
        permute_195: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_116: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg276_1, view_314, permute_195);  arg276_1 = view_314 = permute_195 = None
        view_315: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_116, [1, 512, 2048]);  addmm_116 = None
        view_316: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_315, [1, 512, -1, 64]);  view_315 = None
        permute_196: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_40: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_40, full_default_39);  full_default_40 = full_default_39 = None
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_79, cat_80, permute_196, where_19, False, scale = 0.125);  cat_79 = cat_80 = permute_196 = where_19 = None
        getitem_211: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_19[0];  _scaled_dot_product_cudnn_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_197: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_211, [0, 2, 1, 3]);  getitem_211 = None
        clone_60: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_317: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_60, [1, 512, -1]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_318: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_317, [512, 2048]);  view_317 = None
        permute_198: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_117: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg278_1, view_318, permute_198);  arg278_1 = view_318 = permute_198 = None
        view_319: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_117, [1, 512, 2048]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_320: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_385, [512, 2048]);  convert_element_type_385 = None
        permute_199: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_118: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg280_1, view_320, permute_199);  arg280_1 = view_320 = permute_199 = None
        view_321: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_118, [1, 512, 8192]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_199: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_321, 0.5)
        pow_20: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_321, 3.0)
        mul_200: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_159: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_321, mul_200);  view_321 = mul_200 = None
        mul_201: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_159, 0.7978845608028654);  add_159 = None
        tanh_19: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_201);  mul_201 = None
        add_160: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
        mul_202: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_199, add_160);  mul_199 = add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_322: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_202, [512, 8192]);  mul_202 = None
        permute_200: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_119: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg282_1, view_322, permute_200);  arg282_1 = view_322 = permute_200 = None
        view_323: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_119, [1, 512, 2048]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_161: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_319, view_323);  view_319 = view_323 = None
        add_162: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_161, add_154);  add_161 = add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_404: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_162, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_404, [2], correction = 0, keepdim = True)
        getitem_220: "f32[1, 512, 1]" = var_mean_20[0]
        getitem_221: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_22: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_404, getitem_221);  convert_element_type_404 = getitem_221 = None
        add_163: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-05);  getitem_220 = None
        rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        mul_203: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_20);  sub_22 = rsqrt_20 = None
        mul_204: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_203, arg283_1);  mul_203 = arg283_1 = None
        add_164: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_204, arg284_1);  mul_204 = arg284_1 = None
        convert_element_type_405: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_164, torch.float16);  add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_324: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_405, [512, 2048])
        permute_201: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_120: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg286_1, view_324, permute_201);  arg286_1 = view_324 = permute_201 = None
        view_325: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_120, [1, 512, 2048]);  addmm_120 = None
        view_326: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_325, [1, 512, -1, 64]);  view_325 = None
        permute_202: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_164: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_202, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_54: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_205: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_164, unsqueeze_54)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_169: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_164, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_40: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_169);  slice_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_168: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_164, 3, 0, 16);  slice_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_81: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_40, slice_168], -1);  neg_40 = slice_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_55: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_206: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_81, unsqueeze_55);  cat_81 = None
        add_165: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_165: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_202, 3, 32, 9223372036854775807);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_83: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_165, slice_165], -1);  add_165 = slice_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_327: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_405, [512, 2048])
        permute_203: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_121: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg288_1, view_327, permute_203);  arg288_1 = view_327 = permute_203 = None
        view_328: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_121, [1, 512, 2048]);  addmm_121 = None
        view_329: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_328, [1, 512, -1, 64]);  view_328 = None
        permute_204: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_166: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_204, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_207: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_166, unsqueeze_54);  unsqueeze_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_171: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_166, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_41: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_171);  slice_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_170: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_166, 3, 0, 16);  slice_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_82: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_41, slice_170], -1);  neg_41 = slice_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_208: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_82, unsqueeze_55);  cat_82 = unsqueeze_55 = None
        add_166: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_167: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_204, 3, 32, 9223372036854775807);  permute_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_84: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_166, slice_167], -1);  add_166 = slice_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_330: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_405, [512, 2048])
        permute_205: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_122: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg290_1, view_330, permute_205);  arg290_1 = view_330 = permute_205 = None
        view_331: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_122, [1, 512, 2048]);  addmm_122 = None
        view_332: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_331, [1, 512, -1, 64]);  view_331 = None
        permute_206: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_42: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_42, full_default_41);  full_default_42 = full_default_41 = None
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_83, cat_84, permute_206, where_20, False, scale = 0.125);  cat_83 = cat_84 = permute_206 = where_20 = None
        getitem_222: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_20[0];  _scaled_dot_product_cudnn_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_207: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_222, [0, 2, 1, 3]);  getitem_222 = None
        clone_63: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_333: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_63, [1, 512, -1]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_334: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_333, [512, 2048]);  view_333 = None
        permute_208: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_123: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg292_1, view_334, permute_208);  arg292_1 = view_334 = permute_208 = None
        view_335: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_123, [1, 512, 2048]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_336: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_405, [512, 2048]);  convert_element_type_405 = None
        permute_209: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_124: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg294_1, view_336, permute_209);  arg294_1 = view_336 = permute_209 = None
        view_337: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_124, [1, 512, 8192]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_209: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_337, 0.5)
        pow_21: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_337, 3.0)
        mul_210: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_167: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_337, mul_210);  view_337 = mul_210 = None
        mul_211: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_167, 0.7978845608028654);  add_167 = None
        tanh_20: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_211);  mul_211 = None
        add_168: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
        mul_212: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_209, add_168);  mul_209 = add_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_338: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_212, [512, 8192]);  mul_212 = None
        permute_210: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_125: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg296_1, view_338, permute_210);  arg296_1 = view_338 = permute_210 = None
        view_339: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_125, [1, 512, 2048]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_169: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_335, view_339);  view_335 = view_339 = None
        add_170: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_169, add_162);  add_169 = add_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_424: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_170, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_424, [2], correction = 0, keepdim = True)
        getitem_231: "f32[1, 512, 1]" = var_mean_21[0]
        getitem_232: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_23: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_424, getitem_232);  convert_element_type_424 = getitem_232 = None
        add_171: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_231, 1e-05);  getitem_231 = None
        rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        mul_213: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_21);  sub_23 = rsqrt_21 = None
        mul_214: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_213, arg297_1);  mul_213 = arg297_1 = None
        add_172: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_214, arg298_1);  mul_214 = arg298_1 = None
        convert_element_type_425: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_172, torch.float16);  add_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_340: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_425, [512, 2048])
        permute_211: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_126: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg300_1, view_340, permute_211);  arg300_1 = view_340 = permute_211 = None
        view_341: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_126, [1, 512, 2048]);  addmm_126 = None
        view_342: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_341, [1, 512, -1, 64]);  view_341 = None
        permute_212: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_172: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_212, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_56: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_215: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_172, unsqueeze_56)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_177: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_172, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_42: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_177);  slice_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_176: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_172, 3, 0, 16);  slice_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_85: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_42, slice_176], -1);  neg_42 = slice_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_57: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_216: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_85, unsqueeze_57);  cat_85 = None
        add_173: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_215, mul_216);  mul_215 = mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_173: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_212, 3, 32, 9223372036854775807);  permute_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_87: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_173, slice_173], -1);  add_173 = slice_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_343: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_425, [512, 2048])
        permute_213: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        addmm_127: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg302_1, view_343, permute_213);  arg302_1 = view_343 = permute_213 = None
        view_344: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_127, [1, 512, 2048]);  addmm_127 = None
        view_345: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_344, [1, 512, -1, 64]);  view_344 = None
        permute_214: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_174: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_214, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_217: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_174, unsqueeze_56);  unsqueeze_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_179: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_174, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_43: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_179);  slice_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_178: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_174, 3, 0, 16);  slice_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_86: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_43, slice_178], -1);  neg_43 = slice_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_218: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_86, unsqueeze_57);  cat_86 = unsqueeze_57 = None
        add_174: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_175: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_214, 3, 32, 9223372036854775807);  permute_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_88: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_174, slice_175], -1);  add_174 = slice_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_346: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_425, [512, 2048])
        permute_215: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_128: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg304_1, view_346, permute_215);  arg304_1 = view_346 = permute_215 = None
        view_347: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_128, [1, 512, 2048]);  addmm_128 = None
        view_348: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_347, [1, 512, -1, 64]);  view_347 = None
        permute_216: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_348, [0, 2, 1, 3]);  view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_44: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_44, full_default_43);  full_default_44 = full_default_43 = None
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_87, cat_88, permute_216, where_21, False, scale = 0.125);  cat_87 = cat_88 = permute_216 = where_21 = None
        getitem_233: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_21[0];  _scaled_dot_product_cudnn_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_217: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_233, [0, 2, 1, 3]);  getitem_233 = None
        clone_66: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_349: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_66, [1, 512, -1]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_350: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_349, [512, 2048]);  view_349 = None
        permute_218: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_129: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg306_1, view_350, permute_218);  arg306_1 = view_350 = permute_218 = None
        view_351: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_129, [1, 512, 2048]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_352: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_425, [512, 2048]);  convert_element_type_425 = None
        permute_219: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_130: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg308_1, view_352, permute_219);  arg308_1 = view_352 = permute_219 = None
        view_353: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_130, [1, 512, 8192]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_219: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_353, 0.5)
        pow_22: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_353, 3.0)
        mul_220: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_175: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_353, mul_220);  view_353 = mul_220 = None
        mul_221: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_175, 0.7978845608028654);  add_175 = None
        tanh_21: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_221);  mul_221 = None
        add_176: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
        mul_222: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_219, add_176);  mul_219 = add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_354: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_222, [512, 8192]);  mul_222 = None
        permute_220: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_131: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg310_1, view_354, permute_220);  arg310_1 = view_354 = permute_220 = None
        view_355: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_131, [1, 512, 2048]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_177: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_351, view_355);  view_351 = view_355 = None
        add_178: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_177, add_170);  add_177 = add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_444: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_178, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_444, [2], correction = 0, keepdim = True)
        getitem_242: "f32[1, 512, 1]" = var_mean_22[0]
        getitem_243: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_24: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_444, getitem_243);  convert_element_type_444 = getitem_243 = None
        add_179: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-05);  getitem_242 = None
        rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
        mul_223: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_22);  sub_24 = rsqrt_22 = None
        mul_224: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_223, arg311_1);  mul_223 = arg311_1 = None
        add_180: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_224, arg312_1);  mul_224 = arg312_1 = None
        convert_element_type_445: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_180, torch.float16);  add_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_356: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_445, [512, 2048])
        permute_221: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_132: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg314_1, view_356, permute_221);  arg314_1 = view_356 = permute_221 = None
        view_357: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_132, [1, 512, 2048]);  addmm_132 = None
        view_358: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_357, [1, 512, -1, 64]);  view_357 = None
        permute_222: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_180: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_222, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_58: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_225: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_180, unsqueeze_58)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_185: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_180, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_44: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_185);  slice_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_184: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_180, 3, 0, 16);  slice_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_89: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_44, slice_184], -1);  neg_44 = slice_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_59: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_226: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_89, unsqueeze_59);  cat_89 = None
        add_181: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_181: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_222, 3, 32, 9223372036854775807);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_91: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_181, slice_181], -1);  add_181 = slice_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_359: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_445, [512, 2048])
        permute_223: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_133: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg316_1, view_359, permute_223);  arg316_1 = view_359 = permute_223 = None
        view_360: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_133, [1, 512, 2048]);  addmm_133 = None
        view_361: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_360, [1, 512, -1, 64]);  view_360 = None
        permute_224: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_182: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_224, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_227: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_182, unsqueeze_58);  unsqueeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_187: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_182, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_45: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_187);  slice_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_186: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_182, 3, 0, 16);  slice_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_90: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_45, slice_186], -1);  neg_45 = slice_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_228: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_90, unsqueeze_59);  cat_90 = unsqueeze_59 = None
        add_182: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_183: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_224, 3, 32, 9223372036854775807);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_92: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_182, slice_183], -1);  add_182 = slice_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_362: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_445, [512, 2048])
        permute_225: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        addmm_134: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg318_1, view_362, permute_225);  arg318_1 = view_362 = permute_225 = None
        view_363: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_134, [1, 512, 2048]);  addmm_134 = None
        view_364: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_363, [1, 512, -1, 64]);  view_363 = None
        permute_226: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_46: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_46, full_default_45);  full_default_46 = full_default_45 = None
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_91, cat_92, permute_226, where_22, False, scale = 0.125);  cat_91 = cat_92 = permute_226 = where_22 = None
        getitem_244: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_22[0];  _scaled_dot_product_cudnn_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_227: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_244, [0, 2, 1, 3]);  getitem_244 = None
        clone_69: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_365: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_69, [1, 512, -1]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_366: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_365, [512, 2048]);  view_365 = None
        permute_228: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_135: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg320_1, view_366, permute_228);  arg320_1 = view_366 = permute_228 = None
        view_367: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_135, [1, 512, 2048]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_368: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_445, [512, 2048]);  convert_element_type_445 = None
        permute_229: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_136: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg322_1, view_368, permute_229);  arg322_1 = view_368 = permute_229 = None
        view_369: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_136, [1, 512, 8192]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_229: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_369, 0.5)
        pow_23: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_369, 3.0)
        mul_230: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_183: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_369, mul_230);  view_369 = mul_230 = None
        mul_231: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_183, 0.7978845608028654);  add_183 = None
        tanh_22: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_231);  mul_231 = None
        add_184: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
        mul_232: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_229, add_184);  mul_229 = add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_370: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_232, [512, 8192]);  mul_232 = None
        permute_230: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        addmm_137: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg324_1, view_370, permute_230);  arg324_1 = view_370 = permute_230 = None
        view_371: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_137, [1, 512, 2048]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_185: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_367, view_371);  view_367 = view_371 = None
        add_186: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_185, add_178);  add_185 = add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        convert_element_type_464: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_186, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_464, [2], correction = 0, keepdim = True)
        getitem_253: "f32[1, 512, 1]" = var_mean_23[0]
        getitem_254: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_25: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_464, getitem_254);  convert_element_type_464 = getitem_254 = None
        add_187: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_253, 1e-05);  getitem_253 = None
        rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
        mul_233: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_23);  sub_25 = rsqrt_23 = None
        mul_234: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_233, arg325_1);  mul_233 = arg325_1 = None
        add_188: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_234, arg326_1);  mul_234 = arg326_1 = None
        convert_element_type_465: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_188, torch.float16);  add_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_372: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_465, [512, 2048])
        permute_231: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_138: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg328_1, view_372, permute_231);  arg328_1 = view_372 = permute_231 = None
        view_373: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_138, [1, 512, 2048]);  addmm_138 = None
        view_374: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_373, [1, 512, -1, 64]);  view_373 = None
        permute_232: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_188: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_232, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_60: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_235: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_188, unsqueeze_60)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_193: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_188, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_46: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_193);  slice_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_192: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_188, 3, 0, 16);  slice_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_93: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_46, slice_192], -1);  neg_46 = slice_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_61: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_236: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_93, unsqueeze_61);  cat_93 = None
        add_189: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_189: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_232, 3, 32, 9223372036854775807);  permute_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_95: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_189, slice_189], -1);  add_189 = slice_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_375: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_465, [512, 2048])
        permute_233: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        addmm_139: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg330_1, view_375, permute_233);  arg330_1 = view_375 = permute_233 = None
        view_376: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_139, [1, 512, 2048]);  addmm_139 = None
        view_377: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_376, [1, 512, -1, 64]);  view_376 = None
        permute_234: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_377, [0, 2, 1, 3]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_190: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_234, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_237: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_190, unsqueeze_60);  unsqueeze_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_195: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_190, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_47: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_195);  slice_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_194: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_190, 3, 0, 16);  slice_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_94: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_47, slice_194], -1);  neg_47 = slice_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_238: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_94, unsqueeze_61);  cat_94 = unsqueeze_61 = None
        add_190: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_191: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_234, 3, 32, 9223372036854775807);  permute_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_96: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_190, slice_191], -1);  add_190 = slice_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_378: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_465, [512, 2048])
        permute_235: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_140: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg332_1, view_378, permute_235);  arg332_1 = view_378 = permute_235 = None
        view_379: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_140, [1, 512, 2048]);  addmm_140 = None
        view_380: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(view_379, [1, 512, -1, 64]);  view_379 = None
        permute_236: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_48: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default_48, full_default_47);  expand = full_default_48 = full_default_47 = None
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(cat_95, cat_96, permute_236, where_23, False, scale = 0.125);  cat_95 = cat_96 = permute_236 = where_23 = None
        getitem_255: "f16[1, 32, 512, 64]" = _scaled_dot_product_cudnn_attention_23[0];  _scaled_dot_product_cudnn_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_237: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_255, [0, 2, 1, 3]);  getitem_255 = None
        clone_72: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_381: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_72, [1, 512, -1]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        view_382: "f16[512, 2048]" = torch.ops.aten.reshape.default(view_381, [512, 2048]);  view_381 = None
        permute_238: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_141: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg334_1, view_382, permute_238);  arg334_1 = view_382 = permute_238 = None
        view_383: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_141, [1, 512, 2048]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        view_384: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_465, [512, 2048]);  convert_element_type_465 = None
        permute_239: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_142: "f16[512, 8192]" = torch.ops.aten.addmm.default(arg336_1, view_384, permute_239);  arg336_1 = view_384 = permute_239 = None
        view_385: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_142, [1, 512, 8192]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_239: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(view_385, 0.5)
        pow_24: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_385, 3.0)
        mul_240: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_191: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(view_385, mul_240);  view_385 = mul_240 = None
        mul_241: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_191, 0.7978845608028654);  add_191 = None
        tanh_23: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_241);  mul_241 = None
        add_192: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
        mul_242: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_239, add_192);  mul_239 = add_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        view_386: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_242, [512, 8192]);  mul_242 = None
        permute_240: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_143: "f16[512, 2048]" = torch.ops.aten.addmm.default(arg338_1, view_386, permute_240);  arg338_1 = view_386 = permute_240 = None
        view_387: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_143, [1, 512, 2048]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_193: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_383, view_387);  view_383 = view_387 = None
        add_194: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_193, add_186);  add_193 = add_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:398 in forward, code: hidden_states = self.final_layernorm(hidden_states)
        convert_element_type_484: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_194, torch.float32);  add_194 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_484, [2], correction = 0, keepdim = True)
        getitem_264: "f32[1, 512, 1]" = var_mean_24[0]
        getitem_265: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_26: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_484, getitem_265);  convert_element_type_484 = getitem_265 = None
        add_195: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_264, 1e-05);  getitem_264 = None
        rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        mul_243: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_24);  sub_26 = rsqrt_24 = None
        mul_244: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_243, arg339_1);  mul_243 = arg339_1 = None
        add_196: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_244, arg340_1);  mul_244 = arg340_1 = None
        convert_element_type_485: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_196, torch.float16);  add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:465 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_388: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_485, [512, 2048]);  convert_element_type_485 = None
        permute_241: "f16[2048, 51200]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_144: "f16[512, 51200]" = torch.ops.aten.addmm.default(arg342_1, view_388, permute_241);  arg342_1 = view_388 = permute_241 = None
        view_389: "f16[1, 512, 51200]" = torch.ops.aten.reshape.default(addmm_144, [1, 512, 51200]);  addmm_144 = None
        return (view_389,)
