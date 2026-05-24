import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 80, 3000]", arg1_1: "f16[1024, 80, 3]", arg2_1: "f16[1024]", arg3_1: "f16[1024, 1024, 3]", arg4_1: "f16[1024]", arg5_1: "f16[1500, 1024]", arg6_1: "f16[1024]", arg7_1: "f16[1024]", arg8_1: "f16[1024, 1024]", arg9_1: "f16[1024]", arg10_1: "f16[1024, 1024]", arg11_1: "f16[1024, 1024]", arg12_1: "f16[1024]", arg13_1: "f16[1024, 1024]", arg14_1: "f16[1024]", arg15_1: "f16[1024]", arg16_1: "f16[1024]", arg17_1: "f16[4096, 1024]", arg18_1: "f16[4096]", arg19_1: "f16[1024, 4096]", arg20_1: "f16[1024]", arg21_1: "f16[1024]", arg22_1: "f16[1024]", arg23_1: "f16[1024, 1024]", arg24_1: "f16[1024]", arg25_1: "f16[1024, 1024]", arg26_1: "f16[1024, 1024]", arg27_1: "f16[1024]", arg28_1: "f16[1024, 1024]", arg29_1: "f16[1024]", arg30_1: "f16[1024]", arg31_1: "f16[1024]", arg32_1: "f16[4096, 1024]", arg33_1: "f16[4096]", arg34_1: "f16[1024, 4096]", arg35_1: "f16[1024]", arg36_1: "f16[1024]", arg37_1: "f16[1024]", arg38_1: "f16[1024, 1024]", arg39_1: "f16[1024]", arg40_1: "f16[1024, 1024]", arg41_1: "f16[1024, 1024]", arg42_1: "f16[1024]", arg43_1: "f16[1024, 1024]", arg44_1: "f16[1024]", arg45_1: "f16[1024]", arg46_1: "f16[1024]", arg47_1: "f16[4096, 1024]", arg48_1: "f16[4096]", arg49_1: "f16[1024, 4096]", arg50_1: "f16[1024]", arg51_1: "f16[1024]", arg52_1: "f16[1024]", arg53_1: "f16[1024, 1024]", arg54_1: "f16[1024]", arg55_1: "f16[1024, 1024]", arg56_1: "f16[1024, 1024]", arg57_1: "f16[1024]", arg58_1: "f16[1024, 1024]", arg59_1: "f16[1024]", arg60_1: "f16[1024]", arg61_1: "f16[1024]", arg62_1: "f16[4096, 1024]", arg63_1: "f16[4096]", arg64_1: "f16[1024, 4096]", arg65_1: "f16[1024]", arg66_1: "f16[1024]", arg67_1: "f16[1024]", arg68_1: "f16[1024, 1024]", arg69_1: "f16[1024]", arg70_1: "f16[1024, 1024]", arg71_1: "f16[1024, 1024]", arg72_1: "f16[1024]", arg73_1: "f16[1024, 1024]", arg74_1: "f16[1024]", arg75_1: "f16[1024]", arg76_1: "f16[1024]", arg77_1: "f16[4096, 1024]", arg78_1: "f16[4096]", arg79_1: "f16[1024, 4096]", arg80_1: "f16[1024]", arg81_1: "f16[1024]", arg82_1: "f16[1024]", arg83_1: "f16[1024, 1024]", arg84_1: "f16[1024]", arg85_1: "f16[1024, 1024]", arg86_1: "f16[1024, 1024]", arg87_1: "f16[1024]", arg88_1: "f16[1024, 1024]", arg89_1: "f16[1024]", arg90_1: "f16[1024]", arg91_1: "f16[1024]", arg92_1: "f16[4096, 1024]", arg93_1: "f16[4096]", arg94_1: "f16[1024, 4096]", arg95_1: "f16[1024]", arg96_1: "f16[1024]", arg97_1: "f16[1024]", arg98_1: "f16[1024, 1024]", arg99_1: "f16[1024]", arg100_1: "f16[1024, 1024]", arg101_1: "f16[1024, 1024]", arg102_1: "f16[1024]", arg103_1: "f16[1024, 1024]", arg104_1: "f16[1024]", arg105_1: "f16[1024]", arg106_1: "f16[1024]", arg107_1: "f16[4096, 1024]", arg108_1: "f16[4096]", arg109_1: "f16[1024, 4096]", arg110_1: "f16[1024]", arg111_1: "f16[1024]", arg112_1: "f16[1024]", arg113_1: "f16[1024, 1024]", arg114_1: "f16[1024]", arg115_1: "f16[1024, 1024]", arg116_1: "f16[1024, 1024]", arg117_1: "f16[1024]", arg118_1: "f16[1024, 1024]", arg119_1: "f16[1024]", arg120_1: "f16[1024]", arg121_1: "f16[1024]", arg122_1: "f16[4096, 1024]", arg123_1: "f16[4096]", arg124_1: "f16[1024, 4096]", arg125_1: "f16[1024]", arg126_1: "f16[1024]", arg127_1: "f16[1024]", arg128_1: "f16[1024, 1024]", arg129_1: "f16[1024]", arg130_1: "f16[1024, 1024]", arg131_1: "f16[1024, 1024]", arg132_1: "f16[1024]", arg133_1: "f16[1024, 1024]", arg134_1: "f16[1024]", arg135_1: "f16[1024]", arg136_1: "f16[1024]", arg137_1: "f16[4096, 1024]", arg138_1: "f16[4096]", arg139_1: "f16[1024, 4096]", arg140_1: "f16[1024]", arg141_1: "f16[1024]", arg142_1: "f16[1024]", arg143_1: "f16[1024, 1024]", arg144_1: "f16[1024]", arg145_1: "f16[1024, 1024]", arg146_1: "f16[1024, 1024]", arg147_1: "f16[1024]", arg148_1: "f16[1024, 1024]", arg149_1: "f16[1024]", arg150_1: "f16[1024]", arg151_1: "f16[1024]", arg152_1: "f16[4096, 1024]", arg153_1: "f16[4096]", arg154_1: "f16[1024, 4096]", arg155_1: "f16[1024]", arg156_1: "f16[1024]", arg157_1: "f16[1024]", arg158_1: "f16[1024, 1024]", arg159_1: "f16[1024]", arg160_1: "f16[1024, 1024]", arg161_1: "f16[1024, 1024]", arg162_1: "f16[1024]", arg163_1: "f16[1024, 1024]", arg164_1: "f16[1024]", arg165_1: "f16[1024]", arg166_1: "f16[1024]", arg167_1: "f16[4096, 1024]", arg168_1: "f16[4096]", arg169_1: "f16[1024, 4096]", arg170_1: "f16[1024]", arg171_1: "f16[1024]", arg172_1: "f16[1024]", arg173_1: "f16[1024, 1024]", arg174_1: "f16[1024]", arg175_1: "f16[1024, 1024]", arg176_1: "f16[1024, 1024]", arg177_1: "f16[1024]", arg178_1: "f16[1024, 1024]", arg179_1: "f16[1024]", arg180_1: "f16[1024]", arg181_1: "f16[1024]", arg182_1: "f16[4096, 1024]", arg183_1: "f16[4096]", arg184_1: "f16[1024, 4096]", arg185_1: "f16[1024]", arg186_1: "f16[1024]", arg187_1: "f16[1024]", arg188_1: "f16[1024, 1024]", arg189_1: "f16[1024]", arg190_1: "f16[1024, 1024]", arg191_1: "f16[1024, 1024]", arg192_1: "f16[1024]", arg193_1: "f16[1024, 1024]", arg194_1: "f16[1024]", arg195_1: "f16[1024]", arg196_1: "f16[1024]", arg197_1: "f16[4096, 1024]", arg198_1: "f16[4096]", arg199_1: "f16[1024, 4096]", arg200_1: "f16[1024]", arg201_1: "f16[1024]", arg202_1: "f16[1024]", arg203_1: "f16[1024, 1024]", arg204_1: "f16[1024]", arg205_1: "f16[1024, 1024]", arg206_1: "f16[1024, 1024]", arg207_1: "f16[1024]", arg208_1: "f16[1024, 1024]", arg209_1: "f16[1024]", arg210_1: "f16[1024]", arg211_1: "f16[1024]", arg212_1: "f16[4096, 1024]", arg213_1: "f16[4096]", arg214_1: "f16[1024, 4096]", arg215_1: "f16[1024]", arg216_1: "f16[1024]", arg217_1: "f16[1024]", arg218_1: "f16[1024, 1024]", arg219_1: "f16[1024]", arg220_1: "f16[1024, 1024]", arg221_1: "f16[1024, 1024]", arg222_1: "f16[1024]", arg223_1: "f16[1024, 1024]", arg224_1: "f16[1024]", arg225_1: "f16[1024]", arg226_1: "f16[1024]", arg227_1: "f16[4096, 1024]", arg228_1: "f16[4096]", arg229_1: "f16[1024, 4096]", arg230_1: "f16[1024]", arg231_1: "f16[1024]", arg232_1: "f16[1024]", arg233_1: "f16[1024, 1024]", arg234_1: "f16[1024]", arg235_1: "f16[1024, 1024]", arg236_1: "f16[1024, 1024]", arg237_1: "f16[1024]", arg238_1: "f16[1024, 1024]", arg239_1: "f16[1024]", arg240_1: "f16[1024]", arg241_1: "f16[1024]", arg242_1: "f16[4096, 1024]", arg243_1: "f16[4096]", arg244_1: "f16[1024, 4096]", arg245_1: "f16[1024]", arg246_1: "f16[1024]", arg247_1: "f16[1024]", arg248_1: "f16[1024, 1024]", arg249_1: "f16[1024]", arg250_1: "f16[1024, 1024]", arg251_1: "f16[1024, 1024]", arg252_1: "f16[1024]", arg253_1: "f16[1024, 1024]", arg254_1: "f16[1024]", arg255_1: "f16[1024]", arg256_1: "f16[1024]", arg257_1: "f16[4096, 1024]", arg258_1: "f16[4096]", arg259_1: "f16[1024, 4096]", arg260_1: "f16[1024]", arg261_1: "f16[1024]", arg262_1: "f16[1024]", arg263_1: "f16[1024, 1024]", arg264_1: "f16[1024]", arg265_1: "f16[1024, 1024]", arg266_1: "f16[1024, 1024]", arg267_1: "f16[1024]", arg268_1: "f16[1024, 1024]", arg269_1: "f16[1024]", arg270_1: "f16[1024]", arg271_1: "f16[1024]", arg272_1: "f16[4096, 1024]", arg273_1: "f16[4096]", arg274_1: "f16[1024, 4096]", arg275_1: "f16[1024]", arg276_1: "f16[1024]", arg277_1: "f16[1024]", arg278_1: "f16[1024, 1024]", arg279_1: "f16[1024]", arg280_1: "f16[1024, 1024]", arg281_1: "f16[1024, 1024]", arg282_1: "f16[1024]", arg283_1: "f16[1024, 1024]", arg284_1: "f16[1024]", arg285_1: "f16[1024]", arg286_1: "f16[1024]", arg287_1: "f16[4096, 1024]", arg288_1: "f16[4096]", arg289_1: "f16[1024, 4096]", arg290_1: "f16[1024]", arg291_1: "f16[1024]", arg292_1: "f16[1024]", arg293_1: "f16[1024, 1024]", arg294_1: "f16[1024]", arg295_1: "f16[1024, 1024]", arg296_1: "f16[1024, 1024]", arg297_1: "f16[1024]", arg298_1: "f16[1024, 1024]", arg299_1: "f16[1024]", arg300_1: "f16[1024]", arg301_1: "f16[1024]", arg302_1: "f16[4096, 1024]", arg303_1: "f16[4096]", arg304_1: "f16[1024, 4096]", arg305_1: "f16[1024]", arg306_1: "f16[1024]", arg307_1: "f16[1024]", arg308_1: "f16[1024, 1024]", arg309_1: "f16[1024]", arg310_1: "f16[1024, 1024]", arg311_1: "f16[1024, 1024]", arg312_1: "f16[1024]", arg313_1: "f16[1024, 1024]", arg314_1: "f16[1024]", arg315_1: "f16[1024]", arg316_1: "f16[1024]", arg317_1: "f16[4096, 1024]", arg318_1: "f16[4096]", arg319_1: "f16[1024, 4096]", arg320_1: "f16[1024]", arg321_1: "f16[1024]", arg322_1: "f16[1024]", arg323_1: "f16[1024, 1024]", arg324_1: "f16[1024]", arg325_1: "f16[1024, 1024]", arg326_1: "f16[1024, 1024]", arg327_1: "f16[1024]", arg328_1: "f16[1024, 1024]", arg329_1: "f16[1024]", arg330_1: "f16[1024]", arg331_1: "f16[1024]", arg332_1: "f16[4096, 1024]", arg333_1: "f16[4096]", arg334_1: "f16[1024, 4096]", arg335_1: "f16[1024]", arg336_1: "f16[1024]", arg337_1: "f16[1024]", arg338_1: "f16[1024, 1024]", arg339_1: "f16[1024]", arg340_1: "f16[1024, 1024]", arg341_1: "f16[1024, 1024]", arg342_1: "f16[1024]", arg343_1: "f16[1024, 1024]", arg344_1: "f16[1024]", arg345_1: "f16[1024]", arg346_1: "f16[1024]", arg347_1: "f16[4096, 1024]", arg348_1: "f16[4096]", arg349_1: "f16[1024, 4096]", arg350_1: "f16[1024]", arg351_1: "f16[1024]", arg352_1: "f16[1024]", arg353_1: "f16[1024, 1024]", arg354_1: "f16[1024]", arg355_1: "f16[1024, 1024]", arg356_1: "f16[1024, 1024]", arg357_1: "f16[1024]", arg358_1: "f16[1024, 1024]", arg359_1: "f16[1024]", arg360_1: "f16[1024]", arg361_1: "f16[1024]", arg362_1: "f16[4096, 1024]", arg363_1: "f16[4096]", arg364_1: "f16[1024, 4096]", arg365_1: "f16[1024]", arg366_1: "f16[1024]", arg367_1: "f16[1024]", arg368_1: "f16[256, 1024]", arg369_1: "f16[256]", arg370_1: "f16[2, 256]", arg371_1: "f16[2]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:618 in forward, code: inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        convolution: "f16[1, 1024, 3000]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [1], [1], [1], False, [0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        convert_element_type: "f32[1, 1024, 3000]" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        mul: "f32[1, 1024, 3000]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[1, 1024, 3000]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.7071067811865476);  convert_element_type = None
        erf: "f32[1, 1024, 3000]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[1, 1024, 3000]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[1, 1024, 3000]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_1: "f16[1, 1024, 3000]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:619 in forward, code: inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        convolution_1: "f16[1, 1024, 1500]" = torch.ops.aten.convolution.default(convert_element_type_1, arg3_1, arg4_1, [2], [1], [1], False, [0], 1);  convert_element_type_1 = arg3_1 = arg4_1 = None
        convert_element_type_2: "f32[1, 1024, 1500]" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        mul_3: "f32[1, 1024, 1500]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5)
        mul_4: "f32[1, 1024, 1500]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476);  convert_element_type_2 = None
        erf_1: "f32[1, 1024, 1500]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_1: "f32[1, 1024, 1500]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_5: "f32[1, 1024, 1500]" = torch.ops.aten.mul.Tensor(mul_3, add_1);  mul_3 = add_1 = None
        convert_element_type_3: "f16[1, 1024, 1500]" = torch.ops.prims.convert_element_type.default(mul_5, torch.float16);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:621 in forward, code: inputs_embeds = inputs_embeds.permute(0, 2, 1)
        permute: "f16[1, 1500, 1024]" = torch.ops.aten.permute.default(convert_element_type_3, [0, 2, 1]);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:622 in forward, code: all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)
        iota: "i64[1500]" = torch.ops.prims.iota.default(1500, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:624 in forward, code: hidden_states = inputs_embeds + self.embed_positions(all_positions)
        embedding: "f16[1500, 1024]" = torch.ops.aten.embedding.default(arg5_1, iota);  arg5_1 = iota = None
        add_2: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(permute, embedding);  permute = embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_1: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_2, memory_format = torch.contiguous_format)
        convert_element_type_4: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.float32);  clone_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 1500, 1]" = var_mean[0]
        getitem_1: "f32[1, 1500, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_1);  convert_element_type_4 = getitem_1 = None
        add_3: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_6: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_7: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_6, arg6_1);  mul_6 = arg6_1 = None
        add_4: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_7, arg7_1);  mul_7 = arg7_1 = None
        convert_element_type_5: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_4, torch.float16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_5, [1500, 1024])
        permute_1: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg9_1, view, permute_1);  arg9_1 = view = permute_1 = None
        view_1: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm, [1, 1500, 1024]);  addmm = None
        mul_8: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
        view_2: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_8, [1, 1500, -1, 64]);  mul_8 = None
        permute_2: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone_2: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_3: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_5, [1500, 1024])
        permute_3: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_3, permute_3);  view_3 = permute_3 = None
        view_4: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm, [1, 1500, 1024]);  mm = None
        view_5: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_4, [1, -1, 16, 64]);  view_4 = None
        permute_4: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_3: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_6: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_5, [1500, 1024]);  convert_element_type_5 = None
        permute_5: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_1: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg12_1, view_6, permute_5);  arg12_1 = view_6 = permute_5 = None
        view_7: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_1, [1, 1500, 1024]);  addmm_1 = None
        view_8: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_7, [1, -1, 16, 64]);  view_7 = None
        permute_6: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone_4: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_2, clone_3, clone_4, None, False, scale = 1.0);  clone_2 = clone_3 = clone_4 = None
        getitem_2: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        clone_5: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_5, [1, 1500, -1]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_9, [1500, 1024]);  view_9 = None
        permute_8: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg14_1, view_10, permute_8);  arg14_1 = view_10 = permute_8 = None
        view_11: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_2, [1, 1500, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_5: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_2, view_11);  add_2 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_7: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_5, memory_format = torch.contiguous_format)
        convert_element_type_17: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_7, torch.float32);  clone_7 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_17, [2], correction = 0, keepdim = True)
        getitem_11: "f32[1, 1500, 1]" = var_mean_1[0]
        getitem_12: "f32[1, 1500, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_17, getitem_12);  convert_element_type_17 = getitem_12 = None
        add_6: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_9: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_10: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_9, arg15_1);  mul_9 = arg15_1 = None
        add_7: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_10, arg16_1);  mul_10 = arg16_1 = None
        convert_element_type_18: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_7, torch.float16);  add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_12: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_18, [1500, 1024]);  convert_element_type_18 = None
        permute_9: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_3: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg18_1, view_12, permute_9);  arg18_1 = view_12 = permute_9 = None
        view_13: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_3, [1, 1500, 4096]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_22: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_11: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.5)
        mul_12: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.7071067811865476);  convert_element_type_22 = None
        erf_2: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_8: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_13: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_11, add_8);  mul_11 = add_8 = None
        convert_element_type_23: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_13, torch.float16);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_14: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_23, [1500, 4096]);  convert_element_type_23 = None
        permute_10: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_4: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg20_1, view_14, permute_10);  arg20_1 = view_14 = permute_10 = None
        view_15: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_4, [1, 1500, 1024]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_9: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_5, view_15);  add_5 = view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_27: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
        clamp_min: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_27, -64504.0);  convert_element_type_27 = None
        clamp_max: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min, 64504.0);  clamp_min = None
        convert_element_type_28: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.float16);  clamp_max = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_10: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_28, memory_format = torch.contiguous_format)
        convert_element_type_29: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_10, torch.float32);  clone_10 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_29, [2], correction = 0, keepdim = True)
        getitem_13: "f32[1, 1500, 1]" = var_mean_2[0]
        getitem_14: "f32[1, 1500, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_29, getitem_14);  convert_element_type_29 = getitem_14 = None
        add_10: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_14: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_15: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_14, arg21_1);  mul_14 = arg21_1 = None
        add_11: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_15, arg22_1);  mul_15 = arg22_1 = None
        convert_element_type_30: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_11, torch.float16);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_16: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_30, [1500, 1024])
        permute_11: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_5: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg24_1, view_16, permute_11);  arg24_1 = view_16 = permute_11 = None
        view_17: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_5, [1, 1500, 1024]);  addmm_5 = None
        mul_16: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_17, 0.125);  view_17 = None
        view_18: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_16, [1, 1500, -1, 64]);  mul_16 = None
        permute_12: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_11: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_19: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_30, [1500, 1024])
        permute_13: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_1: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_19, permute_13);  view_19 = permute_13 = None
        view_20: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_1, [1, 1500, 1024]);  mm_1 = None
        view_21: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_20, [1, -1, 16, 64]);  view_20 = None
        permute_14: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        clone_12: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_22: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_30, [1500, 1024]);  convert_element_type_30 = None
        permute_15: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_6: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg27_1, view_22, permute_15);  arg27_1 = view_22 = permute_15 = None
        view_23: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_6, [1, 1500, 1024]);  addmm_6 = None
        view_24: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_23, [1, -1, 16, 64]);  view_23 = None
        permute_16: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        clone_13: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_11, clone_12, clone_13, None, False, scale = 1.0);  clone_11 = clone_12 = clone_13 = None
        getitem_15: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_17: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None
        clone_14: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_25: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_14, [1, 1500, -1]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_26: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_25, [1500, 1024]);  view_25 = None
        permute_18: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_7: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg29_1, view_26, permute_18);  arg29_1 = view_26 = permute_18 = None
        view_27: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_7, [1, 1500, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_12: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_28, view_27);  convert_element_type_28 = view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_16: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_12, memory_format = torch.contiguous_format)
        convert_element_type_42: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_16, torch.float32);  clone_16 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_42, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 1500, 1]" = var_mean_3[0]
        getitem_25: "f32[1, 1500, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_42, getitem_25);  convert_element_type_42 = getitem_25 = None
        add_13: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_17: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_18: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg30_1);  mul_17 = arg30_1 = None
        add_14: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_18, arg31_1);  mul_18 = arg31_1 = None
        convert_element_type_43: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_14, torch.float16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_28: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_43, [1500, 1024]);  convert_element_type_43 = None
        permute_19: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_8: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg33_1, view_28, permute_19);  arg33_1 = view_28 = permute_19 = None
        view_29: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_8, [1, 1500, 4096]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_47: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        mul_19: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 0.5)
        mul_20: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 0.7071067811865476);  convert_element_type_47 = None
        erf_3: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_15: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_21: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_19, add_15);  mul_19 = add_15 = None
        convert_element_type_48: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_21, torch.float16);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_30: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_48, [1500, 4096]);  convert_element_type_48 = None
        permute_20: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_9: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg35_1, view_30, permute_20);  arg35_1 = view_30 = permute_20 = None
        view_31: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_9, [1, 1500, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_16: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_12, view_31);  add_12 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_52: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        clamp_min_1: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_52, -64504.0);  convert_element_type_52 = None
        clamp_max_1: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_1, 64504.0);  clamp_min_1 = None
        convert_element_type_53: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.float16);  clamp_max_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_19: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_53, memory_format = torch.contiguous_format)
        convert_element_type_54: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_19, torch.float32);  clone_19 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_54, [2], correction = 0, keepdim = True)
        getitem_26: "f32[1, 1500, 1]" = var_mean_4[0]
        getitem_27: "f32[1, 1500, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_54, getitem_27);  convert_element_type_54 = getitem_27 = None
        add_17: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_22: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_23: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_22, arg36_1);  mul_22 = arg36_1 = None
        add_18: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_23, arg37_1);  mul_23 = arg37_1 = None
        convert_element_type_55: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_18, torch.float16);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_32: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_55, [1500, 1024])
        permute_21: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_10: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg39_1, view_32, permute_21);  arg39_1 = view_32 = permute_21 = None
        view_33: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_10, [1, 1500, 1024]);  addmm_10 = None
        mul_24: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_33, 0.125);  view_33 = None
        view_34: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_24, [1, 1500, -1, 64]);  mul_24 = None
        permute_22: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        clone_20: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_35: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_55, [1500, 1024])
        permute_23: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_2: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_35, permute_23);  view_35 = permute_23 = None
        view_36: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_2, [1, 1500, 1024]);  mm_2 = None
        view_37: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_36, [1, -1, 16, 64]);  view_36 = None
        permute_24: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        clone_21: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_38: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_55, [1500, 1024]);  convert_element_type_55 = None
        permute_25: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_11: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg42_1, view_38, permute_25);  arg42_1 = view_38 = permute_25 = None
        view_39: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_11, [1, 1500, 1024]);  addmm_11 = None
        view_40: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_39, [1, -1, 16, 64]);  view_39 = None
        permute_26: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        clone_22: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_20, clone_21, clone_22, None, False, scale = 1.0);  clone_20 = clone_21 = clone_22 = None
        getitem_28: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None
        clone_23: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_41: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_23, [1, 1500, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_42: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_41, [1500, 1024]);  view_41 = None
        permute_28: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_12: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg44_1, view_42, permute_28);  arg44_1 = view_42 = permute_28 = None
        view_43: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_12, [1, 1500, 1024]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_19: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_53, view_43);  convert_element_type_53 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_25: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_19, memory_format = torch.contiguous_format)
        convert_element_type_67: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_25, torch.float32);  clone_25 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_67, [2], correction = 0, keepdim = True)
        getitem_37: "f32[1, 1500, 1]" = var_mean_5[0]
        getitem_38: "f32[1, 1500, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_67, getitem_38);  convert_element_type_67 = getitem_38 = None
        add_20: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_25: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_26: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_25, arg45_1);  mul_25 = arg45_1 = None
        add_21: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_26, arg46_1);  mul_26 = arg46_1 = None
        convert_element_type_68: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_21, torch.float16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_44: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_68, [1500, 1024]);  convert_element_type_68 = None
        permute_29: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_13: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg48_1, view_44, permute_29);  arg48_1 = view_44 = permute_29 = None
        view_45: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_13, [1, 1500, 4096]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_72: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_27: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 0.5)
        mul_28: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 0.7071067811865476);  convert_element_type_72 = None
        erf_4: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_22: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_29: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_27, add_22);  mul_27 = add_22 = None
        convert_element_type_73: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_29, torch.float16);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_46: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_73, [1500, 4096]);  convert_element_type_73 = None
        permute_30: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_14: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg50_1, view_46, permute_30);  arg50_1 = view_46 = permute_30 = None
        view_47: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_14, [1, 1500, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_23: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_19, view_47);  add_19 = view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_77: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        clamp_min_2: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_77, -64504.0);  convert_element_type_77 = None
        clamp_max_2: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_2, 64504.0);  clamp_min_2 = None
        convert_element_type_78: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.float16);  clamp_max_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_28: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_78, memory_format = torch.contiguous_format)
        convert_element_type_79: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_28, torch.float32);  clone_28 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_79, [2], correction = 0, keepdim = True)
        getitem_39: "f32[1, 1500, 1]" = var_mean_6[0]
        getitem_40: "f32[1, 1500, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_79, getitem_40);  convert_element_type_79 = getitem_40 = None
        add_24: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_30: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_31: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_30, arg51_1);  mul_30 = arg51_1 = None
        add_25: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_31, arg52_1);  mul_31 = arg52_1 = None
        convert_element_type_80: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_25, torch.float16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_48: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_80, [1500, 1024])
        permute_31: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_15: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg54_1, view_48, permute_31);  arg54_1 = view_48 = permute_31 = None
        view_49: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_15, [1, 1500, 1024]);  addmm_15 = None
        mul_32: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_49, 0.125);  view_49 = None
        view_50: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_32, [1, 1500, -1, 64]);  mul_32 = None
        permute_32: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_29: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_51: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_80, [1500, 1024])
        permute_33: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_3: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_51, permute_33);  view_51 = permute_33 = None
        view_52: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_3, [1, 1500, 1024]);  mm_3 = None
        view_53: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_52, [1, -1, 16, 64]);  view_52 = None
        permute_34: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        clone_30: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_54: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_80, [1500, 1024]);  convert_element_type_80 = None
        permute_35: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_16: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg57_1, view_54, permute_35);  arg57_1 = view_54 = permute_35 = None
        view_55: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_16, [1, 1500, 1024]);  addmm_16 = None
        view_56: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_55, [1, -1, 16, 64]);  view_55 = None
        permute_36: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        clone_31: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_29, clone_30, clone_31, None, False, scale = 1.0);  clone_29 = clone_30 = clone_31 = None
        getitem_41: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_37: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_41, [0, 2, 1, 3]);  getitem_41 = None
        clone_32: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_57: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_32, [1, 1500, -1]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_58: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_57, [1500, 1024]);  view_57 = None
        permute_38: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_17: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg59_1, view_58, permute_38);  arg59_1 = view_58 = permute_38 = None
        view_59: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_17, [1, 1500, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_26: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_78, view_59);  convert_element_type_78 = view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_34: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
        convert_element_type_92: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_34, torch.float32);  clone_34 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_92, [2], correction = 0, keepdim = True)
        getitem_50: "f32[1, 1500, 1]" = var_mean_7[0]
        getitem_51: "f32[1, 1500, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_92, getitem_51);  convert_element_type_92 = getitem_51 = None
        add_27: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_33: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_34: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_33, arg60_1);  mul_33 = arg60_1 = None
        add_28: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_34, arg61_1);  mul_34 = arg61_1 = None
        convert_element_type_93: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_28, torch.float16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_60: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_93, [1500, 1024]);  convert_element_type_93 = None
        permute_39: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_18: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg63_1, view_60, permute_39);  arg63_1 = view_60 = permute_39 = None
        view_61: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_18, [1, 1500, 4096]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_97: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        mul_35: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_97, 0.5)
        mul_36: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_97, 0.7071067811865476);  convert_element_type_97 = None
        erf_5: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_29: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_37: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_35, add_29);  mul_35 = add_29 = None
        convert_element_type_98: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_37, torch.float16);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_62: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_98, [1500, 4096]);  convert_element_type_98 = None
        permute_40: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_19: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg65_1, view_62, permute_40);  arg65_1 = view_62 = permute_40 = None
        view_63: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_19, [1, 1500, 1024]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_30: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_26, view_63);  add_26 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_102: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None
        clamp_min_3: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_102, -64504.0);  convert_element_type_102 = None
        clamp_max_3: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_3, 64504.0);  clamp_min_3 = None
        convert_element_type_103: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.float16);  clamp_max_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_37: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_103, memory_format = torch.contiguous_format)
        convert_element_type_104: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_37, torch.float32);  clone_37 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_104, [2], correction = 0, keepdim = True)
        getitem_52: "f32[1, 1500, 1]" = var_mean_8[0]
        getitem_53: "f32[1, 1500, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_8: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_104, getitem_53);  convert_element_type_104 = getitem_53 = None
        add_31: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_38: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_39: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_38, arg66_1);  mul_38 = arg66_1 = None
        add_32: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_39, arg67_1);  mul_39 = arg67_1 = None
        convert_element_type_105: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_32, torch.float16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_64: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_105, [1500, 1024])
        permute_41: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_20: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg69_1, view_64, permute_41);  arg69_1 = view_64 = permute_41 = None
        view_65: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_20, [1, 1500, 1024]);  addmm_20 = None
        mul_40: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.125);  view_65 = None
        view_66: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_40, [1, 1500, -1, 64]);  mul_40 = None
        permute_42: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        clone_38: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_67: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_105, [1500, 1024])
        permute_43: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_4: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_67, permute_43);  view_67 = permute_43 = None
        view_68: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_4, [1, 1500, 1024]);  mm_4 = None
        view_69: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_68, [1, -1, 16, 64]);  view_68 = None
        permute_44: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        clone_39: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_70: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_105, [1500, 1024]);  convert_element_type_105 = None
        permute_45: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_21: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg72_1, view_70, permute_45);  arg72_1 = view_70 = permute_45 = None
        view_71: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_21, [1, 1500, 1024]);  addmm_21 = None
        view_72: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_71, [1, -1, 16, 64]);  view_71 = None
        permute_46: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_40: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_38, clone_39, clone_40, None, False, scale = 1.0);  clone_38 = clone_39 = clone_40 = None
        getitem_54: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None
        clone_41: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_73: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_41, [1, 1500, -1]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_74: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_73, [1500, 1024]);  view_73 = None
        permute_48: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_22: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg74_1, view_74, permute_48);  arg74_1 = view_74 = permute_48 = None
        view_75: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_22, [1, 1500, 1024]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_33: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_103, view_75);  convert_element_type_103 = view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_43: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_33, memory_format = torch.contiguous_format)
        convert_element_type_117: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_43, torch.float32);  clone_43 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_117, [2], correction = 0, keepdim = True)
        getitem_63: "f32[1, 1500, 1]" = var_mean_9[0]
        getitem_64: "f32[1, 1500, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_117, getitem_64);  convert_element_type_117 = getitem_64 = None
        add_34: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
        rsqrt_9: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_41: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_42: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_41, arg75_1);  mul_41 = arg75_1 = None
        add_35: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_42, arg76_1);  mul_42 = arg76_1 = None
        convert_element_type_118: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_35, torch.float16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_76: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_118, [1500, 1024]);  convert_element_type_118 = None
        permute_49: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_23: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg78_1, view_76, permute_49);  arg78_1 = view_76 = permute_49 = None
        view_77: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_23, [1, 1500, 4096]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_122: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_43: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_122, 0.5)
        mul_44: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_122, 0.7071067811865476);  convert_element_type_122 = None
        erf_6: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
        add_36: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_45: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_43, add_36);  mul_43 = add_36 = None
        convert_element_type_123: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_45, torch.float16);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_78: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_123, [1500, 4096]);  convert_element_type_123 = None
        permute_50: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_24: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg80_1, view_78, permute_50);  arg80_1 = view_78 = permute_50 = None
        view_79: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_24, [1, 1500, 1024]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_37: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_33, view_79);  add_33 = view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_127: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None
        clamp_min_4: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_127, -64504.0);  convert_element_type_127 = None
        clamp_max_4: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_4, 64504.0);  clamp_min_4 = None
        convert_element_type_128: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_4, torch.float16);  clamp_max_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_46: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_128, memory_format = torch.contiguous_format)
        convert_element_type_129: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_46, torch.float32);  clone_46 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_129, [2], correction = 0, keepdim = True)
        getitem_65: "f32[1, 1500, 1]" = var_mean_10[0]
        getitem_66: "f32[1, 1500, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_10: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_129, getitem_66);  convert_element_type_129 = getitem_66 = None
        add_38: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
        rsqrt_10: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_46: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_47: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_46, arg81_1);  mul_46 = arg81_1 = None
        add_39: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_47, arg82_1);  mul_47 = arg82_1 = None
        convert_element_type_130: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_39, torch.float16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_80: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_130, [1500, 1024])
        permute_51: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_25: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg84_1, view_80, permute_51);  arg84_1 = view_80 = permute_51 = None
        view_81: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_25, [1, 1500, 1024]);  addmm_25 = None
        mul_48: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_81, 0.125);  view_81 = None
        view_82: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_48, [1, 1500, -1, 64]);  mul_48 = None
        permute_52: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_47: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_83: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_130, [1500, 1024])
        permute_53: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_5: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_83, permute_53);  view_83 = permute_53 = None
        view_84: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_5, [1, 1500, 1024]);  mm_5 = None
        view_85: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_84, [1, -1, 16, 64]);  view_84 = None
        permute_54: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        clone_48: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_86: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_130, [1500, 1024]);  convert_element_type_130 = None
        permute_55: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_26: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg87_1, view_86, permute_55);  arg87_1 = view_86 = permute_55 = None
        view_87: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_26, [1, 1500, 1024]);  addmm_26 = None
        view_88: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_87, [1, -1, 16, 64]);  view_87 = None
        permute_56: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        clone_49: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_47, clone_48, clone_49, None, False, scale = 1.0);  clone_47 = clone_48 = clone_49 = None
        getitem_67: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_57: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_67, [0, 2, 1, 3]);  getitem_67 = None
        clone_50: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_89: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_50, [1, 1500, -1]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_90: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_89, [1500, 1024]);  view_89 = None
        permute_58: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_27: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg89_1, view_90, permute_58);  arg89_1 = view_90 = permute_58 = None
        view_91: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_27, [1, 1500, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_40: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_128, view_91);  convert_element_type_128 = view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_52: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_40, memory_format = torch.contiguous_format)
        convert_element_type_142: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_52, torch.float32);  clone_52 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_142, [2], correction = 0, keepdim = True)
        getitem_76: "f32[1, 1500, 1]" = var_mean_11[0]
        getitem_77: "f32[1, 1500, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_11: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_142, getitem_77);  convert_element_type_142 = getitem_77 = None
        add_41: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_11: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        mul_49: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_50: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_49, arg90_1);  mul_49 = arg90_1 = None
        add_42: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_50, arg91_1);  mul_50 = arg91_1 = None
        convert_element_type_143: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_42, torch.float16);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_92: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_143, [1500, 1024]);  convert_element_type_143 = None
        permute_59: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_28: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg93_1, view_92, permute_59);  arg93_1 = view_92 = permute_59 = None
        view_93: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_28, [1, 1500, 4096]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_147: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_51: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_147, 0.5)
        mul_52: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_147, 0.7071067811865476);  convert_element_type_147 = None
        erf_7: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_43: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_53: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_51, add_43);  mul_51 = add_43 = None
        convert_element_type_148: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_53, torch.float16);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_94: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_148, [1500, 4096]);  convert_element_type_148 = None
        permute_60: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_29: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg95_1, view_94, permute_60);  arg95_1 = view_94 = permute_60 = None
        view_95: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_29, [1, 1500, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_44: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_40, view_95);  add_40 = view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_152: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32);  add_44 = None
        clamp_min_5: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_152, -64504.0);  convert_element_type_152 = None
        clamp_max_5: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_5, 64504.0);  clamp_min_5 = None
        convert_element_type_153: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_5, torch.float16);  clamp_max_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_55: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_153, memory_format = torch.contiguous_format)
        convert_element_type_154: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_55, torch.float32);  clone_55 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_154, [2], correction = 0, keepdim = True)
        getitem_78: "f32[1, 1500, 1]" = var_mean_12[0]
        getitem_79: "f32[1, 1500, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_12: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_154, getitem_79);  convert_element_type_154 = getitem_79 = None
        add_45: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_12: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_54: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_55: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_54, arg96_1);  mul_54 = arg96_1 = None
        add_46: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_55, arg97_1);  mul_55 = arg97_1 = None
        convert_element_type_155: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_46, torch.float16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_96: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_155, [1500, 1024])
        permute_61: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_30: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg99_1, view_96, permute_61);  arg99_1 = view_96 = permute_61 = None
        view_97: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_30, [1, 1500, 1024]);  addmm_30 = None
        mul_56: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_97, 0.125);  view_97 = None
        view_98: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_56, [1, 1500, -1, 64]);  mul_56 = None
        permute_62: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        clone_56: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_99: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_155, [1500, 1024])
        permute_63: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_6: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_99, permute_63);  view_99 = permute_63 = None
        view_100: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_6, [1, 1500, 1024]);  mm_6 = None
        view_101: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_100, [1, -1, 16, 64]);  view_100 = None
        permute_64: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        clone_57: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_102: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_155, [1500, 1024]);  convert_element_type_155 = None
        permute_65: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_31: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg102_1, view_102, permute_65);  arg102_1 = view_102 = permute_65 = None
        view_103: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_31, [1, 1500, 1024]);  addmm_31 = None
        view_104: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_103, [1, -1, 16, 64]);  view_103 = None
        permute_66: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_58: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_56, clone_57, clone_58, None, False, scale = 1.0);  clone_56 = clone_57 = clone_58 = None
        getitem_80: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_67: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_80, [0, 2, 1, 3]);  getitem_80 = None
        clone_59: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_59, [1, 1500, -1]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_106: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_105, [1500, 1024]);  view_105 = None
        permute_68: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_32: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg104_1, view_106, permute_68);  arg104_1 = view_106 = permute_68 = None
        view_107: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_32, [1, 1500, 1024]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_47: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_153, view_107);  convert_element_type_153 = view_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_61: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_47, memory_format = torch.contiguous_format)
        convert_element_type_167: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_61, torch.float32);  clone_61 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_167, [2], correction = 0, keepdim = True)
        getitem_89: "f32[1, 1500, 1]" = var_mean_13[0]
        getitem_90: "f32[1, 1500, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_13: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_167, getitem_90);  convert_element_type_167 = getitem_90 = None
        add_48: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_89, 1e-05);  getitem_89 = None
        rsqrt_13: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_57: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_58: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_57, arg105_1);  mul_57 = arg105_1 = None
        add_49: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_58, arg106_1);  mul_58 = arg106_1 = None
        convert_element_type_168: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_49, torch.float16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_108: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_168, [1500, 1024]);  convert_element_type_168 = None
        permute_69: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_33: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg108_1, view_108, permute_69);  arg108_1 = view_108 = permute_69 = None
        view_109: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_33, [1, 1500, 4096]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_172: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None
        mul_59: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_172, 0.5)
        mul_60: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_172, 0.7071067811865476);  convert_element_type_172 = None
        erf_8: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_50: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_61: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_59, add_50);  mul_59 = add_50 = None
        convert_element_type_173: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_61, torch.float16);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_110: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_173, [1500, 4096]);  convert_element_type_173 = None
        permute_70: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_34: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg110_1, view_110, permute_70);  arg110_1 = view_110 = permute_70 = None
        view_111: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_34, [1, 1500, 1024]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_51: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_47, view_111);  add_47 = view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_177: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32);  add_51 = None
        clamp_min_6: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_177, -64504.0);  convert_element_type_177 = None
        clamp_max_6: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_6, 64504.0);  clamp_min_6 = None
        convert_element_type_178: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_6, torch.float16);  clamp_max_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_64: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_178, memory_format = torch.contiguous_format)
        convert_element_type_179: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_64, torch.float32);  clone_64 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_179, [2], correction = 0, keepdim = True)
        getitem_91: "f32[1, 1500, 1]" = var_mean_14[0]
        getitem_92: "f32[1, 1500, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_14: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_179, getitem_92);  convert_element_type_179 = getitem_92 = None
        add_52: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
        rsqrt_14: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_62: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_63: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_62, arg111_1);  mul_62 = arg111_1 = None
        add_53: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_63, arg112_1);  mul_63 = arg112_1 = None
        convert_element_type_180: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_53, torch.float16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_112: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_180, [1500, 1024])
        permute_71: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_35: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg114_1, view_112, permute_71);  arg114_1 = view_112 = permute_71 = None
        view_113: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_35, [1, 1500, 1024]);  addmm_35 = None
        mul_64: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_113, 0.125);  view_113 = None
        view_114: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_64, [1, 1500, -1, 64]);  mul_64 = None
        permute_72: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        clone_65: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_115: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_180, [1500, 1024])
        permute_73: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_7: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_115, permute_73);  view_115 = permute_73 = None
        view_116: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_7, [1, 1500, 1024]);  mm_7 = None
        view_117: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_116, [1, -1, 16, 64]);  view_116 = None
        permute_74: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        clone_66: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_118: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_180, [1500, 1024]);  convert_element_type_180 = None
        permute_75: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_36: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg117_1, view_118, permute_75);  arg117_1 = view_118 = permute_75 = None
        view_119: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_36, [1, 1500, 1024]);  addmm_36 = None
        view_120: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_119, [1, -1, 16, 64]);  view_119 = None
        permute_76: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        clone_67: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_76, memory_format = torch.contiguous_format);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_65, clone_66, clone_67, None, False, scale = 1.0);  clone_65 = clone_66 = clone_67 = None
        getitem_93: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_77: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        clone_68: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_121: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_68, [1, 1500, -1]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_122: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_121, [1500, 1024]);  view_121 = None
        permute_78: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_37: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg119_1, view_122, permute_78);  arg119_1 = view_122 = permute_78 = None
        view_123: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_37, [1, 1500, 1024]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_54: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_178, view_123);  convert_element_type_178 = view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_70: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_54, memory_format = torch.contiguous_format)
        convert_element_type_192: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_70, torch.float32);  clone_70 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_192, [2], correction = 0, keepdim = True)
        getitem_102: "f32[1, 1500, 1]" = var_mean_15[0]
        getitem_103: "f32[1, 1500, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_192, getitem_103);  convert_element_type_192 = getitem_103 = None
        add_55: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_15: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_65: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_66: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_65, arg120_1);  mul_65 = arg120_1 = None
        add_56: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_66, arg121_1);  mul_66 = arg121_1 = None
        convert_element_type_193: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_56, torch.float16);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_124: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_193, [1500, 1024]);  convert_element_type_193 = None
        permute_79: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_38: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg123_1, view_124, permute_79);  arg123_1 = view_124 = permute_79 = None
        view_125: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_38, [1, 1500, 4096]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_197: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None
        mul_67: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_197, 0.5)
        mul_68: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_197, 0.7071067811865476);  convert_element_type_197 = None
        erf_9: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_57: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_67, add_57);  mul_67 = add_57 = None
        convert_element_type_198: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_69, torch.float16);  mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_126: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_198, [1500, 4096]);  convert_element_type_198 = None
        permute_80: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_39: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg125_1, view_126, permute_80);  arg125_1 = view_126 = permute_80 = None
        view_127: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_39, [1, 1500, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_58: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_54, view_127);  add_54 = view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_202: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32);  add_58 = None
        clamp_min_7: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_202, -64504.0);  convert_element_type_202 = None
        clamp_max_7: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_7, 64504.0);  clamp_min_7 = None
        convert_element_type_203: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_7, torch.float16);  clamp_max_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_73: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_203, memory_format = torch.contiguous_format)
        convert_element_type_204: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_73, torch.float32);  clone_73 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_204, [2], correction = 0, keepdim = True)
        getitem_104: "f32[1, 1500, 1]" = var_mean_16[0]
        getitem_105: "f32[1, 1500, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_16: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_204, getitem_105);  convert_element_type_204 = getitem_105 = None
        add_59: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_16: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_70: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_71: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_70, arg126_1);  mul_70 = arg126_1 = None
        add_60: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_71, arg127_1);  mul_71 = arg127_1 = None
        convert_element_type_205: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_60, torch.float16);  add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_128: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_205, [1500, 1024])
        permute_81: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_40: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg129_1, view_128, permute_81);  arg129_1 = view_128 = permute_81 = None
        view_129: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_40, [1, 1500, 1024]);  addmm_40 = None
        mul_72: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_129, 0.125);  view_129 = None
        view_130: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_72, [1, 1500, -1, 64]);  mul_72 = None
        permute_82: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        clone_74: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_131: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_205, [1500, 1024])
        permute_83: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        mm_8: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_131, permute_83);  view_131 = permute_83 = None
        view_132: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_8, [1, 1500, 1024]);  mm_8 = None
        view_133: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_132, [1, -1, 16, 64]);  view_132 = None
        permute_84: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        clone_75: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_134: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_205, [1500, 1024]);  convert_element_type_205 = None
        permute_85: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_41: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg132_1, view_134, permute_85);  arg132_1 = view_134 = permute_85 = None
        view_135: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_41, [1, 1500, 1024]);  addmm_41 = None
        view_136: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_135, [1, -1, 16, 64]);  view_135 = None
        permute_86: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        clone_76: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_74, clone_75, clone_76, None, False, scale = 1.0);  clone_74 = clone_75 = clone_76 = None
        getitem_106: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_87: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None
        clone_77: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_137: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_77, [1, 1500, -1]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_138: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_137, [1500, 1024]);  view_137 = None
        permute_88: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_42: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg134_1, view_138, permute_88);  arg134_1 = view_138 = permute_88 = None
        view_139: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_42, [1, 1500, 1024]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_61: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_203, view_139);  convert_element_type_203 = view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_79: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_61, memory_format = torch.contiguous_format)
        convert_element_type_217: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_79, torch.float32);  clone_79 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_217, [2], correction = 0, keepdim = True)
        getitem_115: "f32[1, 1500, 1]" = var_mean_17[0]
        getitem_116: "f32[1, 1500, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_17: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_217, getitem_116);  convert_element_type_217 = getitem_116 = None
        add_62: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
        rsqrt_17: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_73: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_74: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_73, arg135_1);  mul_73 = arg135_1 = None
        add_63: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_74, arg136_1);  mul_74 = arg136_1 = None
        convert_element_type_218: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_63, torch.float16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_140: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_218, [1500, 1024]);  convert_element_type_218 = None
        permute_89: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_43: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg138_1, view_140, permute_89);  arg138_1 = view_140 = permute_89 = None
        view_141: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_43, [1, 1500, 4096]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_222: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_75: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_222, 0.5)
        mul_76: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_222, 0.7071067811865476);  convert_element_type_222 = None
        erf_10: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_64: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_77: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_75, add_64);  mul_75 = add_64 = None
        convert_element_type_223: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_77, torch.float16);  mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_142: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_223, [1500, 4096]);  convert_element_type_223 = None
        permute_90: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_44: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg140_1, view_142, permute_90);  arg140_1 = view_142 = permute_90 = None
        view_143: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_44, [1, 1500, 1024]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_65: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_61, view_143);  add_61 = view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_227: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_65, torch.float32);  add_65 = None
        clamp_min_8: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_227, -64504.0);  convert_element_type_227 = None
        clamp_max_8: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_8, 64504.0);  clamp_min_8 = None
        convert_element_type_228: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_8, torch.float16);  clamp_max_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_82: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_228, memory_format = torch.contiguous_format)
        convert_element_type_229: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_82, torch.float32);  clone_82 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_229, [2], correction = 0, keepdim = True)
        getitem_117: "f32[1, 1500, 1]" = var_mean_18[0]
        getitem_118: "f32[1, 1500, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_229, getitem_118);  convert_element_type_229 = getitem_118 = None
        add_66: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_117, 1e-05);  getitem_117 = None
        rsqrt_18: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_78: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_79: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_78, arg141_1);  mul_78 = arg141_1 = None
        add_67: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_79, arg142_1);  mul_79 = arg142_1 = None
        convert_element_type_230: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_67, torch.float16);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_144: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_230, [1500, 1024])
        permute_91: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_45: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg144_1, view_144, permute_91);  arg144_1 = view_144 = permute_91 = None
        view_145: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_45, [1, 1500, 1024]);  addmm_45 = None
        mul_80: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_145, 0.125);  view_145 = None
        view_146: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_80, [1, 1500, -1, 64]);  mul_80 = None
        permute_92: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_83: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_147: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_230, [1500, 1024])
        permute_93: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        mm_9: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_147, permute_93);  view_147 = permute_93 = None
        view_148: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_9, [1, 1500, 1024]);  mm_9 = None
        view_149: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_148, [1, -1, 16, 64]);  view_148 = None
        permute_94: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        clone_84: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_150: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_230, [1500, 1024]);  convert_element_type_230 = None
        permute_95: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_46: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg147_1, view_150, permute_95);  arg147_1 = view_150 = permute_95 = None
        view_151: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_46, [1, 1500, 1024]);  addmm_46 = None
        view_152: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_151, [1, -1, 16, 64]);  view_151 = None
        permute_96: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        clone_85: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_83, clone_84, clone_85, None, False, scale = 1.0);  clone_83 = clone_84 = clone_85 = None
        getitem_119: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_97: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None
        clone_86: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_153: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_86, [1, 1500, -1]);  clone_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_154: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_153, [1500, 1024]);  view_153 = None
        permute_98: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_47: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg149_1, view_154, permute_98);  arg149_1 = view_154 = permute_98 = None
        view_155: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_47, [1, 1500, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_68: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_228, view_155);  convert_element_type_228 = view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_88: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_68, memory_format = torch.contiguous_format)
        convert_element_type_242: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_88, torch.float32);  clone_88 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_242, [2], correction = 0, keepdim = True)
        getitem_128: "f32[1, 1500, 1]" = var_mean_19[0]
        getitem_129: "f32[1, 1500, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_19: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_242, getitem_129);  convert_element_type_242 = getitem_129 = None
        add_69: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_19: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_81: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_82: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_81, arg150_1);  mul_81 = arg150_1 = None
        add_70: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_82, arg151_1);  mul_82 = arg151_1 = None
        convert_element_type_243: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_70, torch.float16);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_156: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_243, [1500, 1024]);  convert_element_type_243 = None
        permute_99: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_48: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg153_1, view_156, permute_99);  arg153_1 = view_156 = permute_99 = None
        view_157: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_48, [1, 1500, 4096]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_247: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_157, torch.float32);  view_157 = None
        mul_83: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_247, 0.5)
        mul_84: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_247, 0.7071067811865476);  convert_element_type_247 = None
        erf_11: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_71: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_85: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_83, add_71);  mul_83 = add_71 = None
        convert_element_type_248: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_85, torch.float16);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_158: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_248, [1500, 4096]);  convert_element_type_248 = None
        permute_100: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_49: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg155_1, view_158, permute_100);  arg155_1 = view_158 = permute_100 = None
        view_159: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_49, [1, 1500, 1024]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_72: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_68, view_159);  add_68 = view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_252: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_72, torch.float32);  add_72 = None
        clamp_min_9: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_252, -64504.0);  convert_element_type_252 = None
        clamp_max_9: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_9, 64504.0);  clamp_min_9 = None
        convert_element_type_253: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_9, torch.float16);  clamp_max_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_91: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_253, memory_format = torch.contiguous_format)
        convert_element_type_254: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_91, torch.float32);  clone_91 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_254, [2], correction = 0, keepdim = True)
        getitem_130: "f32[1, 1500, 1]" = var_mean_20[0]
        getitem_131: "f32[1, 1500, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_20: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_254, getitem_131);  convert_element_type_254 = getitem_131 = None
        add_73: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_20: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_86: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_87: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_86, arg156_1);  mul_86 = arg156_1 = None
        add_74: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_87, arg157_1);  mul_87 = arg157_1 = None
        convert_element_type_255: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_74, torch.float16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_160: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_255, [1500, 1024])
        permute_101: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_50: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg159_1, view_160, permute_101);  arg159_1 = view_160 = permute_101 = None
        view_161: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_50, [1, 1500, 1024]);  addmm_50 = None
        mul_88: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_161, 0.125);  view_161 = None
        view_162: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_88, [1, 1500, -1, 64]);  mul_88 = None
        permute_102: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        clone_92: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_163: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_255, [1500, 1024])
        permute_103: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_10: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_163, permute_103);  view_163 = permute_103 = None
        view_164: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_10, [1, 1500, 1024]);  mm_10 = None
        view_165: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_164, [1, -1, 16, 64]);  view_164 = None
        permute_104: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        clone_93: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_166: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_255, [1500, 1024]);  convert_element_type_255 = None
        permute_105: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_51: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg162_1, view_166, permute_105);  arg162_1 = view_166 = permute_105 = None
        view_167: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_51, [1, 1500, 1024]);  addmm_51 = None
        view_168: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_167, [1, -1, 16, 64]);  view_167 = None
        permute_106: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_94: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_92, clone_93, clone_94, None, False, scale = 1.0);  clone_92 = clone_93 = clone_94 = None
        getitem_132: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None
        clone_95: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_95, [1, 1500, -1]);  clone_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_170: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_169, [1500, 1024]);  view_169 = None
        permute_108: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_52: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg164_1, view_170, permute_108);  arg164_1 = view_170 = permute_108 = None
        view_171: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_52, [1, 1500, 1024]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_75: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_253, view_171);  convert_element_type_253 = view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_97: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_75, memory_format = torch.contiguous_format)
        convert_element_type_267: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_97, torch.float32);  clone_97 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_267, [2], correction = 0, keepdim = True)
        getitem_141: "f32[1, 1500, 1]" = var_mean_21[0]
        getitem_142: "f32[1, 1500, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_21: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_267, getitem_142);  convert_element_type_267 = getitem_142 = None
        add_76: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
        rsqrt_21: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_89: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_90: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_89, arg165_1);  mul_89 = arg165_1 = None
        add_77: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_90, arg166_1);  mul_90 = arg166_1 = None
        convert_element_type_268: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_77, torch.float16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_172: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_268, [1500, 1024]);  convert_element_type_268 = None
        permute_109: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_53: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg168_1, view_172, permute_109);  arg168_1 = view_172 = permute_109 = None
        view_173: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_53, [1, 1500, 4096]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_272: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_91: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_272, 0.5)
        mul_92: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_272, 0.7071067811865476);  convert_element_type_272 = None
        erf_12: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_78: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_93: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_91, add_78);  mul_91 = add_78 = None
        convert_element_type_273: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_93, torch.float16);  mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_174: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_273, [1500, 4096]);  convert_element_type_273 = None
        permute_110: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_54: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg170_1, view_174, permute_110);  arg170_1 = view_174 = permute_110 = None
        view_175: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_54, [1, 1500, 1024]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_79: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_75, view_175);  add_75 = view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_277: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32);  add_79 = None
        clamp_min_10: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_277, -64504.0);  convert_element_type_277 = None
        clamp_max_10: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_10, 64504.0);  clamp_min_10 = None
        convert_element_type_278: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_10, torch.float16);  clamp_max_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_100: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_278, memory_format = torch.contiguous_format)
        convert_element_type_279: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_100, torch.float32);  clone_100 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_279, [2], correction = 0, keepdim = True)
        getitem_143: "f32[1, 1500, 1]" = var_mean_22[0]
        getitem_144: "f32[1, 1500, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_22: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_279, getitem_144);  convert_element_type_279 = getitem_144 = None
        add_80: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_22: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_94: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_95: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_94, arg171_1);  mul_94 = arg171_1 = None
        add_81: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_95, arg172_1);  mul_95 = arg172_1 = None
        convert_element_type_280: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_81, torch.float16);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_176: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_280, [1500, 1024])
        permute_111: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_55: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg174_1, view_176, permute_111);  arg174_1 = view_176 = permute_111 = None
        view_177: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_55, [1, 1500, 1024]);  addmm_55 = None
        mul_96: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_177, 0.125);  view_177 = None
        view_178: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_96, [1, 1500, -1, 64]);  mul_96 = None
        permute_112: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        clone_101: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_179: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_280, [1500, 1024])
        permute_113: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        mm_11: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_179, permute_113);  view_179 = permute_113 = None
        view_180: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_11, [1, 1500, 1024]);  mm_11 = None
        view_181: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_180, [1, -1, 16, 64]);  view_180 = None
        permute_114: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        clone_102: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_182: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_280, [1500, 1024]);  convert_element_type_280 = None
        permute_115: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_56: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg177_1, view_182, permute_115);  arg177_1 = view_182 = permute_115 = None
        view_183: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_56, [1, 1500, 1024]);  addmm_56 = None
        view_184: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_183, [1, -1, 16, 64]);  view_183 = None
        permute_116: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        clone_103: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_101, clone_102, clone_103, None, False, scale = 1.0);  clone_101 = clone_102 = clone_103 = None
        getitem_145: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None
        clone_104: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_185: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_104, [1, 1500, -1]);  clone_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_186: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_185, [1500, 1024]);  view_185 = None
        permute_118: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_57: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg179_1, view_186, permute_118);  arg179_1 = view_186 = permute_118 = None
        view_187: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_57, [1, 1500, 1024]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_82: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_278, view_187);  convert_element_type_278 = view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_106: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_82, memory_format = torch.contiguous_format)
        convert_element_type_292: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_106, torch.float32);  clone_106 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_292, [2], correction = 0, keepdim = True)
        getitem_154: "f32[1, 1500, 1]" = var_mean_23[0]
        getitem_155: "f32[1, 1500, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_23: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_292, getitem_155);  convert_element_type_292 = getitem_155 = None
        add_83: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_23: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_97: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_98: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_97, arg180_1);  mul_97 = arg180_1 = None
        add_84: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_98, arg181_1);  mul_98 = arg181_1 = None
        convert_element_type_293: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_84, torch.float16);  add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_188: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_293, [1500, 1024]);  convert_element_type_293 = None
        permute_119: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_58: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg183_1, view_188, permute_119);  arg183_1 = view_188 = permute_119 = None
        view_189: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_58, [1, 1500, 4096]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_297: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_189, torch.float32);  view_189 = None
        mul_99: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_297, 0.5)
        mul_100: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_297, 0.7071067811865476);  convert_element_type_297 = None
        erf_13: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
        add_85: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_101: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_99, add_85);  mul_99 = add_85 = None
        convert_element_type_298: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_101, torch.float16);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_190: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_298, [1500, 4096]);  convert_element_type_298 = None
        permute_120: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_59: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg185_1, view_190, permute_120);  arg185_1 = view_190 = permute_120 = None
        view_191: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_59, [1, 1500, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_86: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_82, view_191);  add_82 = view_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_302: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_86, torch.float32);  add_86 = None
        clamp_min_11: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_302, -64504.0);  convert_element_type_302 = None
        clamp_max_11: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_11, 64504.0);  clamp_min_11 = None
        convert_element_type_303: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_11, torch.float16);  clamp_max_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_109: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_303, memory_format = torch.contiguous_format)
        convert_element_type_304: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_109, torch.float32);  clone_109 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_304, [2], correction = 0, keepdim = True)
        getitem_156: "f32[1, 1500, 1]" = var_mean_24[0]
        getitem_157: "f32[1, 1500, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_24: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_304, getitem_157);  convert_element_type_304 = getitem_157 = None
        add_87: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_24: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_102: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_103: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_102, arg186_1);  mul_102 = arg186_1 = None
        add_88: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_103, arg187_1);  mul_103 = arg187_1 = None
        convert_element_type_305: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_88, torch.float16);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_192: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_305, [1500, 1024])
        permute_121: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_60: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg189_1, view_192, permute_121);  arg189_1 = view_192 = permute_121 = None
        view_193: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_60, [1, 1500, 1024]);  addmm_60 = None
        mul_104: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_193, 0.125);  view_193 = None
        view_194: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_104, [1, 1500, -1, 64]);  mul_104 = None
        permute_122: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        clone_110: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_195: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_305, [1500, 1024])
        permute_123: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        mm_12: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_195, permute_123);  view_195 = permute_123 = None
        view_196: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_12, [1, 1500, 1024]);  mm_12 = None
        view_197: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_196, [1, -1, 16, 64]);  view_196 = None
        permute_124: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_197, [0, 2, 1, 3]);  view_197 = None
        clone_111: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_198: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_305, [1500, 1024]);  convert_element_type_305 = None
        permute_125: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_61: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg192_1, view_198, permute_125);  arg192_1 = view_198 = permute_125 = None
        view_199: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_61, [1, 1500, 1024]);  addmm_61 = None
        view_200: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_199, [1, -1, 16, 64]);  view_199 = None
        permute_126: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        clone_112: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_110, clone_111, clone_112, None, False, scale = 1.0);  clone_110 = clone_111 = clone_112 = None
        getitem_158: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_12[0];  _scaled_dot_product_cudnn_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_127: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_158, [0, 2, 1, 3]);  getitem_158 = None
        clone_113: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_201: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_113, [1, 1500, -1]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_202: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_201, [1500, 1024]);  view_201 = None
        permute_128: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_62: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg194_1, view_202, permute_128);  arg194_1 = view_202 = permute_128 = None
        view_203: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_62, [1, 1500, 1024]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_89: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_303, view_203);  convert_element_type_303 = view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_115: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_89, memory_format = torch.contiguous_format)
        convert_element_type_317: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_115, torch.float32);  clone_115 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_317, [2], correction = 0, keepdim = True)
        getitem_167: "f32[1, 1500, 1]" = var_mean_25[0]
        getitem_168: "f32[1, 1500, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_25: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_317, getitem_168);  convert_element_type_317 = getitem_168 = None
        add_90: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-05);  getitem_167 = None
        rsqrt_25: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_105: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        mul_106: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_105, arg195_1);  mul_105 = arg195_1 = None
        add_91: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_106, arg196_1);  mul_106 = arg196_1 = None
        convert_element_type_318: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_91, torch.float16);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_204: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_318, [1500, 1024]);  convert_element_type_318 = None
        permute_129: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_63: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg198_1, view_204, permute_129);  arg198_1 = view_204 = permute_129 = None
        view_205: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_63, [1, 1500, 4096]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_322: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        mul_107: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_322, 0.5)
        mul_108: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_322, 0.7071067811865476);  convert_element_type_322 = None
        erf_14: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
        add_92: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_109: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_107, add_92);  mul_107 = add_92 = None
        convert_element_type_323: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_109, torch.float16);  mul_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_206: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_323, [1500, 4096]);  convert_element_type_323 = None
        permute_130: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_64: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg200_1, view_206, permute_130);  arg200_1 = view_206 = permute_130 = None
        view_207: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_64, [1, 1500, 1024]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_93: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_89, view_207);  add_89 = view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_327: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_93, torch.float32);  add_93 = None
        clamp_min_12: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_327, -64504.0);  convert_element_type_327 = None
        clamp_max_12: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_12, 64504.0);  clamp_min_12 = None
        convert_element_type_328: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_12, torch.float16);  clamp_max_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_118: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_328, memory_format = torch.contiguous_format)
        convert_element_type_329: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_118, torch.float32);  clone_118 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_329, [2], correction = 0, keepdim = True)
        getitem_169: "f32[1, 1500, 1]" = var_mean_26[0]
        getitem_170: "f32[1, 1500, 1]" = var_mean_26[1];  var_mean_26 = None
        sub_26: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_329, getitem_170);  convert_element_type_329 = getitem_170 = None
        add_94: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_169, 1e-05);  getitem_169 = None
        rsqrt_26: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_110: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
        mul_111: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_110, arg201_1);  mul_110 = arg201_1 = None
        add_95: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_111, arg202_1);  mul_111 = arg202_1 = None
        convert_element_type_330: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_95, torch.float16);  add_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_208: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_330, [1500, 1024])
        permute_131: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_65: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg204_1, view_208, permute_131);  arg204_1 = view_208 = permute_131 = None
        view_209: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_65, [1, 1500, 1024]);  addmm_65 = None
        mul_112: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_209, 0.125);  view_209 = None
        view_210: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_112, [1, 1500, -1, 64]);  mul_112 = None
        permute_132: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
        clone_119: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_211: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_330, [1500, 1024])
        permute_133: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        mm_13: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_211, permute_133);  view_211 = permute_133 = None
        view_212: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_13, [1, 1500, 1024]);  mm_13 = None
        view_213: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_212, [1, -1, 16, 64]);  view_212 = None
        permute_134: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        clone_120: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_214: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_330, [1500, 1024]);  convert_element_type_330 = None
        permute_135: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_66: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg207_1, view_214, permute_135);  arg207_1 = view_214 = permute_135 = None
        view_215: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_66, [1, 1500, 1024]);  addmm_66 = None
        view_216: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_215, [1, -1, 16, 64]);  view_215 = None
        permute_136: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
        clone_121: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_119, clone_120, clone_121, None, False, scale = 1.0);  clone_119 = clone_120 = clone_121 = None
        getitem_171: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_13[0];  _scaled_dot_product_cudnn_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_137: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None
        clone_122: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_217: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_122, [1, 1500, -1]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_218: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_217, [1500, 1024]);  view_217 = None
        permute_138: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_67: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg209_1, view_218, permute_138);  arg209_1 = view_218 = permute_138 = None
        view_219: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_67, [1, 1500, 1024]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_96: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_328, view_219);  convert_element_type_328 = view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_124: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_96, memory_format = torch.contiguous_format)
        convert_element_type_342: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_124, torch.float32);  clone_124 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_342, [2], correction = 0, keepdim = True)
        getitem_180: "f32[1, 1500, 1]" = var_mean_27[0]
        getitem_181: "f32[1, 1500, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_27: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_342, getitem_181);  convert_element_type_342 = getitem_181 = None
        add_97: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05);  getitem_180 = None
        rsqrt_27: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        mul_113: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
        mul_114: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_113, arg210_1);  mul_113 = arg210_1 = None
        add_98: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_114, arg211_1);  mul_114 = arg211_1 = None
        convert_element_type_343: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_98, torch.float16);  add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_220: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_343, [1500, 1024]);  convert_element_type_343 = None
        permute_139: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        addmm_68: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg213_1, view_220, permute_139);  arg213_1 = view_220 = permute_139 = None
        view_221: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_68, [1, 1500, 4096]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_347: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_221, torch.float32);  view_221 = None
        mul_115: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.5)
        mul_116: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.7071067811865476);  convert_element_type_347 = None
        erf_15: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_99: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_117: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_115, add_99);  mul_115 = add_99 = None
        convert_element_type_348: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_117, torch.float16);  mul_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_222: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_348, [1500, 4096]);  convert_element_type_348 = None
        permute_140: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        addmm_69: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg215_1, view_222, permute_140);  arg215_1 = view_222 = permute_140 = None
        view_223: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_69, [1, 1500, 1024]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_100: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_96, view_223);  add_96 = view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_352: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_100, torch.float32);  add_100 = None
        clamp_min_13: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_352, -64504.0);  convert_element_type_352 = None
        clamp_max_13: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_13, 64504.0);  clamp_min_13 = None
        convert_element_type_353: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_13, torch.float16);  clamp_max_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_127: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_353, memory_format = torch.contiguous_format)
        convert_element_type_354: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_127, torch.float32);  clone_127 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_354, [2], correction = 0, keepdim = True)
        getitem_182: "f32[1, 1500, 1]" = var_mean_28[0]
        getitem_183: "f32[1, 1500, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_28: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_354, getitem_183);  convert_element_type_354 = getitem_183 = None
        add_101: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-05);  getitem_182 = None
        rsqrt_28: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        mul_118: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
        mul_119: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_118, arg216_1);  mul_118 = arg216_1 = None
        add_102: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_119, arg217_1);  mul_119 = arg217_1 = None
        convert_element_type_355: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_102, torch.float16);  add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_224: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_355, [1500, 1024])
        permute_141: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_70: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg219_1, view_224, permute_141);  arg219_1 = view_224 = permute_141 = None
        view_225: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_70, [1, 1500, 1024]);  addmm_70 = None
        mul_120: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_225, 0.125);  view_225 = None
        view_226: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_120, [1, 1500, -1, 64]);  mul_120 = None
        permute_142: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_128: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_227: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_355, [1500, 1024])
        permute_143: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        mm_14: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_227, permute_143);  view_227 = permute_143 = None
        view_228: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_14, [1, 1500, 1024]);  mm_14 = None
        view_229: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_228, [1, -1, 16, 64]);  view_228 = None
        permute_144: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        clone_129: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_230: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_355, [1500, 1024]);  convert_element_type_355 = None
        permute_145: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        addmm_71: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg222_1, view_230, permute_145);  arg222_1 = view_230 = permute_145 = None
        view_231: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_71, [1, 1500, 1024]);  addmm_71 = None
        view_232: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_231, [1, -1, 16, 64]);  view_231 = None
        permute_146: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
        clone_130: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_128, clone_129, clone_130, None, False, scale = 1.0);  clone_128 = clone_129 = clone_130 = None
        getitem_184: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_14[0];  _scaled_dot_product_cudnn_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_147: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_184, [0, 2, 1, 3]);  getitem_184 = None
        clone_131: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_233: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_131, [1, 1500, -1]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_234: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_233, [1500, 1024]);  view_233 = None
        permute_148: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_72: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg224_1, view_234, permute_148);  arg224_1 = view_234 = permute_148 = None
        view_235: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_72, [1, 1500, 1024]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_103: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_353, view_235);  convert_element_type_353 = view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_133: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_103, memory_format = torch.contiguous_format)
        convert_element_type_367: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_133, torch.float32);  clone_133 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_367, [2], correction = 0, keepdim = True)
        getitem_193: "f32[1, 1500, 1]" = var_mean_29[0]
        getitem_194: "f32[1, 1500, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_29: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_367, getitem_194);  convert_element_type_367 = getitem_194 = None
        add_104: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_193, 1e-05);  getitem_193 = None
        rsqrt_29: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_121: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        mul_122: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_121, arg225_1);  mul_121 = arg225_1 = None
        add_105: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_122, arg226_1);  mul_122 = arg226_1 = None
        convert_element_type_368: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_105, torch.float16);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_236: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_368, [1500, 1024]);  convert_element_type_368 = None
        permute_149: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_73: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg228_1, view_236, permute_149);  arg228_1 = view_236 = permute_149 = None
        view_237: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_73, [1, 1500, 4096]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_372: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_237, torch.float32);  view_237 = None
        mul_123: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_372, 0.5)
        mul_124: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_372, 0.7071067811865476);  convert_element_type_372 = None
        erf_16: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_124);  mul_124 = None
        add_106: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_125: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_123, add_106);  mul_123 = add_106 = None
        convert_element_type_373: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_125, torch.float16);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_238: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_373, [1500, 4096]);  convert_element_type_373 = None
        permute_150: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_74: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg230_1, view_238, permute_150);  arg230_1 = view_238 = permute_150 = None
        view_239: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_74, [1, 1500, 1024]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_107: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_103, view_239);  add_103 = view_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_377: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32);  add_107 = None
        clamp_min_14: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_377, -64504.0);  convert_element_type_377 = None
        clamp_max_14: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_14, 64504.0);  clamp_min_14 = None
        convert_element_type_378: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_14, torch.float16);  clamp_max_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_136: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_378, memory_format = torch.contiguous_format)
        convert_element_type_379: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_136, torch.float32);  clone_136 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_379, [2], correction = 0, keepdim = True)
        getitem_195: "f32[1, 1500, 1]" = var_mean_30[0]
        getitem_196: "f32[1, 1500, 1]" = var_mean_30[1];  var_mean_30 = None
        sub_30: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_379, getitem_196);  convert_element_type_379 = getitem_196 = None
        add_108: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_195, 1e-05);  getitem_195 = None
        rsqrt_30: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_126: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
        mul_127: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_126, arg231_1);  mul_126 = arg231_1 = None
        add_109: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_127, arg232_1);  mul_127 = arg232_1 = None
        convert_element_type_380: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_109, torch.float16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_240: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_380, [1500, 1024])
        permute_151: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_75: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg234_1, view_240, permute_151);  arg234_1 = view_240 = permute_151 = None
        view_241: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_75, [1, 1500, 1024]);  addmm_75 = None
        mul_128: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.125);  view_241 = None
        view_242: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_128, [1, 1500, -1, 64]);  mul_128 = None
        permute_152: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None
        clone_137: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_243: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_380, [1500, 1024])
        permute_153: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        mm_15: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_243, permute_153);  view_243 = permute_153 = None
        view_244: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_15, [1, 1500, 1024]);  mm_15 = None
        view_245: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_244, [1, -1, 16, 64]);  view_244 = None
        permute_154: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
        clone_138: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_246: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_380, [1500, 1024]);  convert_element_type_380 = None
        permute_155: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        addmm_76: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg237_1, view_246, permute_155);  arg237_1 = view_246 = permute_155 = None
        view_247: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_76, [1, 1500, 1024]);  addmm_76 = None
        view_248: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_247, [1, -1, 16, 64]);  view_247 = None
        permute_156: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        clone_139: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_137, clone_138, clone_139, None, False, scale = 1.0);  clone_137 = clone_138 = clone_139 = None
        getitem_197: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_15[0];  _scaled_dot_product_cudnn_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_157: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_197, [0, 2, 1, 3]);  getitem_197 = None
        clone_140: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_249: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_140, [1, 1500, -1]);  clone_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_250: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_249, [1500, 1024]);  view_249 = None
        permute_158: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        addmm_77: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg239_1, view_250, permute_158);  arg239_1 = view_250 = permute_158 = None
        view_251: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_77, [1, 1500, 1024]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_110: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_378, view_251);  convert_element_type_378 = view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_142: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_110, memory_format = torch.contiguous_format)
        convert_element_type_392: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_142, torch.float32);  clone_142 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_392, [2], correction = 0, keepdim = True)
        getitem_206: "f32[1, 1500, 1]" = var_mean_31[0]
        getitem_207: "f32[1, 1500, 1]" = var_mean_31[1];  var_mean_31 = None
        sub_31: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_392, getitem_207);  convert_element_type_392 = getitem_207 = None
        add_111: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_31: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_129: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
        mul_130: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_129, arg240_1);  mul_129 = arg240_1 = None
        add_112: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_130, arg241_1);  mul_130 = arg241_1 = None
        convert_element_type_393: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_112, torch.float16);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_252: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_393, [1500, 1024]);  convert_element_type_393 = None
        permute_159: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        addmm_78: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg243_1, view_252, permute_159);  arg243_1 = view_252 = permute_159 = None
        view_253: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_78, [1, 1500, 4096]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_397: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_253, torch.float32);  view_253 = None
        mul_131: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_397, 0.5)
        mul_132: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_397, 0.7071067811865476);  convert_element_type_397 = None
        erf_17: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_113: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_133: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_131, add_113);  mul_131 = add_113 = None
        convert_element_type_398: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_133, torch.float16);  mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_254: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_398, [1500, 4096]);  convert_element_type_398 = None
        permute_160: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        addmm_79: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg245_1, view_254, permute_160);  arg245_1 = view_254 = permute_160 = None
        view_255: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_79, [1, 1500, 1024]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_114: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_110, view_255);  add_110 = view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_402: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32);  add_114 = None
        clamp_min_15: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_402, -64504.0);  convert_element_type_402 = None
        clamp_max_15: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_15, 64504.0);  clamp_min_15 = None
        convert_element_type_403: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_15, torch.float16);  clamp_max_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_145: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_403, memory_format = torch.contiguous_format)
        convert_element_type_404: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_145, torch.float32);  clone_145 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_404, [2], correction = 0, keepdim = True)
        getitem_208: "f32[1, 1500, 1]" = var_mean_32[0]
        getitem_209: "f32[1, 1500, 1]" = var_mean_32[1];  var_mean_32 = None
        sub_32: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_404, getitem_209);  convert_element_type_404 = getitem_209 = None
        add_115: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_32: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_134: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
        mul_135: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_134, arg246_1);  mul_134 = arg246_1 = None
        add_116: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_135, arg247_1);  mul_135 = arg247_1 = None
        convert_element_type_405: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_116, torch.float16);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_256: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_405, [1500, 1024])
        permute_161: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        addmm_80: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg249_1, view_256, permute_161);  arg249_1 = view_256 = permute_161 = None
        view_257: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_80, [1, 1500, 1024]);  addmm_80 = None
        mul_136: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_257, 0.125);  view_257 = None
        view_258: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_136, [1, 1500, -1, 64]);  mul_136 = None
        permute_162: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        clone_146: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_259: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_405, [1500, 1024])
        permute_163: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        mm_16: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_259, permute_163);  view_259 = permute_163 = None
        view_260: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_16, [1, 1500, 1024]);  mm_16 = None
        view_261: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_260, [1, -1, 16, 64]);  view_260 = None
        permute_164: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        clone_147: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_262: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_405, [1500, 1024]);  convert_element_type_405 = None
        permute_165: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_81: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg252_1, view_262, permute_165);  arg252_1 = view_262 = permute_165 = None
        view_263: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_81, [1, 1500, 1024]);  addmm_81 = None
        view_264: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_263, [1, -1, 16, 64]);  view_263 = None
        permute_166: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
        clone_148: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_146, clone_147, clone_148, None, False, scale = 1.0);  clone_146 = clone_147 = clone_148 = None
        getitem_210: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_16[0];  _scaled_dot_product_cudnn_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_167: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_210, [0, 2, 1, 3]);  getitem_210 = None
        clone_149: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_265: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_149, [1, 1500, -1]);  clone_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_266: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_265, [1500, 1024]);  view_265 = None
        permute_168: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_82: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg254_1, view_266, permute_168);  arg254_1 = view_266 = permute_168 = None
        view_267: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_82, [1, 1500, 1024]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_117: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_403, view_267);  convert_element_type_403 = view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_151: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_117, memory_format = torch.contiguous_format)
        convert_element_type_417: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_151, torch.float32);  clone_151 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_417, [2], correction = 0, keepdim = True)
        getitem_219: "f32[1, 1500, 1]" = var_mean_33[0]
        getitem_220: "f32[1, 1500, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_33: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_417, getitem_220);  convert_element_type_417 = getitem_220 = None
        add_118: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_219, 1e-05);  getitem_219 = None
        rsqrt_33: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_137: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
        mul_138: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_137, arg255_1);  mul_137 = arg255_1 = None
        add_119: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_138, arg256_1);  mul_138 = arg256_1 = None
        convert_element_type_418: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_119, torch.float16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_268: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_418, [1500, 1024]);  convert_element_type_418 = None
        permute_169: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_83: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg258_1, view_268, permute_169);  arg258_1 = view_268 = permute_169 = None
        view_269: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_83, [1, 1500, 4096]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_422: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_269, torch.float32);  view_269 = None
        mul_139: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_422, 0.5)
        mul_140: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_422, 0.7071067811865476);  convert_element_type_422 = None
        erf_18: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_120: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_141: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_139, add_120);  mul_139 = add_120 = None
        convert_element_type_423: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_141, torch.float16);  mul_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_270: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_423, [1500, 4096]);  convert_element_type_423 = None
        permute_170: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_84: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg260_1, view_270, permute_170);  arg260_1 = view_270 = permute_170 = None
        view_271: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_84, [1, 1500, 1024]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_121: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_117, view_271);  add_117 = view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_427: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_121, torch.float32);  add_121 = None
        clamp_min_16: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_427, -64504.0);  convert_element_type_427 = None
        clamp_max_16: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_16, 64504.0);  clamp_min_16 = None
        convert_element_type_428: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_16, torch.float16);  clamp_max_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_154: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_428, memory_format = torch.contiguous_format)
        convert_element_type_429: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_154, torch.float32);  clone_154 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_429, [2], correction = 0, keepdim = True)
        getitem_221: "f32[1, 1500, 1]" = var_mean_34[0]
        getitem_222: "f32[1, 1500, 1]" = var_mean_34[1];  var_mean_34 = None
        sub_34: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_429, getitem_222);  convert_element_type_429 = getitem_222 = None
        add_122: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_221, 1e-05);  getitem_221 = None
        rsqrt_34: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_142: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
        mul_143: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_142, arg261_1);  mul_142 = arg261_1 = None
        add_123: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_143, arg262_1);  mul_143 = arg262_1 = None
        convert_element_type_430: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_123, torch.float16);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_272: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_430, [1500, 1024])
        permute_171: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_85: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg264_1, view_272, permute_171);  arg264_1 = view_272 = permute_171 = None
        view_273: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_85, [1, 1500, 1024]);  addmm_85 = None
        mul_144: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_273, 0.125);  view_273 = None
        view_274: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_144, [1, 1500, -1, 64]);  mul_144 = None
        permute_172: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        clone_155: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_275: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_430, [1500, 1024])
        permute_173: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        mm_17: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_275, permute_173);  view_275 = permute_173 = None
        view_276: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_17, [1, 1500, 1024]);  mm_17 = None
        view_277: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_276, [1, -1, 16, 64]);  view_276 = None
        permute_174: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
        clone_156: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_278: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_430, [1500, 1024]);  convert_element_type_430 = None
        permute_175: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_86: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg267_1, view_278, permute_175);  arg267_1 = view_278 = permute_175 = None
        view_279: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_86, [1, 1500, 1024]);  addmm_86 = None
        view_280: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_279, [1, -1, 16, 64]);  view_279 = None
        permute_176: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
        clone_157: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_155, clone_156, clone_157, None, False, scale = 1.0);  clone_155 = clone_156 = clone_157 = None
        getitem_223: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_17[0];  _scaled_dot_product_cudnn_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_177: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_223, [0, 2, 1, 3]);  getitem_223 = None
        clone_158: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_281: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_158, [1, 1500, -1]);  clone_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_282: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_281, [1500, 1024]);  view_281 = None
        permute_178: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_87: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg269_1, view_282, permute_178);  arg269_1 = view_282 = permute_178 = None
        view_283: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_87, [1, 1500, 1024]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_124: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_428, view_283);  convert_element_type_428 = view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_160: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_124, memory_format = torch.contiguous_format)
        convert_element_type_442: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_160, torch.float32);  clone_160 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_442, [2], correction = 0, keepdim = True)
        getitem_232: "f32[1, 1500, 1]" = var_mean_35[0]
        getitem_233: "f32[1, 1500, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_35: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_442, getitem_233);  convert_element_type_442 = getitem_233 = None
        add_125: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_232, 1e-05);  getitem_232 = None
        rsqrt_35: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_145: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
        mul_146: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_145, arg270_1);  mul_145 = arg270_1 = None
        add_126: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_146, arg271_1);  mul_146 = arg271_1 = None
        convert_element_type_443: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_126, torch.float16);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_284: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_443, [1500, 1024]);  convert_element_type_443 = None
        permute_179: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        addmm_88: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg273_1, view_284, permute_179);  arg273_1 = view_284 = permute_179 = None
        view_285: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_88, [1, 1500, 4096]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_447: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_285, torch.float32);  view_285 = None
        mul_147: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_447, 0.5)
        mul_148: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_447, 0.7071067811865476);  convert_element_type_447 = None
        erf_19: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_127: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_149: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_147, add_127);  mul_147 = add_127 = None
        convert_element_type_448: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_149, torch.float16);  mul_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_286: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_448, [1500, 4096]);  convert_element_type_448 = None
        permute_180: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_89: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg275_1, view_286, permute_180);  arg275_1 = view_286 = permute_180 = None
        view_287: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_89, [1, 1500, 1024]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_128: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_124, view_287);  add_124 = view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_452: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_128, torch.float32);  add_128 = None
        clamp_min_17: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_452, -64504.0);  convert_element_type_452 = None
        clamp_max_17: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_17, 64504.0);  clamp_min_17 = None
        convert_element_type_453: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_17, torch.float16);  clamp_max_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_163: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_453, memory_format = torch.contiguous_format)
        convert_element_type_454: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_163, torch.float32);  clone_163 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_454, [2], correction = 0, keepdim = True)
        getitem_234: "f32[1, 1500, 1]" = var_mean_36[0]
        getitem_235: "f32[1, 1500, 1]" = var_mean_36[1];  var_mean_36 = None
        sub_36: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_454, getitem_235);  convert_element_type_454 = getitem_235 = None
        add_129: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_234, 1e-05);  getitem_234 = None
        rsqrt_36: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_150: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
        mul_151: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_150, arg276_1);  mul_150 = arg276_1 = None
        add_130: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_151, arg277_1);  mul_151 = arg277_1 = None
        convert_element_type_455: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_130, torch.float16);  add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_288: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_455, [1500, 1024])
        permute_181: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_90: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg279_1, view_288, permute_181);  arg279_1 = view_288 = permute_181 = None
        view_289: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_90, [1, 1500, 1024]);  addmm_90 = None
        mul_152: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_289, 0.125);  view_289 = None
        view_290: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_152, [1, 1500, -1, 64]);  mul_152 = None
        permute_182: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        clone_164: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_291: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_455, [1500, 1024])
        permute_183: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        mm_18: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_291, permute_183);  view_291 = permute_183 = None
        view_292: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_18, [1, 1500, 1024]);  mm_18 = None
        view_293: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_292, [1, -1, 16, 64]);  view_292 = None
        permute_184: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
        clone_165: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_294: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_455, [1500, 1024]);  convert_element_type_455 = None
        permute_185: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_91: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg282_1, view_294, permute_185);  arg282_1 = view_294 = permute_185 = None
        view_295: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_91, [1, 1500, 1024]);  addmm_91 = None
        view_296: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_295, [1, -1, 16, 64]);  view_295 = None
        permute_186: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        clone_166: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_164, clone_165, clone_166, None, False, scale = 1.0);  clone_164 = clone_165 = clone_166 = None
        getitem_236: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_18[0];  _scaled_dot_product_cudnn_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_187: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_236, [0, 2, 1, 3]);  getitem_236 = None
        clone_167: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_297: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_167, [1, 1500, -1]);  clone_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_298: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_297, [1500, 1024]);  view_297 = None
        permute_188: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_92: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg284_1, view_298, permute_188);  arg284_1 = view_298 = permute_188 = None
        view_299: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_92, [1, 1500, 1024]);  addmm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_131: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_453, view_299);  convert_element_type_453 = view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_169: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_131, memory_format = torch.contiguous_format)
        convert_element_type_467: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_169, torch.float32);  clone_169 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_467, [2], correction = 0, keepdim = True)
        getitem_245: "f32[1, 1500, 1]" = var_mean_37[0]
        getitem_246: "f32[1, 1500, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_37: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_467, getitem_246);  convert_element_type_467 = getitem_246 = None
        add_132: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_245, 1e-05);  getitem_245 = None
        rsqrt_37: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        mul_153: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
        mul_154: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_153, arg285_1);  mul_153 = arg285_1 = None
        add_133: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_154, arg286_1);  mul_154 = arg286_1 = None
        convert_element_type_468: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_133, torch.float16);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_300: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_468, [1500, 1024]);  convert_element_type_468 = None
        permute_189: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_93: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg288_1, view_300, permute_189);  arg288_1 = view_300 = permute_189 = None
        view_301: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_93, [1, 1500, 4096]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_472: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_301, torch.float32);  view_301 = None
        mul_155: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_472, 0.5)
        mul_156: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_472, 0.7071067811865476);  convert_element_type_472 = None
        erf_20: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_156);  mul_156 = None
        add_134: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_157: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_155, add_134);  mul_155 = add_134 = None
        convert_element_type_473: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_157, torch.float16);  mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_302: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_473, [1500, 4096]);  convert_element_type_473 = None
        permute_190: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_94: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg290_1, view_302, permute_190);  arg290_1 = view_302 = permute_190 = None
        view_303: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_94, [1, 1500, 1024]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_135: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_131, view_303);  add_131 = view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_477: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_135, torch.float32);  add_135 = None
        clamp_min_18: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_477, -64504.0);  convert_element_type_477 = None
        clamp_max_18: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_18, 64504.0);  clamp_min_18 = None
        convert_element_type_478: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_18, torch.float16);  clamp_max_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_172: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_478, memory_format = torch.contiguous_format)
        convert_element_type_479: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_172, torch.float32);  clone_172 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_479, [2], correction = 0, keepdim = True)
        getitem_247: "f32[1, 1500, 1]" = var_mean_38[0]
        getitem_248: "f32[1, 1500, 1]" = var_mean_38[1];  var_mean_38 = None
        sub_38: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_479, getitem_248);  convert_element_type_479 = getitem_248 = None
        add_136: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_247, 1e-05);  getitem_247 = None
        rsqrt_38: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_158: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
        mul_159: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_158, arg291_1);  mul_158 = arg291_1 = None
        add_137: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_159, arg292_1);  mul_159 = arg292_1 = None
        convert_element_type_480: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_137, torch.float16);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_304: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_480, [1500, 1024])
        permute_191: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_95: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg294_1, view_304, permute_191);  arg294_1 = view_304 = permute_191 = None
        view_305: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_95, [1, 1500, 1024]);  addmm_95 = None
        mul_160: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_305, 0.125);  view_305 = None
        view_306: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_160, [1, 1500, -1, 64]);  mul_160 = None
        permute_192: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
        clone_173: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_307: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_480, [1500, 1024])
        permute_193: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        mm_19: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_307, permute_193);  view_307 = permute_193 = None
        view_308: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_19, [1, 1500, 1024]);  mm_19 = None
        view_309: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_308, [1, -1, 16, 64]);  view_308 = None
        permute_194: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None
        clone_174: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_310: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_480, [1500, 1024]);  convert_element_type_480 = None
        permute_195: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_96: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg297_1, view_310, permute_195);  arg297_1 = view_310 = permute_195 = None
        view_311: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_96, [1, 1500, 1024]);  addmm_96 = None
        view_312: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_311, [1, -1, 16, 64]);  view_311 = None
        permute_196: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
        clone_175: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_173, clone_174, clone_175, None, False, scale = 1.0);  clone_173 = clone_174 = clone_175 = None
        getitem_249: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_19[0];  _scaled_dot_product_cudnn_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_197: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_249, [0, 2, 1, 3]);  getitem_249 = None
        clone_176: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_313: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_176, [1, 1500, -1]);  clone_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_314: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_313, [1500, 1024]);  view_313 = None
        permute_198: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_97: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg299_1, view_314, permute_198);  arg299_1 = view_314 = permute_198 = None
        view_315: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_97, [1, 1500, 1024]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_138: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_478, view_315);  convert_element_type_478 = view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_178: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_138, memory_format = torch.contiguous_format)
        convert_element_type_492: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_178, torch.float32);  clone_178 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_492, [2], correction = 0, keepdim = True)
        getitem_258: "f32[1, 1500, 1]" = var_mean_39[0]
        getitem_259: "f32[1, 1500, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_39: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_492, getitem_259);  convert_element_type_492 = getitem_259 = None
        add_139: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_258, 1e-05);  getitem_258 = None
        rsqrt_39: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_161: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
        mul_162: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_161, arg300_1);  mul_161 = arg300_1 = None
        add_140: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_162, arg301_1);  mul_162 = arg301_1 = None
        convert_element_type_493: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_140, torch.float16);  add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_316: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_493, [1500, 1024]);  convert_element_type_493 = None
        permute_199: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
        addmm_98: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg303_1, view_316, permute_199);  arg303_1 = view_316 = permute_199 = None
        view_317: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_98, [1, 1500, 4096]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_497: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_317, torch.float32);  view_317 = None
        mul_163: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_497, 0.5)
        mul_164: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_497, 0.7071067811865476);  convert_element_type_497 = None
        erf_21: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
        add_141: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_165: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_163, add_141);  mul_163 = add_141 = None
        convert_element_type_498: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_165, torch.float16);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_318: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_498, [1500, 4096]);  convert_element_type_498 = None
        permute_200: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
        addmm_99: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg305_1, view_318, permute_200);  arg305_1 = view_318 = permute_200 = None
        view_319: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_99, [1, 1500, 1024]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_142: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_138, view_319);  add_138 = view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_502: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_142, torch.float32);  add_142 = None
        clamp_min_19: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_502, -64504.0);  convert_element_type_502 = None
        clamp_max_19: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_19, 64504.0);  clamp_min_19 = None
        convert_element_type_503: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_19, torch.float16);  clamp_max_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_181: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_503, memory_format = torch.contiguous_format)
        convert_element_type_504: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_181, torch.float32);  clone_181 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_504, [2], correction = 0, keepdim = True)
        getitem_260: "f32[1, 1500, 1]" = var_mean_40[0]
        getitem_261: "f32[1, 1500, 1]" = var_mean_40[1];  var_mean_40 = None
        sub_40: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_504, getitem_261);  convert_element_type_504 = getitem_261 = None
        add_143: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_260, 1e-05);  getitem_260 = None
        rsqrt_40: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        mul_166: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
        mul_167: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_166, arg306_1);  mul_166 = arg306_1 = None
        add_144: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_167, arg307_1);  mul_167 = arg307_1 = None
        convert_element_type_505: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_144, torch.float16);  add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_320: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_505, [1500, 1024])
        permute_201: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
        addmm_100: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg309_1, view_320, permute_201);  arg309_1 = view_320 = permute_201 = None
        view_321: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_100, [1, 1500, 1024]);  addmm_100 = None
        mul_168: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_321, 0.125);  view_321 = None
        view_322: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_168, [1, 1500, -1, 64]);  mul_168 = None
        permute_202: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
        clone_182: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_323: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_505, [1500, 1024])
        permute_203: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        mm_20: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_323, permute_203);  view_323 = permute_203 = None
        view_324: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_20, [1, 1500, 1024]);  mm_20 = None
        view_325: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_324, [1, -1, 16, 64]);  view_324 = None
        permute_204: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
        clone_183: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_326: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_505, [1500, 1024]);  convert_element_type_505 = None
        permute_205: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_101: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg312_1, view_326, permute_205);  arg312_1 = view_326 = permute_205 = None
        view_327: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_101, [1, 1500, 1024]);  addmm_101 = None
        view_328: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_327, [1, -1, 16, 64]);  view_327 = None
        permute_206: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        clone_184: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_182, clone_183, clone_184, None, False, scale = 1.0);  clone_182 = clone_183 = clone_184 = None
        getitem_262: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_20[0];  _scaled_dot_product_cudnn_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_207: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_262, [0, 2, 1, 3]);  getitem_262 = None
        clone_185: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_329: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_185, [1, 1500, -1]);  clone_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_330: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_329, [1500, 1024]);  view_329 = None
        permute_208: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_102: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg314_1, view_330, permute_208);  arg314_1 = view_330 = permute_208 = None
        view_331: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_102, [1, 1500, 1024]);  addmm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_145: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_503, view_331);  convert_element_type_503 = view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_187: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_145, memory_format = torch.contiguous_format)
        convert_element_type_517: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_187, torch.float32);  clone_187 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_517, [2], correction = 0, keepdim = True)
        getitem_271: "f32[1, 1500, 1]" = var_mean_41[0]
        getitem_272: "f32[1, 1500, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_41: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_517, getitem_272);  convert_element_type_517 = getitem_272 = None
        add_146: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_271, 1e-05);  getitem_271 = None
        rsqrt_41: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        mul_169: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_170: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_169, arg315_1);  mul_169 = arg315_1 = None
        add_147: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_170, arg316_1);  mul_170 = arg316_1 = None
        convert_element_type_518: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_147, torch.float16);  add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_332: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_518, [1500, 1024]);  convert_element_type_518 = None
        permute_209: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        addmm_103: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg318_1, view_332, permute_209);  arg318_1 = view_332 = permute_209 = None
        view_333: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_103, [1, 1500, 4096]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_522: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_333, torch.float32);  view_333 = None
        mul_171: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_522, 0.5)
        mul_172: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_522, 0.7071067811865476);  convert_element_type_522 = None
        erf_22: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
        add_148: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_173: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_171, add_148);  mul_171 = add_148 = None
        convert_element_type_523: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_173, torch.float16);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_334: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_523, [1500, 4096]);  convert_element_type_523 = None
        permute_210: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_104: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg320_1, view_334, permute_210);  arg320_1 = view_334 = permute_210 = None
        view_335: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_104, [1, 1500, 1024]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_149: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_145, view_335);  add_145 = view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_527: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_149, torch.float32);  add_149 = None
        clamp_min_20: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_527, -64504.0);  convert_element_type_527 = None
        clamp_max_20: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_20, 64504.0);  clamp_min_20 = None
        convert_element_type_528: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_20, torch.float16);  clamp_max_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_190: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_528, memory_format = torch.contiguous_format)
        convert_element_type_529: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_190, torch.float32);  clone_190 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_529, [2], correction = 0, keepdim = True)
        getitem_273: "f32[1, 1500, 1]" = var_mean_42[0]
        getitem_274: "f32[1, 1500, 1]" = var_mean_42[1];  var_mean_42 = None
        sub_42: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_529, getitem_274);  convert_element_type_529 = getitem_274 = None
        add_150: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_273, 1e-05);  getitem_273 = None
        rsqrt_42: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        mul_174: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_175: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_174, arg321_1);  mul_174 = arg321_1 = None
        add_151: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_175, arg322_1);  mul_175 = arg322_1 = None
        convert_element_type_530: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_151, torch.float16);  add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_336: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_530, [1500, 1024])
        permute_211: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        addmm_105: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg324_1, view_336, permute_211);  arg324_1 = view_336 = permute_211 = None
        view_337: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_105, [1, 1500, 1024]);  addmm_105 = None
        mul_176: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_337, 0.125);  view_337 = None
        view_338: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_176, [1, 1500, -1, 64]);  mul_176 = None
        permute_212: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        clone_191: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_339: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_530, [1500, 1024])
        permute_213: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
        mm_21: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_339, permute_213);  view_339 = permute_213 = None
        view_340: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_21, [1, 1500, 1024]);  mm_21 = None
        view_341: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_340, [1, -1, 16, 64]);  view_340 = None
        permute_214: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
        clone_192: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_342: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_530, [1500, 1024]);  convert_element_type_530 = None
        permute_215: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_106: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg327_1, view_342, permute_215);  arg327_1 = view_342 = permute_215 = None
        view_343: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_106, [1, 1500, 1024]);  addmm_106 = None
        view_344: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_343, [1, -1, 16, 64]);  view_343 = None
        permute_216: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_193: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_191, clone_192, clone_193, None, False, scale = 1.0);  clone_191 = clone_192 = clone_193 = None
        getitem_275: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_21[0];  _scaled_dot_product_cudnn_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_217: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_275, [0, 2, 1, 3]);  getitem_275 = None
        clone_194: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_345: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_194, [1, 1500, -1]);  clone_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_346: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_345, [1500, 1024]);  view_345 = None
        permute_218: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_107: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg329_1, view_346, permute_218);  arg329_1 = view_346 = permute_218 = None
        view_347: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_107, [1, 1500, 1024]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_152: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_528, view_347);  convert_element_type_528 = view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_196: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_152, memory_format = torch.contiguous_format)
        convert_element_type_542: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_196, torch.float32);  clone_196 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_542, [2], correction = 0, keepdim = True)
        getitem_284: "f32[1, 1500, 1]" = var_mean_43[0]
        getitem_285: "f32[1, 1500, 1]" = var_mean_43[1];  var_mean_43 = None
        sub_43: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_542, getitem_285);  convert_element_type_542 = getitem_285 = None
        add_153: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_284, 1e-05);  getitem_284 = None
        rsqrt_43: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        mul_177: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_178: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_177, arg330_1);  mul_177 = arg330_1 = None
        add_154: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_178, arg331_1);  mul_178 = arg331_1 = None
        convert_element_type_543: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_154, torch.float16);  add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_348: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_543, [1500, 1024]);  convert_element_type_543 = None
        permute_219: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
        addmm_108: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg333_1, view_348, permute_219);  arg333_1 = view_348 = permute_219 = None
        view_349: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_108, [1, 1500, 4096]);  addmm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_547: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        mul_179: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_547, 0.5)
        mul_180: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_547, 0.7071067811865476);  convert_element_type_547 = None
        erf_23: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_180);  mul_180 = None
        add_155: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_181: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_179, add_155);  mul_179 = add_155 = None
        convert_element_type_548: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_181, torch.float16);  mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_350: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_548, [1500, 4096]);  convert_element_type_548 = None
        permute_220: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
        addmm_109: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg335_1, view_350, permute_220);  arg335_1 = view_350 = permute_220 = None
        view_351: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_109, [1, 1500, 1024]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_156: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_152, view_351);  add_152 = view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_552: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_156, torch.float32);  add_156 = None
        clamp_min_21: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_552, -64504.0);  convert_element_type_552 = None
        clamp_max_21: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_21, 64504.0);  clamp_min_21 = None
        convert_element_type_553: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_21, torch.float16);  clamp_max_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_199: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_553, memory_format = torch.contiguous_format)
        convert_element_type_554: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_199, torch.float32);  clone_199 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_554, [2], correction = 0, keepdim = True)
        getitem_286: "f32[1, 1500, 1]" = var_mean_44[0]
        getitem_287: "f32[1, 1500, 1]" = var_mean_44[1];  var_mean_44 = None
        sub_44: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_554, getitem_287);  convert_element_type_554 = getitem_287 = None
        add_157: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_286, 1e-05);  getitem_286 = None
        rsqrt_44: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        mul_182: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_183: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_182, arg336_1);  mul_182 = arg336_1 = None
        add_158: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_183, arg337_1);  mul_183 = arg337_1 = None
        convert_element_type_555: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_158, torch.float16);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_352: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_555, [1500, 1024])
        permute_221: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
        addmm_110: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg339_1, view_352, permute_221);  arg339_1 = view_352 = permute_221 = None
        view_353: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_110, [1, 1500, 1024]);  addmm_110 = None
        mul_184: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_353, 0.125);  view_353 = None
        view_354: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_184, [1, 1500, -1, 64]);  mul_184 = None
        permute_222: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        clone_200: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_355: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_555, [1500, 1024])
        permute_223: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
        mm_22: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_355, permute_223);  view_355 = permute_223 = None
        view_356: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_22, [1, 1500, 1024]);  mm_22 = None
        view_357: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_356, [1, -1, 16, 64]);  view_356 = None
        permute_224: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
        clone_201: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_358: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_555, [1500, 1024]);  convert_element_type_555 = None
        permute_225: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_111: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg342_1, view_358, permute_225);  arg342_1 = view_358 = permute_225 = None
        view_359: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_111, [1, 1500, 1024]);  addmm_111 = None
        view_360: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_359, [1, -1, 16, 64]);  view_359 = None
        permute_226: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        clone_202: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_200, clone_201, clone_202, None, False, scale = 1.0);  clone_200 = clone_201 = clone_202 = None
        getitem_288: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_22[0];  _scaled_dot_product_cudnn_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_227: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_288, [0, 2, 1, 3]);  getitem_288 = None
        clone_203: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_361: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_203, [1, 1500, -1]);  clone_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_362: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_361, [1500, 1024]);  view_361 = None
        permute_228: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_112: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg344_1, view_362, permute_228);  arg344_1 = view_362 = permute_228 = None
        view_363: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_112, [1, 1500, 1024]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_159: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_553, view_363);  convert_element_type_553 = view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_205: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_159, memory_format = torch.contiguous_format)
        convert_element_type_567: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_205, torch.float32);  clone_205 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_567, [2], correction = 0, keepdim = True)
        getitem_297: "f32[1, 1500, 1]" = var_mean_45[0]
        getitem_298: "f32[1, 1500, 1]" = var_mean_45[1];  var_mean_45 = None
        sub_45: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_567, getitem_298);  convert_element_type_567 = getitem_298 = None
        add_160: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_297, 1e-05);  getitem_297 = None
        rsqrt_45: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        mul_185: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_186: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_185, arg345_1);  mul_185 = arg345_1 = None
        add_161: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_186, arg346_1);  mul_186 = arg346_1 = None
        convert_element_type_568: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_161, torch.float16);  add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_364: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_568, [1500, 1024]);  convert_element_type_568 = None
        permute_229: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_113: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg348_1, view_364, permute_229);  arg348_1 = view_364 = permute_229 = None
        view_365: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_113, [1, 1500, 4096]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_572: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_365, torch.float32);  view_365 = None
        mul_187: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_572, 0.5)
        mul_188: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_572, 0.7071067811865476);  convert_element_type_572 = None
        erf_24: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
        add_162: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_189: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_187, add_162);  mul_187 = add_162 = None
        convert_element_type_573: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_189, torch.float16);  mul_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_366: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_573, [1500, 4096]);  convert_element_type_573 = None
        permute_230: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_114: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg350_1, view_366, permute_230);  arg350_1 = view_366 = permute_230 = None
        view_367: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_114, [1, 1500, 1024]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_163: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_159, view_367);  add_159 = view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_577: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_163, torch.float32);  add_163 = None
        clamp_min_22: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_577, -64504.0);  convert_element_type_577 = None
        clamp_max_22: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_22, 64504.0);  clamp_min_22 = None
        convert_element_type_578: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_22, torch.float16);  clamp_max_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_208: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_578, memory_format = torch.contiguous_format)
        convert_element_type_579: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_208, torch.float32);  clone_208 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_579, [2], correction = 0, keepdim = True)
        getitem_299: "f32[1, 1500, 1]" = var_mean_46[0]
        getitem_300: "f32[1, 1500, 1]" = var_mean_46[1];  var_mean_46 = None
        sub_46: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_579, getitem_300);  convert_element_type_579 = getitem_300 = None
        add_164: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_299, 1e-05);  getitem_299 = None
        rsqrt_46: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        mul_190: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_191: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_190, arg351_1);  mul_190 = arg351_1 = None
        add_165: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_191, arg352_1);  mul_191 = arg352_1 = None
        convert_element_type_580: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_165, torch.float16);  add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_368: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_580, [1500, 1024])
        permute_231: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_115: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg354_1, view_368, permute_231);  arg354_1 = view_368 = permute_231 = None
        view_369: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_115, [1, 1500, 1024]);  addmm_115 = None
        mul_192: "f16[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(view_369, 0.125);  view_369 = None
        view_370: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(mul_192, [1, 1500, -1, 64]);  mul_192 = None
        permute_232: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
        clone_209: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_371: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_580, [1500, 1024])
        permute_233: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
        mm_23: "f16[1500, 1024]" = torch.ops.aten.mm.default(view_371, permute_233);  view_371 = permute_233 = None
        view_372: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(mm_23, [1, 1500, 1024]);  mm_23 = None
        view_373: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_372, [1, -1, 16, 64]);  view_372 = None
        permute_234: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
        clone_210: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_374: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_580, [1500, 1024]);  convert_element_type_580 = None
        permute_235: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
        addmm_116: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg357_1, view_374, permute_235);  arg357_1 = view_374 = permute_235 = None
        view_375: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_116, [1, 1500, 1024]);  addmm_116 = None
        view_376: "f16[1, 1500, 16, 64]" = torch.ops.aten.reshape.default(view_375, [1, -1, 16, 64]);  view_375 = None
        permute_236: "f16[1, 16, 1500, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        clone_211: "f16[1, 16, 1500, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_209, clone_210, clone_211, None, False, scale = 1.0);  clone_209 = clone_210 = clone_211 = None
        getitem_301: "f16[1, 16, 1500, 64]" = _scaled_dot_product_cudnn_attention_23[0];  _scaled_dot_product_cudnn_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_237: "f16[1, 1500, 16, 64]" = torch.ops.aten.permute.default(getitem_301, [0, 2, 1, 3]);  getitem_301 = None
        clone_212: "f16[1, 1500, 16, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_377: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(clone_212, [1, 1500, -1]);  clone_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_378: "f16[1500, 1024]" = torch.ops.aten.reshape.default(view_377, [1500, 1024]);  view_377 = None
        permute_238: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
        addmm_117: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg359_1, view_378, permute_238);  arg359_1 = view_378 = permute_238 = None
        view_379: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_117, [1, 1500, 1024]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_166: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_578, view_379);  convert_element_type_578 = view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_214: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_166, memory_format = torch.contiguous_format)
        convert_element_type_592: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_214, torch.float32);  clone_214 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_592, [2], correction = 0, keepdim = True)
        getitem_310: "f32[1, 1500, 1]" = var_mean_47[0]
        getitem_311: "f32[1, 1500, 1]" = var_mean_47[1];  var_mean_47 = None
        sub_47: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_592, getitem_311);  convert_element_type_592 = getitem_311 = None
        add_167: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-05);  getitem_310 = None
        rsqrt_47: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
        mul_193: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_194: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_193, arg360_1);  mul_193 = arg360_1 = None
        add_168: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_194, arg361_1);  mul_194 = arg361_1 = None
        convert_element_type_593: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_168, torch.float16);  add_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_380: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_593, [1500, 1024]);  convert_element_type_593 = None
        permute_239: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
        addmm_118: "f16[1500, 4096]" = torch.ops.aten.addmm.default(arg363_1, view_380, permute_239);  arg363_1 = view_380 = permute_239 = None
        view_381: "f16[1, 1500, 4096]" = torch.ops.aten.reshape.default(addmm_118, [1, 1500, 4096]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_597: "f32[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(view_381, torch.float32);  view_381 = None
        mul_195: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_597, 0.5)
        mul_196: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_597, 0.7071067811865476);  convert_element_type_597 = None
        erf_25: "f32[1, 1500, 4096]" = torch.ops.aten.erf.default(mul_196);  mul_196 = None
        add_169: "f32[1, 1500, 4096]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_197: "f32[1, 1500, 4096]" = torch.ops.aten.mul.Tensor(mul_195, add_169);  mul_195 = add_169 = None
        convert_element_type_598: "f16[1, 1500, 4096]" = torch.ops.prims.convert_element_type.default(mul_197, torch.float16);  mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_382: "f16[1500, 4096]" = torch.ops.aten.reshape.default(convert_element_type_598, [1500, 4096]);  convert_element_type_598 = None
        permute_240: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg364_1, [1, 0]);  arg364_1 = None
        addmm_119: "f16[1500, 1024]" = torch.ops.aten.addmm.default(arg365_1, view_382, permute_240);  arg365_1 = view_382 = permute_240 = None
        view_383: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_119, [1, 1500, 1024]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_170: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_166, view_383);  add_166 = view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_602: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_170, torch.float32);  add_170 = None
        clamp_min_23: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_602, -64504.0);  convert_element_type_602 = None
        clamp_max_23: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_23, 64504.0);  clamp_min_23 = None
        convert_element_type_603: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_23, torch.float16);  clamp_max_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:642 in forward, code: hidden_states = self.layer_norm(hidden_states)
        clone_217: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_603, memory_format = torch.contiguous_format);  convert_element_type_603 = None
        convert_element_type_604: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_217, torch.float32);  clone_217 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_604, [2], correction = 0, keepdim = True)
        getitem_312: "f32[1, 1500, 1]" = var_mean_48[0]
        getitem_313: "f32[1, 1500, 1]" = var_mean_48[1];  var_mean_48 = None
        sub_48: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_604, getitem_313);  convert_element_type_604 = getitem_313 = None
        add_171: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_312, 1e-05);  getitem_312 = None
        rsqrt_48: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        mul_198: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_199: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_198, arg366_1);  mul_198 = arg366_1 = None
        add_172: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_199, arg367_1);  mul_199 = arg367_1 = None
        convert_element_type_605: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_172, torch.float16);  add_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1333 in forward, code: hidden_states = self.projector(hidden_states)
        view_384: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_605, [1500, 1024]);  convert_element_type_605 = None
        permute_241: "f16[1024, 256]" = torch.ops.aten.permute.default(arg368_1, [1, 0]);  arg368_1 = None
        addmm_120: "f16[1500, 256]" = torch.ops.aten.addmm.default(arg369_1, view_384, permute_241);  arg369_1 = view_384 = permute_241 = None
        view_385: "f16[1, 1500, 256]" = torch.ops.aten.reshape.default(addmm_120, [1, 1500, 256]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1334 in forward, code: pooled_output = hidden_states.mean(dim=1)
        mean: "f16[1, 256]" = torch.ops.aten.mean.dim(view_385, [1]);  view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1336 in forward, code: logits = self.classifier(pooled_output)
        permute_242: "f16[256, 2]" = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
        addmm_121: "f16[1, 2]" = torch.ops.aten.addmm.default(arg371_1, mean, permute_242);  arg371_1 = mean = permute_242 = None
        return (addmm_121,)
