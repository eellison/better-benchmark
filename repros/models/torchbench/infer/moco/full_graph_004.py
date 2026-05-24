import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 128]", arg1_1: "f32[64, 3, 7, 7]", arg2_1: "f32[64, 3, 7, 7]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64]", arg7_1: "f32[64, 64, 1, 1]", arg8_1: "f32[64, 64, 1, 1]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[64]", arg13_1: "f32[64, 64, 3, 3]", arg14_1: "f32[64, 64, 3, 3]", arg15_1: "f32[64]", arg16_1: "f32[64]", arg17_1: "f32[64]", arg18_1: "f32[64]", arg19_1: "f32[256, 64, 1, 1]", arg20_1: "f32[256, 64, 1, 1]", arg21_1: "f32[256]", arg22_1: "f32[256]", arg23_1: "f32[256]", arg24_1: "f32[256]", arg25_1: "f32[256, 64, 1, 1]", arg26_1: "f32[256, 64, 1, 1]", arg27_1: "f32[256]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[256]", arg31_1: "f32[64, 256, 1, 1]", arg32_1: "f32[64, 256, 1, 1]", arg33_1: "f32[64]", arg34_1: "f32[64]", arg35_1: "f32[64]", arg36_1: "f32[64]", arg37_1: "f32[64, 64, 3, 3]", arg38_1: "f32[64, 64, 3, 3]", arg39_1: "f32[64]", arg40_1: "f32[64]", arg41_1: "f32[64]", arg42_1: "f32[64]", arg43_1: "f32[256, 64, 1, 1]", arg44_1: "f32[256, 64, 1, 1]", arg45_1: "f32[256]", arg46_1: "f32[256]", arg47_1: "f32[256]", arg48_1: "f32[256]", arg49_1: "f32[64, 256, 1, 1]", arg50_1: "f32[64, 256, 1, 1]", arg51_1: "f32[64]", arg52_1: "f32[64]", arg53_1: "f32[64]", arg54_1: "f32[64]", arg55_1: "f32[64, 64, 3, 3]", arg56_1: "f32[64, 64, 3, 3]", arg57_1: "f32[64]", arg58_1: "f32[64]", arg59_1: "f32[64]", arg60_1: "f32[64]", arg61_1: "f32[256, 64, 1, 1]", arg62_1: "f32[256, 64, 1, 1]", arg63_1: "f32[256]", arg64_1: "f32[256]", arg65_1: "f32[256]", arg66_1: "f32[256]", arg67_1: "f32[128, 256, 1, 1]", arg68_1: "f32[128, 256, 1, 1]", arg69_1: "f32[128]", arg70_1: "f32[128]", arg71_1: "f32[128]", arg72_1: "f32[128]", arg73_1: "f32[128, 128, 3, 3]", arg74_1: "f32[128, 128, 3, 3]", arg75_1: "f32[128]", arg76_1: "f32[128]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[512, 128, 1, 1]", arg80_1: "f32[512, 128, 1, 1]", arg81_1: "f32[512]", arg82_1: "f32[512]", arg83_1: "f32[512]", arg84_1: "f32[512]", arg85_1: "f32[512, 256, 1, 1]", arg86_1: "f32[512, 256, 1, 1]", arg87_1: "f32[512]", arg88_1: "f32[512]", arg89_1: "f32[512]", arg90_1: "f32[512]", arg91_1: "f32[128, 512, 1, 1]", arg92_1: "f32[128, 512, 1, 1]", arg93_1: "f32[128]", arg94_1: "f32[128]", arg95_1: "f32[128]", arg96_1: "f32[128]", arg97_1: "f32[128, 128, 3, 3]", arg98_1: "f32[128, 128, 3, 3]", arg99_1: "f32[128]", arg100_1: "f32[128]", arg101_1: "f32[128]", arg102_1: "f32[128]", arg103_1: "f32[512, 128, 1, 1]", arg104_1: "f32[512, 128, 1, 1]", arg105_1: "f32[512]", arg106_1: "f32[512]", arg107_1: "f32[512]", arg108_1: "f32[512]", arg109_1: "f32[128, 512, 1, 1]", arg110_1: "f32[128, 512, 1, 1]", arg111_1: "f32[128]", arg112_1: "f32[128]", arg113_1: "f32[128]", arg114_1: "f32[128]", arg115_1: "f32[128, 128, 3, 3]", arg116_1: "f32[128, 128, 3, 3]", arg117_1: "f32[128]", arg118_1: "f32[128]", arg119_1: "f32[128]", arg120_1: "f32[128]", arg121_1: "f32[512, 128, 1, 1]", arg122_1: "f32[512, 128, 1, 1]", arg123_1: "f32[512]", arg124_1: "f32[512]", arg125_1: "f32[512]", arg126_1: "f32[512]", arg127_1: "f32[128, 512, 1, 1]", arg128_1: "f32[128, 512, 1, 1]", arg129_1: "f32[128]", arg130_1: "f32[128]", arg131_1: "f32[128]", arg132_1: "f32[128]", arg133_1: "f32[128, 128, 3, 3]", arg134_1: "f32[128, 128, 3, 3]", arg135_1: "f32[128]", arg136_1: "f32[128]", arg137_1: "f32[128]", arg138_1: "f32[128]", arg139_1: "f32[512, 128, 1, 1]", arg140_1: "f32[512, 128, 1, 1]", arg141_1: "f32[512]", arg142_1: "f32[512]", arg143_1: "f32[512]", arg144_1: "f32[512]", arg145_1: "f32[256, 512, 1, 1]", arg146_1: "f32[256, 512, 1, 1]", arg147_1: "f32[256]", arg148_1: "f32[256]", arg149_1: "f32[256]", arg150_1: "f32[256]", arg151_1: "f32[256, 256, 3, 3]", arg152_1: "f32[256, 256, 3, 3]", arg153_1: "f32[256]", arg154_1: "f32[256]", arg155_1: "f32[256]", arg156_1: "f32[256]", arg157_1: "f32[1024, 256, 1, 1]", arg158_1: "f32[1024, 256, 1, 1]", arg159_1: "f32[1024]", arg160_1: "f32[1024]", arg161_1: "f32[1024]", arg162_1: "f32[1024]", arg163_1: "f32[1024, 512, 1, 1]", arg164_1: "f32[1024, 512, 1, 1]", arg165_1: "f32[1024]", arg166_1: "f32[1024]", arg167_1: "f32[1024]", arg168_1: "f32[1024]", arg169_1: "f32[256, 1024, 1, 1]", arg170_1: "f32[256, 1024, 1, 1]", arg171_1: "f32[256]", arg172_1: "f32[256]", arg173_1: "f32[256]", arg174_1: "f32[256]", arg175_1: "f32[256, 256, 3, 3]", arg176_1: "f32[256, 256, 3, 3]", arg177_1: "f32[256]", arg178_1: "f32[256]", arg179_1: "f32[256]", arg180_1: "f32[256]", arg181_1: "f32[1024, 256, 1, 1]", arg182_1: "f32[1024, 256, 1, 1]", arg183_1: "f32[1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024]", arg186_1: "f32[1024]", arg187_1: "f32[256, 1024, 1, 1]", arg188_1: "f32[256, 1024, 1, 1]", arg189_1: "f32[256]", arg190_1: "f32[256]", arg191_1: "f32[256]", arg192_1: "f32[256]", arg193_1: "f32[256, 256, 3, 3]", arg194_1: "f32[256, 256, 3, 3]", arg195_1: "f32[256]", arg196_1: "f32[256]", arg197_1: "f32[256]", arg198_1: "f32[256]", arg199_1: "f32[1024, 256, 1, 1]", arg200_1: "f32[1024, 256, 1, 1]", arg201_1: "f32[1024]", arg202_1: "f32[1024]", arg203_1: "f32[1024]", arg204_1: "f32[1024]", arg205_1: "f32[256, 1024, 1, 1]", arg206_1: "f32[256, 1024, 1, 1]", arg207_1: "f32[256]", arg208_1: "f32[256]", arg209_1: "f32[256]", arg210_1: "f32[256]", arg211_1: "f32[256, 256, 3, 3]", arg212_1: "f32[256, 256, 3, 3]", arg213_1: "f32[256]", arg214_1: "f32[256]", arg215_1: "f32[256]", arg216_1: "f32[256]", arg217_1: "f32[1024, 256, 1, 1]", arg218_1: "f32[1024, 256, 1, 1]", arg219_1: "f32[1024]", arg220_1: "f32[1024]", arg221_1: "f32[1024]", arg222_1: "f32[1024]", arg223_1: "f32[256, 1024, 1, 1]", arg224_1: "f32[256, 1024, 1, 1]", arg225_1: "f32[256]", arg226_1: "f32[256]", arg227_1: "f32[256]", arg228_1: "f32[256]", arg229_1: "f32[256, 256, 3, 3]", arg230_1: "f32[256, 256, 3, 3]", arg231_1: "f32[256]", arg232_1: "f32[256]", arg233_1: "f32[256]", arg234_1: "f32[256]", arg235_1: "f32[1024, 256, 1, 1]", arg236_1: "f32[1024, 256, 1, 1]", arg237_1: "f32[1024]", arg238_1: "f32[1024]", arg239_1: "f32[1024]", arg240_1: "f32[1024]", arg241_1: "f32[256, 1024, 1, 1]", arg242_1: "f32[256, 1024, 1, 1]", arg243_1: "f32[256]", arg244_1: "f32[256]", arg245_1: "f32[256]", arg246_1: "f32[256]", arg247_1: "f32[256, 256, 3, 3]", arg248_1: "f32[256, 256, 3, 3]", arg249_1: "f32[256]", arg250_1: "f32[256]", arg251_1: "f32[256]", arg252_1: "f32[256]", arg253_1: "f32[1024, 256, 1, 1]", arg254_1: "f32[1024, 256, 1, 1]", arg255_1: "f32[1024]", arg256_1: "f32[1024]", arg257_1: "f32[1024]", arg258_1: "f32[1024]", arg259_1: "f32[512, 1024, 1, 1]", arg260_1: "f32[512, 1024, 1, 1]", arg261_1: "f32[512]", arg262_1: "f32[512]", arg263_1: "f32[512]", arg264_1: "f32[512]", arg265_1: "f32[512, 512, 3, 3]", arg266_1: "f32[512, 512, 3, 3]", arg267_1: "f32[512]", arg268_1: "f32[512]", arg269_1: "f32[512]", arg270_1: "f32[512]", arg271_1: "f32[2048, 512, 1, 1]", arg272_1: "f32[2048, 512, 1, 1]", arg273_1: "f32[2048]", arg274_1: "f32[2048]", arg275_1: "f32[2048]", arg276_1: "f32[2048]", arg277_1: "f32[2048, 1024, 1, 1]", arg278_1: "f32[2048, 1024, 1, 1]", arg279_1: "f32[2048]", arg280_1: "f32[2048]", arg281_1: "f32[2048]", arg282_1: "f32[2048]", arg283_1: "f32[512, 2048, 1, 1]", arg284_1: "f32[512, 2048, 1, 1]", arg285_1: "f32[512]", arg286_1: "f32[512]", arg287_1: "f32[512]", arg288_1: "f32[512]", arg289_1: "f32[512, 512, 3, 3]", arg290_1: "f32[512, 512, 3, 3]", arg291_1: "f32[512]", arg292_1: "f32[512]", arg293_1: "f32[512]", arg294_1: "f32[512]", arg295_1: "f32[2048, 512, 1, 1]", arg296_1: "f32[2048, 512, 1, 1]", arg297_1: "f32[2048]", arg298_1: "f32[2048]", arg299_1: "f32[2048]", arg300_1: "f32[2048]", arg301_1: "f32[512, 2048, 1, 1]", arg302_1: "f32[512, 2048, 1, 1]", arg303_1: "f32[512]", arg304_1: "f32[512]", arg305_1: "f32[512]", arg306_1: "f32[512]", arg307_1: "f32[512, 512, 3, 3]", arg308_1: "f32[512, 512, 3, 3]", arg309_1: "f32[512]", arg310_1: "f32[512]", arg311_1: "f32[512]", arg312_1: "f32[512]", arg313_1: "f32[2048, 512, 1, 1]", arg314_1: "f32[2048, 512, 1, 1]", arg315_1: "f32[2048]", arg316_1: "f32[2048]", arg317_1: "f32[2048]", arg318_1: "f32[2048]", arg319_1: "f32[128, 2048]", arg320_1: "f32[128, 2048]", arg321_1: "f32[128]", arg322_1: "f32[128]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:135 in forward, code: q = nn.functional.normalize(q, dim=1)
        pow_1: "f32[32, 128]" = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 2.0)
        sum_1: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        pow_2: "f32[32, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        clamp_min: "f32[32, 1]" = torch.ops.aten.clamp_min.default(pow_2, 1e-12);  pow_2 = None
        expand: "f32[32, 128]" = torch.ops.aten.expand.default(clamp_min, [32, 128]);  clamp_min = None
        div: "f32[32, 128]" = torch.ops.aten.div.Tensor(arg0_1, expand);  arg0_1 = expand = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:59 in _momentum_update_key_encoder, code: param_k.mul_(self.m).add_(param_q.mul(1.0 - self.m))
        mul: "f32[64, 3, 7, 7]" = torch.ops.aten.mul.Tensor(arg1_1, 0.999)
        mul_1: "f32[64, 3, 7, 7]" = torch.ops.aten.mul.Tensor(arg2_1, 0.0010000000000000009);  arg2_1 = None
        add: "f32[64, 3, 7, 7]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(arg3_1, 0.999)
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(arg4_1, 0.0010000000000000009);  arg4_1 = None
        add_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(arg5_1, 0.999)
        mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(arg6_1, 0.0010000000000000009);  arg6_1 = None
        add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        mul_6: "f32[64, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg7_1, 0.999)
        mul_7: "f32[64, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg8_1, 0.0010000000000000009);  arg8_1 = None
        add_3: "f32[64, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(arg9_1, 0.999)
        mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(arg10_1, 0.0010000000000000009);  arg10_1 = None
        add_4: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(arg11_1, 0.999)
        mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(arg12_1, 0.0010000000000000009);  arg12_1 = None
        add_5: "f32[64]" = torch.ops.aten.add.Tensor(mul_10, mul_11);  mul_10 = mul_11 = None
        mul_12: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg13_1, 0.999)
        mul_13: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg14_1, 0.0010000000000000009);  arg14_1 = None
        add_6: "f32[64, 64, 3, 3]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        mul_14: "f32[64]" = torch.ops.aten.mul.Tensor(arg15_1, 0.999)
        mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(arg16_1, 0.0010000000000000009);  arg16_1 = None
        add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
        mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(arg17_1, 0.999)
        mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(arg18_1, 0.0010000000000000009);  arg18_1 = None
        add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        mul_18: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg19_1, 0.999)
        mul_19: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg20_1, 0.0010000000000000009);  arg20_1 = None
        add_9: "f32[256, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        mul_20: "f32[256]" = torch.ops.aten.mul.Tensor(arg21_1, 0.999)
        mul_21: "f32[256]" = torch.ops.aten.mul.Tensor(arg22_1, 0.0010000000000000009);  arg22_1 = None
        add_10: "f32[256]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
        mul_22: "f32[256]" = torch.ops.aten.mul.Tensor(arg23_1, 0.999)
        mul_23: "f32[256]" = torch.ops.aten.mul.Tensor(arg24_1, 0.0010000000000000009);  arg24_1 = None
        add_11: "f32[256]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        mul_24: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg25_1, 0.999)
        mul_25: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg26_1, 0.0010000000000000009);  arg26_1 = None
        add_12: "f32[256, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None
        mul_26: "f32[256]" = torch.ops.aten.mul.Tensor(arg27_1, 0.999)
        mul_27: "f32[256]" = torch.ops.aten.mul.Tensor(arg28_1, 0.0010000000000000009);  arg28_1 = None
        add_13: "f32[256]" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
        mul_28: "f32[256]" = torch.ops.aten.mul.Tensor(arg29_1, 0.999)
        mul_29: "f32[256]" = torch.ops.aten.mul.Tensor(arg30_1, 0.0010000000000000009);  arg30_1 = None
        add_14: "f32[256]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
        mul_30: "f32[64, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg31_1, 0.999)
        mul_31: "f32[64, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg32_1, 0.0010000000000000009);  arg32_1 = None
        add_15: "f32[64, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
        mul_32: "f32[64]" = torch.ops.aten.mul.Tensor(arg33_1, 0.999)
        mul_33: "f32[64]" = torch.ops.aten.mul.Tensor(arg34_1, 0.0010000000000000009);  arg34_1 = None
        add_16: "f32[64]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        mul_34: "f32[64]" = torch.ops.aten.mul.Tensor(arg35_1, 0.999)
        mul_35: "f32[64]" = torch.ops.aten.mul.Tensor(arg36_1, 0.0010000000000000009);  arg36_1 = None
        add_17: "f32[64]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
        mul_36: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg37_1, 0.999)
        mul_37: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg38_1, 0.0010000000000000009);  arg38_1 = None
        add_18: "f32[64, 64, 3, 3]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        mul_38: "f32[64]" = torch.ops.aten.mul.Tensor(arg39_1, 0.999)
        mul_39: "f32[64]" = torch.ops.aten.mul.Tensor(arg40_1, 0.0010000000000000009);  arg40_1 = None
        add_19: "f32[64]" = torch.ops.aten.add.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None
        mul_40: "f32[64]" = torch.ops.aten.mul.Tensor(arg41_1, 0.999)
        mul_41: "f32[64]" = torch.ops.aten.mul.Tensor(arg42_1, 0.0010000000000000009);  arg42_1 = None
        add_20: "f32[64]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
        mul_42: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg43_1, 0.999)
        mul_43: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg44_1, 0.0010000000000000009);  arg44_1 = None
        add_21: "f32[256, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
        mul_44: "f32[256]" = torch.ops.aten.mul.Tensor(arg45_1, 0.999)
        mul_45: "f32[256]" = torch.ops.aten.mul.Tensor(arg46_1, 0.0010000000000000009);  arg46_1 = None
        add_22: "f32[256]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
        mul_46: "f32[256]" = torch.ops.aten.mul.Tensor(arg47_1, 0.999)
        mul_47: "f32[256]" = torch.ops.aten.mul.Tensor(arg48_1, 0.0010000000000000009);  arg48_1 = None
        add_23: "f32[256]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        mul_48: "f32[64, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg49_1, 0.999)
        mul_49: "f32[64, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg50_1, 0.0010000000000000009);  arg50_1 = None
        add_24: "f32[64, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
        mul_50: "f32[64]" = torch.ops.aten.mul.Tensor(arg51_1, 0.999)
        mul_51: "f32[64]" = torch.ops.aten.mul.Tensor(arg52_1, 0.0010000000000000009);  arg52_1 = None
        add_25: "f32[64]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        mul_52: "f32[64]" = torch.ops.aten.mul.Tensor(arg53_1, 0.999)
        mul_53: "f32[64]" = torch.ops.aten.mul.Tensor(arg54_1, 0.0010000000000000009);  arg54_1 = None
        add_26: "f32[64]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
        mul_54: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg55_1, 0.999)
        mul_55: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg56_1, 0.0010000000000000009);  arg56_1 = None
        add_27: "f32[64, 64, 3, 3]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
        mul_56: "f32[64]" = torch.ops.aten.mul.Tensor(arg57_1, 0.999)
        mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(arg58_1, 0.0010000000000000009);  arg58_1 = None
        add_28: "f32[64]" = torch.ops.aten.add.Tensor(mul_56, mul_57);  mul_56 = mul_57 = None
        mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(arg59_1, 0.999)
        mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(arg60_1, 0.0010000000000000009);  arg60_1 = None
        add_29: "f32[64]" = torch.ops.aten.add.Tensor(mul_58, mul_59);  mul_58 = mul_59 = None
        mul_60: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg61_1, 0.999)
        mul_61: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg62_1, 0.0010000000000000009);  arg62_1 = None
        add_30: "f32[256, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        mul_62: "f32[256]" = torch.ops.aten.mul.Tensor(arg63_1, 0.999)
        mul_63: "f32[256]" = torch.ops.aten.mul.Tensor(arg64_1, 0.0010000000000000009);  arg64_1 = None
        add_31: "f32[256]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
        mul_64: "f32[256]" = torch.ops.aten.mul.Tensor(arg65_1, 0.999)
        mul_65: "f32[256]" = torch.ops.aten.mul.Tensor(arg66_1, 0.0010000000000000009);  arg66_1 = None
        add_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        mul_66: "f32[128, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg67_1, 0.999)
        mul_67: "f32[128, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg68_1, 0.0010000000000000009);  arg68_1 = None
        add_33: "f32[128, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
        mul_68: "f32[128]" = torch.ops.aten.mul.Tensor(arg69_1, 0.999)
        mul_69: "f32[128]" = torch.ops.aten.mul.Tensor(arg70_1, 0.0010000000000000009);  arg70_1 = None
        add_34: "f32[128]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
        mul_70: "f32[128]" = torch.ops.aten.mul.Tensor(arg71_1, 0.999)
        mul_71: "f32[128]" = torch.ops.aten.mul.Tensor(arg72_1, 0.0010000000000000009);  arg72_1 = None
        add_35: "f32[128]" = torch.ops.aten.add.Tensor(mul_70, mul_71);  mul_70 = mul_71 = None
        mul_72: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg73_1, 0.999)
        mul_73: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg74_1, 0.0010000000000000009);  arg74_1 = None
        add_36: "f32[128, 128, 3, 3]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
        mul_74: "f32[128]" = torch.ops.aten.mul.Tensor(arg75_1, 0.999)
        mul_75: "f32[128]" = torch.ops.aten.mul.Tensor(arg76_1, 0.0010000000000000009);  arg76_1 = None
        add_37: "f32[128]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        mul_76: "f32[128]" = torch.ops.aten.mul.Tensor(arg77_1, 0.999)
        mul_77: "f32[128]" = torch.ops.aten.mul.Tensor(arg78_1, 0.0010000000000000009);  arg78_1 = None
        add_38: "f32[128]" = torch.ops.aten.add.Tensor(mul_76, mul_77);  mul_76 = mul_77 = None
        mul_78: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg79_1, 0.999)
        mul_79: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg80_1, 0.0010000000000000009);  arg80_1 = None
        add_39: "f32[512, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        mul_80: "f32[512]" = torch.ops.aten.mul.Tensor(arg81_1, 0.999)
        mul_81: "f32[512]" = torch.ops.aten.mul.Tensor(arg82_1, 0.0010000000000000009);  arg82_1 = None
        add_40: "f32[512]" = torch.ops.aten.add.Tensor(mul_80, mul_81);  mul_80 = mul_81 = None
        mul_82: "f32[512]" = torch.ops.aten.mul.Tensor(arg83_1, 0.999)
        mul_83: "f32[512]" = torch.ops.aten.mul.Tensor(arg84_1, 0.0010000000000000009);  arg84_1 = None
        add_41: "f32[512]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
        mul_84: "f32[512, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg85_1, 0.999)
        mul_85: "f32[512, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg86_1, 0.0010000000000000009);  arg86_1 = None
        add_42: "f32[512, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
        mul_86: "f32[512]" = torch.ops.aten.mul.Tensor(arg87_1, 0.999)
        mul_87: "f32[512]" = torch.ops.aten.mul.Tensor(arg88_1, 0.0010000000000000009);  arg88_1 = None
        add_43: "f32[512]" = torch.ops.aten.add.Tensor(mul_86, mul_87);  mul_86 = mul_87 = None
        mul_88: "f32[512]" = torch.ops.aten.mul.Tensor(arg89_1, 0.999)
        mul_89: "f32[512]" = torch.ops.aten.mul.Tensor(arg90_1, 0.0010000000000000009);  arg90_1 = None
        add_44: "f32[512]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        mul_90: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg91_1, 0.999)
        mul_91: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg92_1, 0.0010000000000000009);  arg92_1 = None
        add_45: "f32[128, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
        mul_92: "f32[128]" = torch.ops.aten.mul.Tensor(arg93_1, 0.999)
        mul_93: "f32[128]" = torch.ops.aten.mul.Tensor(arg94_1, 0.0010000000000000009);  arg94_1 = None
        add_46: "f32[128]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        mul_94: "f32[128]" = torch.ops.aten.mul.Tensor(arg95_1, 0.999)
        mul_95: "f32[128]" = torch.ops.aten.mul.Tensor(arg96_1, 0.0010000000000000009);  arg96_1 = None
        add_47: "f32[128]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
        mul_96: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg97_1, 0.999)
        mul_97: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg98_1, 0.0010000000000000009);  arg98_1 = None
        add_48: "f32[128, 128, 3, 3]" = torch.ops.aten.add.Tensor(mul_96, mul_97);  mul_96 = mul_97 = None
        mul_98: "f32[128]" = torch.ops.aten.mul.Tensor(arg99_1, 0.999)
        mul_99: "f32[128]" = torch.ops.aten.mul.Tensor(arg100_1, 0.0010000000000000009);  arg100_1 = None
        add_49: "f32[128]" = torch.ops.aten.add.Tensor(mul_98, mul_99);  mul_98 = mul_99 = None
        mul_100: "f32[128]" = torch.ops.aten.mul.Tensor(arg101_1, 0.999)
        mul_101: "f32[128]" = torch.ops.aten.mul.Tensor(arg102_1, 0.0010000000000000009);  arg102_1 = None
        add_50: "f32[128]" = torch.ops.aten.add.Tensor(mul_100, mul_101);  mul_100 = mul_101 = None
        mul_102: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg103_1, 0.999)
        mul_103: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg104_1, 0.0010000000000000009);  arg104_1 = None
        add_51: "f32[512, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        mul_104: "f32[512]" = torch.ops.aten.mul.Tensor(arg105_1, 0.999)
        mul_105: "f32[512]" = torch.ops.aten.mul.Tensor(arg106_1, 0.0010000000000000009);  arg106_1 = None
        add_52: "f32[512]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
        mul_106: "f32[512]" = torch.ops.aten.mul.Tensor(arg107_1, 0.999)
        mul_107: "f32[512]" = torch.ops.aten.mul.Tensor(arg108_1, 0.0010000000000000009);  arg108_1 = None
        add_53: "f32[512]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        mul_108: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg109_1, 0.999)
        mul_109: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg110_1, 0.0010000000000000009);  arg110_1 = None
        add_54: "f32[128, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
        mul_110: "f32[128]" = torch.ops.aten.mul.Tensor(arg111_1, 0.999)
        mul_111: "f32[128]" = torch.ops.aten.mul.Tensor(arg112_1, 0.0010000000000000009);  arg112_1 = None
        add_55: "f32[128]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
        mul_112: "f32[128]" = torch.ops.aten.mul.Tensor(arg113_1, 0.999)
        mul_113: "f32[128]" = torch.ops.aten.mul.Tensor(arg114_1, 0.0010000000000000009);  arg114_1 = None
        add_56: "f32[128]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
        mul_114: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg115_1, 0.999)
        mul_115: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg116_1, 0.0010000000000000009);  arg116_1 = None
        add_57: "f32[128, 128, 3, 3]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
        mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(arg117_1, 0.999)
        mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(arg118_1, 0.0010000000000000009);  arg118_1 = None
        add_58: "f32[128]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        mul_118: "f32[128]" = torch.ops.aten.mul.Tensor(arg119_1, 0.999)
        mul_119: "f32[128]" = torch.ops.aten.mul.Tensor(arg120_1, 0.0010000000000000009);  arg120_1 = None
        add_59: "f32[128]" = torch.ops.aten.add.Tensor(mul_118, mul_119);  mul_118 = mul_119 = None
        mul_120: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg121_1, 0.999)
        mul_121: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg122_1, 0.0010000000000000009);  arg122_1 = None
        add_60: "f32[512, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        mul_122: "f32[512]" = torch.ops.aten.mul.Tensor(arg123_1, 0.999)
        mul_123: "f32[512]" = torch.ops.aten.mul.Tensor(arg124_1, 0.0010000000000000009);  arg124_1 = None
        add_61: "f32[512]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
        mul_124: "f32[512]" = torch.ops.aten.mul.Tensor(arg125_1, 0.999)
        mul_125: "f32[512]" = torch.ops.aten.mul.Tensor(arg126_1, 0.0010000000000000009);  arg126_1 = None
        add_62: "f32[512]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
        mul_126: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg127_1, 0.999)
        mul_127: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg128_1, 0.0010000000000000009);  arg128_1 = None
        add_63: "f32[128, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_126, mul_127);  mul_126 = mul_127 = None
        mul_128: "f32[128]" = torch.ops.aten.mul.Tensor(arg129_1, 0.999)
        mul_129: "f32[128]" = torch.ops.aten.mul.Tensor(arg130_1, 0.0010000000000000009);  arg130_1 = None
        add_64: "f32[128]" = torch.ops.aten.add.Tensor(mul_128, mul_129);  mul_128 = mul_129 = None
        mul_130: "f32[128]" = torch.ops.aten.mul.Tensor(arg131_1, 0.999)
        mul_131: "f32[128]" = torch.ops.aten.mul.Tensor(arg132_1, 0.0010000000000000009);  arg132_1 = None
        add_65: "f32[128]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        mul_132: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg133_1, 0.999)
        mul_133: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg134_1, 0.0010000000000000009);  arg134_1 = None
        add_66: "f32[128, 128, 3, 3]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
        mul_134: "f32[128]" = torch.ops.aten.mul.Tensor(arg135_1, 0.999)
        mul_135: "f32[128]" = torch.ops.aten.mul.Tensor(arg136_1, 0.0010000000000000009);  arg136_1 = None
        add_67: "f32[128]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        mul_136: "f32[128]" = torch.ops.aten.mul.Tensor(arg137_1, 0.999)
        mul_137: "f32[128]" = torch.ops.aten.mul.Tensor(arg138_1, 0.0010000000000000009);  arg138_1 = None
        add_68: "f32[128]" = torch.ops.aten.add.Tensor(mul_136, mul_137);  mul_136 = mul_137 = None
        mul_138: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg139_1, 0.999)
        mul_139: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg140_1, 0.0010000000000000009);  arg140_1 = None
        add_69: "f32[512, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_138, mul_139);  mul_138 = mul_139 = None
        mul_140: "f32[512]" = torch.ops.aten.mul.Tensor(arg141_1, 0.999)
        mul_141: "f32[512]" = torch.ops.aten.mul.Tensor(arg142_1, 0.0010000000000000009);  arg142_1 = None
        add_70: "f32[512]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
        mul_142: "f32[512]" = torch.ops.aten.mul.Tensor(arg143_1, 0.999)
        mul_143: "f32[512]" = torch.ops.aten.mul.Tensor(arg144_1, 0.0010000000000000009);  arg144_1 = None
        add_71: "f32[512]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
        mul_144: "f32[256, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg145_1, 0.999)
        mul_145: "f32[256, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg146_1, 0.0010000000000000009);  arg146_1 = None
        add_72: "f32[256, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        mul_146: "f32[256]" = torch.ops.aten.mul.Tensor(arg147_1, 0.999)
        mul_147: "f32[256]" = torch.ops.aten.mul.Tensor(arg148_1, 0.0010000000000000009);  arg148_1 = None
        add_73: "f32[256]" = torch.ops.aten.add.Tensor(mul_146, mul_147);  mul_146 = mul_147 = None
        mul_148: "f32[256]" = torch.ops.aten.mul.Tensor(arg149_1, 0.999)
        mul_149: "f32[256]" = torch.ops.aten.mul.Tensor(arg150_1, 0.0010000000000000009);  arg150_1 = None
        add_74: "f32[256]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        mul_150: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg151_1, 0.999)
        mul_151: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg152_1, 0.0010000000000000009);  arg152_1 = None
        add_75: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None
        mul_152: "f32[256]" = torch.ops.aten.mul.Tensor(arg153_1, 0.999)
        mul_153: "f32[256]" = torch.ops.aten.mul.Tensor(arg154_1, 0.0010000000000000009);  arg154_1 = None
        add_76: "f32[256]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
        mul_154: "f32[256]" = torch.ops.aten.mul.Tensor(arg155_1, 0.999)
        mul_155: "f32[256]" = torch.ops.aten.mul.Tensor(arg156_1, 0.0010000000000000009);  arg156_1 = None
        add_77: "f32[256]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
        mul_156: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg157_1, 0.999)
        mul_157: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg158_1, 0.0010000000000000009);  arg158_1 = None
        add_78: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
        mul_158: "f32[1024]" = torch.ops.aten.mul.Tensor(arg159_1, 0.999)
        mul_159: "f32[1024]" = torch.ops.aten.mul.Tensor(arg160_1, 0.0010000000000000009);  arg160_1 = None
        add_79: "f32[1024]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        mul_160: "f32[1024]" = torch.ops.aten.mul.Tensor(arg161_1, 0.999)
        mul_161: "f32[1024]" = torch.ops.aten.mul.Tensor(arg162_1, 0.0010000000000000009);  arg162_1 = None
        add_80: "f32[1024]" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
        mul_162: "f32[1024, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg163_1, 0.999)
        mul_163: "f32[1024, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg164_1, 0.0010000000000000009);  arg164_1 = None
        add_81: "f32[1024, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        mul_164: "f32[1024]" = torch.ops.aten.mul.Tensor(arg165_1, 0.999)
        mul_165: "f32[1024]" = torch.ops.aten.mul.Tensor(arg166_1, 0.0010000000000000009);  arg166_1 = None
        add_82: "f32[1024]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
        mul_166: "f32[1024]" = torch.ops.aten.mul.Tensor(arg167_1, 0.999)
        mul_167: "f32[1024]" = torch.ops.aten.mul.Tensor(arg168_1, 0.0010000000000000009);  arg168_1 = None
        add_83: "f32[1024]" = torch.ops.aten.add.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None
        mul_168: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg169_1, 0.999)
        mul_169: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg170_1, 0.0010000000000000009);  arg170_1 = None
        add_84: "f32[256, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_168, mul_169);  mul_168 = mul_169 = None
        mul_170: "f32[256]" = torch.ops.aten.mul.Tensor(arg171_1, 0.999)
        mul_171: "f32[256]" = torch.ops.aten.mul.Tensor(arg172_1, 0.0010000000000000009);  arg172_1 = None
        add_85: "f32[256]" = torch.ops.aten.add.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
        mul_172: "f32[256]" = torch.ops.aten.mul.Tensor(arg173_1, 0.999)
        mul_173: "f32[256]" = torch.ops.aten.mul.Tensor(arg174_1, 0.0010000000000000009);  arg174_1 = None
        add_86: "f32[256]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        mul_174: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg175_1, 0.999)
        mul_175: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg176_1, 0.0010000000000000009);  arg176_1 = None
        add_87: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
        mul_176: "f32[256]" = torch.ops.aten.mul.Tensor(arg177_1, 0.999)
        mul_177: "f32[256]" = torch.ops.aten.mul.Tensor(arg178_1, 0.0010000000000000009);  arg178_1 = None
        add_88: "f32[256]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        mul_178: "f32[256]" = torch.ops.aten.mul.Tensor(arg179_1, 0.999)
        mul_179: "f32[256]" = torch.ops.aten.mul.Tensor(arg180_1, 0.0010000000000000009);  arg180_1 = None
        add_89: "f32[256]" = torch.ops.aten.add.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
        mul_180: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg181_1, 0.999)
        mul_181: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg182_1, 0.0010000000000000009);  arg182_1 = None
        add_90: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
        mul_182: "f32[1024]" = torch.ops.aten.mul.Tensor(arg183_1, 0.999)
        mul_183: "f32[1024]" = torch.ops.aten.mul.Tensor(arg184_1, 0.0010000000000000009);  arg184_1 = None
        add_91: "f32[1024]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
        mul_184: "f32[1024]" = torch.ops.aten.mul.Tensor(arg185_1, 0.999)
        mul_185: "f32[1024]" = torch.ops.aten.mul.Tensor(arg186_1, 0.0010000000000000009);  arg186_1 = None
        add_92: "f32[1024]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
        mul_186: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg187_1, 0.999)
        mul_187: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg188_1, 0.0010000000000000009);  arg188_1 = None
        add_93: "f32[256, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        mul_188: "f32[256]" = torch.ops.aten.mul.Tensor(arg189_1, 0.999)
        mul_189: "f32[256]" = torch.ops.aten.mul.Tensor(arg190_1, 0.0010000000000000009);  arg190_1 = None
        add_94: "f32[256]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
        mul_190: "f32[256]" = torch.ops.aten.mul.Tensor(arg191_1, 0.999)
        mul_191: "f32[256]" = torch.ops.aten.mul.Tensor(arg192_1, 0.0010000000000000009);  arg192_1 = None
        add_95: "f32[256]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        mul_192: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg193_1, 0.999)
        mul_193: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg194_1, 0.0010000000000000009);  arg194_1 = None
        add_96: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
        mul_194: "f32[256]" = torch.ops.aten.mul.Tensor(arg195_1, 0.999)
        mul_195: "f32[256]" = torch.ops.aten.mul.Tensor(arg196_1, 0.0010000000000000009);  arg196_1 = None
        add_97: "f32[256]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
        mul_196: "f32[256]" = torch.ops.aten.mul.Tensor(arg197_1, 0.999)
        mul_197: "f32[256]" = torch.ops.aten.mul.Tensor(arg198_1, 0.0010000000000000009);  arg198_1 = None
        add_98: "f32[256]" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
        mul_198: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg199_1, 0.999)
        mul_199: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg200_1, 0.0010000000000000009);  arg200_1 = None
        add_99: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None
        mul_200: "f32[1024]" = torch.ops.aten.mul.Tensor(arg201_1, 0.999)
        mul_201: "f32[1024]" = torch.ops.aten.mul.Tensor(arg202_1, 0.0010000000000000009);  arg202_1 = None
        add_100: "f32[1024]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        mul_202: "f32[1024]" = torch.ops.aten.mul.Tensor(arg203_1, 0.999)
        mul_203: "f32[1024]" = torch.ops.aten.mul.Tensor(arg204_1, 0.0010000000000000009);  arg204_1 = None
        add_101: "f32[1024]" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
        mul_204: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg205_1, 0.999)
        mul_205: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg206_1, 0.0010000000000000009);  arg206_1 = None
        add_102: "f32[256, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        mul_206: "f32[256]" = torch.ops.aten.mul.Tensor(arg207_1, 0.999)
        mul_207: "f32[256]" = torch.ops.aten.mul.Tensor(arg208_1, 0.0010000000000000009);  arg208_1 = None
        add_103: "f32[256]" = torch.ops.aten.add.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
        mul_208: "f32[256]" = torch.ops.aten.mul.Tensor(arg209_1, 0.999)
        mul_209: "f32[256]" = torch.ops.aten.mul.Tensor(arg210_1, 0.0010000000000000009);  arg210_1 = None
        add_104: "f32[256]" = torch.ops.aten.add.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
        mul_210: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg211_1, 0.999)
        mul_211: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg212_1, 0.0010000000000000009);  arg212_1 = None
        add_105: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
        mul_212: "f32[256]" = torch.ops.aten.mul.Tensor(arg213_1, 0.999)
        mul_213: "f32[256]" = torch.ops.aten.mul.Tensor(arg214_1, 0.0010000000000000009);  arg214_1 = None
        add_106: "f32[256]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
        mul_214: "f32[256]" = torch.ops.aten.mul.Tensor(arg215_1, 0.999)
        mul_215: "f32[256]" = torch.ops.aten.mul.Tensor(arg216_1, 0.0010000000000000009);  arg216_1 = None
        add_107: "f32[256]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        mul_216: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg217_1, 0.999)
        mul_217: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg218_1, 0.0010000000000000009);  arg218_1 = None
        add_108: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
        mul_218: "f32[1024]" = torch.ops.aten.mul.Tensor(arg219_1, 0.999)
        mul_219: "f32[1024]" = torch.ops.aten.mul.Tensor(arg220_1, 0.0010000000000000009);  arg220_1 = None
        add_109: "f32[1024]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        mul_220: "f32[1024]" = torch.ops.aten.mul.Tensor(arg221_1, 0.999)
        mul_221: "f32[1024]" = torch.ops.aten.mul.Tensor(arg222_1, 0.0010000000000000009);  arg222_1 = None
        add_110: "f32[1024]" = torch.ops.aten.add.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
        mul_222: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg223_1, 0.999)
        mul_223: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg224_1, 0.0010000000000000009);  arg224_1 = None
        add_111: "f32[256, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
        mul_224: "f32[256]" = torch.ops.aten.mul.Tensor(arg225_1, 0.999)
        mul_225: "f32[256]" = torch.ops.aten.mul.Tensor(arg226_1, 0.0010000000000000009);  arg226_1 = None
        add_112: "f32[256]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
        mul_226: "f32[256]" = torch.ops.aten.mul.Tensor(arg227_1, 0.999)
        mul_227: "f32[256]" = torch.ops.aten.mul.Tensor(arg228_1, 0.0010000000000000009);  arg228_1 = None
        add_113: "f32[256]" = torch.ops.aten.add.Tensor(mul_226, mul_227);  mul_226 = mul_227 = None
        mul_228: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg229_1, 0.999)
        mul_229: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg230_1, 0.0010000000000000009);  arg230_1 = None
        add_114: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        mul_230: "f32[256]" = torch.ops.aten.mul.Tensor(arg231_1, 0.999)
        mul_231: "f32[256]" = torch.ops.aten.mul.Tensor(arg232_1, 0.0010000000000000009);  arg232_1 = None
        add_115: "f32[256]" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None
        mul_232: "f32[256]" = torch.ops.aten.mul.Tensor(arg233_1, 0.999)
        mul_233: "f32[256]" = torch.ops.aten.mul.Tensor(arg234_1, 0.0010000000000000009);  arg234_1 = None
        add_116: "f32[256]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        mul_234: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg235_1, 0.999)
        mul_235: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg236_1, 0.0010000000000000009);  arg236_1 = None
        add_117: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
        mul_236: "f32[1024]" = torch.ops.aten.mul.Tensor(arg237_1, 0.999)
        mul_237: "f32[1024]" = torch.ops.aten.mul.Tensor(arg238_1, 0.0010000000000000009);  arg238_1 = None
        add_118: "f32[1024]" = torch.ops.aten.add.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
        mul_238: "f32[1024]" = torch.ops.aten.mul.Tensor(arg239_1, 0.999)
        mul_239: "f32[1024]" = torch.ops.aten.mul.Tensor(arg240_1, 0.0010000000000000009);  arg240_1 = None
        add_119: "f32[1024]" = torch.ops.aten.add.Tensor(mul_238, mul_239);  mul_238 = mul_239 = None
        mul_240: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg241_1, 0.999)
        mul_241: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg242_1, 0.0010000000000000009);  arg242_1 = None
        add_120: "f32[256, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_240, mul_241);  mul_240 = mul_241 = None
        mul_242: "f32[256]" = torch.ops.aten.mul.Tensor(arg243_1, 0.999)
        mul_243: "f32[256]" = torch.ops.aten.mul.Tensor(arg244_1, 0.0010000000000000009);  arg244_1 = None
        add_121: "f32[256]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        mul_244: "f32[256]" = torch.ops.aten.mul.Tensor(arg245_1, 0.999)
        mul_245: "f32[256]" = torch.ops.aten.mul.Tensor(arg246_1, 0.0010000000000000009);  arg246_1 = None
        add_122: "f32[256]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
        mul_246: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg247_1, 0.999)
        mul_247: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg248_1, 0.0010000000000000009);  arg248_1 = None
        add_123: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        mul_248: "f32[256]" = torch.ops.aten.mul.Tensor(arg249_1, 0.999)
        mul_249: "f32[256]" = torch.ops.aten.mul.Tensor(arg250_1, 0.0010000000000000009);  arg250_1 = None
        add_124: "f32[256]" = torch.ops.aten.add.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
        mul_250: "f32[256]" = torch.ops.aten.mul.Tensor(arg251_1, 0.999)
        mul_251: "f32[256]" = torch.ops.aten.mul.Tensor(arg252_1, 0.0010000000000000009);  arg252_1 = None
        add_125: "f32[256]" = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
        mul_252: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg253_1, 0.999)
        mul_253: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg254_1, 0.0010000000000000009);  arg254_1 = None
        add_126: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
        mul_254: "f32[1024]" = torch.ops.aten.mul.Tensor(arg255_1, 0.999)
        mul_255: "f32[1024]" = torch.ops.aten.mul.Tensor(arg256_1, 0.0010000000000000009);  arg256_1 = None
        add_127: "f32[1024]" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
        mul_256: "f32[1024]" = torch.ops.aten.mul.Tensor(arg257_1, 0.999)
        mul_257: "f32[1024]" = torch.ops.aten.mul.Tensor(arg258_1, 0.0010000000000000009);  arg258_1 = None
        add_128: "f32[1024]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        mul_258: "f32[512, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg259_1, 0.999)
        mul_259: "f32[512, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg260_1, 0.0010000000000000009);  arg260_1 = None
        add_129: "f32[512, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
        mul_260: "f32[512]" = torch.ops.aten.mul.Tensor(arg261_1, 0.999)
        mul_261: "f32[512]" = torch.ops.aten.mul.Tensor(arg262_1, 0.0010000000000000009);  arg262_1 = None
        add_130: "f32[512]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        mul_262: "f32[512]" = torch.ops.aten.mul.Tensor(arg263_1, 0.999)
        mul_263: "f32[512]" = torch.ops.aten.mul.Tensor(arg264_1, 0.0010000000000000009);  arg264_1 = None
        add_131: "f32[512]" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
        mul_264: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg265_1, 0.999)
        mul_265: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg266_1, 0.0010000000000000009);  arg266_1 = None
        add_132: "f32[512, 512, 3, 3]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
        mul_266: "f32[512]" = torch.ops.aten.mul.Tensor(arg267_1, 0.999)
        mul_267: "f32[512]" = torch.ops.aten.mul.Tensor(arg268_1, 0.0010000000000000009);  arg268_1 = None
        add_133: "f32[512]" = torch.ops.aten.add.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
        mul_268: "f32[512]" = torch.ops.aten.mul.Tensor(arg269_1, 0.999)
        mul_269: "f32[512]" = torch.ops.aten.mul.Tensor(arg270_1, 0.0010000000000000009);  arg270_1 = None
        add_134: "f32[512]" = torch.ops.aten.add.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
        mul_270: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg271_1, 0.999)
        mul_271: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg272_1, 0.0010000000000000009);  arg272_1 = None
        add_135: "f32[2048, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        mul_272: "f32[2048]" = torch.ops.aten.mul.Tensor(arg273_1, 0.999)
        mul_273: "f32[2048]" = torch.ops.aten.mul.Tensor(arg274_1, 0.0010000000000000009);  arg274_1 = None
        add_136: "f32[2048]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
        mul_274: "f32[2048]" = torch.ops.aten.mul.Tensor(arg275_1, 0.999)
        mul_275: "f32[2048]" = torch.ops.aten.mul.Tensor(arg276_1, 0.0010000000000000009);  arg276_1 = None
        add_137: "f32[2048]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        mul_276: "f32[2048, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg277_1, 0.999)
        mul_277: "f32[2048, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg278_1, 0.0010000000000000009);  arg278_1 = None
        add_138: "f32[2048, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
        mul_278: "f32[2048]" = torch.ops.aten.mul.Tensor(arg279_1, 0.999)
        mul_279: "f32[2048]" = torch.ops.aten.mul.Tensor(arg280_1, 0.0010000000000000009);  arg280_1 = None
        add_139: "f32[2048]" = torch.ops.aten.add.Tensor(mul_278, mul_279);  mul_278 = mul_279 = None
        mul_280: "f32[2048]" = torch.ops.aten.mul.Tensor(arg281_1, 0.999)
        mul_281: "f32[2048]" = torch.ops.aten.mul.Tensor(arg282_1, 0.0010000000000000009);  arg282_1 = None
        add_140: "f32[2048]" = torch.ops.aten.add.Tensor(mul_280, mul_281);  mul_280 = mul_281 = None
        mul_282: "f32[512, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(arg283_1, 0.999)
        mul_283: "f32[512, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(arg284_1, 0.0010000000000000009);  arg284_1 = None
        add_141: "f32[512, 2048, 1, 1]" = torch.ops.aten.add.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
        mul_284: "f32[512]" = torch.ops.aten.mul.Tensor(arg285_1, 0.999)
        mul_285: "f32[512]" = torch.ops.aten.mul.Tensor(arg286_1, 0.0010000000000000009);  arg286_1 = None
        add_142: "f32[512]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        mul_286: "f32[512]" = torch.ops.aten.mul.Tensor(arg287_1, 0.999)
        mul_287: "f32[512]" = torch.ops.aten.mul.Tensor(arg288_1, 0.0010000000000000009);  arg288_1 = None
        add_143: "f32[512]" = torch.ops.aten.add.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None
        mul_288: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg289_1, 0.999)
        mul_289: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg290_1, 0.0010000000000000009);  arg290_1 = None
        add_144: "f32[512, 512, 3, 3]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        mul_290: "f32[512]" = torch.ops.aten.mul.Tensor(arg291_1, 0.999)
        mul_291: "f32[512]" = torch.ops.aten.mul.Tensor(arg292_1, 0.0010000000000000009);  arg292_1 = None
        add_145: "f32[512]" = torch.ops.aten.add.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
        mul_292: "f32[512]" = torch.ops.aten.mul.Tensor(arg293_1, 0.999)
        mul_293: "f32[512]" = torch.ops.aten.mul.Tensor(arg294_1, 0.0010000000000000009);  arg294_1 = None
        add_146: "f32[512]" = torch.ops.aten.add.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
        mul_294: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg295_1, 0.999)
        mul_295: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg296_1, 0.0010000000000000009);  arg296_1 = None
        add_147: "f32[2048, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_294, mul_295);  mul_294 = mul_295 = None
        mul_296: "f32[2048]" = torch.ops.aten.mul.Tensor(arg297_1, 0.999)
        mul_297: "f32[2048]" = torch.ops.aten.mul.Tensor(arg298_1, 0.0010000000000000009);  arg298_1 = None
        add_148: "f32[2048]" = torch.ops.aten.add.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
        mul_298: "f32[2048]" = torch.ops.aten.mul.Tensor(arg299_1, 0.999)
        mul_299: "f32[2048]" = torch.ops.aten.mul.Tensor(arg300_1, 0.0010000000000000009);  arg300_1 = None
        add_149: "f32[2048]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        mul_300: "f32[512, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(arg301_1, 0.999)
        mul_301: "f32[512, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(arg302_1, 0.0010000000000000009);  arg302_1 = None
        add_150: "f32[512, 2048, 1, 1]" = torch.ops.aten.add.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
        mul_302: "f32[512]" = torch.ops.aten.mul.Tensor(arg303_1, 0.999)
        mul_303: "f32[512]" = torch.ops.aten.mul.Tensor(arg304_1, 0.0010000000000000009);  arg304_1 = None
        add_151: "f32[512]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        mul_304: "f32[512]" = torch.ops.aten.mul.Tensor(arg305_1, 0.999)
        mul_305: "f32[512]" = torch.ops.aten.mul.Tensor(arg306_1, 0.0010000000000000009);  arg306_1 = None
        add_152: "f32[512]" = torch.ops.aten.add.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
        mul_306: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg307_1, 0.999)
        mul_307: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg308_1, 0.0010000000000000009);  arg308_1 = None
        add_153: "f32[512, 512, 3, 3]" = torch.ops.aten.add.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
        mul_308: "f32[512]" = torch.ops.aten.mul.Tensor(arg309_1, 0.999)
        mul_309: "f32[512]" = torch.ops.aten.mul.Tensor(arg310_1, 0.0010000000000000009);  arg310_1 = None
        add_154: "f32[512]" = torch.ops.aten.add.Tensor(mul_308, mul_309);  mul_308 = mul_309 = None
        mul_310: "f32[512]" = torch.ops.aten.mul.Tensor(arg311_1, 0.999)
        mul_311: "f32[512]" = torch.ops.aten.mul.Tensor(arg312_1, 0.0010000000000000009);  arg312_1 = None
        add_155: "f32[512]" = torch.ops.aten.add.Tensor(mul_310, mul_311);  mul_310 = mul_311 = None
        mul_312: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg313_1, 0.999)
        mul_313: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg314_1, 0.0010000000000000009);  arg314_1 = None
        add_156: "f32[2048, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        mul_314: "f32[2048]" = torch.ops.aten.mul.Tensor(arg315_1, 0.999)
        mul_315: "f32[2048]" = torch.ops.aten.mul.Tensor(arg316_1, 0.0010000000000000009);  arg316_1 = None
        add_157: "f32[2048]" = torch.ops.aten.add.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
        mul_316: "f32[2048]" = torch.ops.aten.mul.Tensor(arg317_1, 0.999)
        mul_317: "f32[2048]" = torch.ops.aten.mul.Tensor(arg318_1, 0.0010000000000000009);  arg318_1 = None
        add_158: "f32[2048]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        mul_318: "f32[128, 2048]" = torch.ops.aten.mul.Tensor(arg319_1, 0.999)
        mul_319: "f32[128, 2048]" = torch.ops.aten.mul.Tensor(arg320_1, 0.0010000000000000009);  arg320_1 = None
        add_159: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
        mul_320: "f32[128]" = torch.ops.aten.mul.Tensor(arg321_1, 0.999)
        mul_321: "f32[128]" = torch.ops.aten.mul.Tensor(arg322_1, 0.0010000000000000009);  arg322_1 = None
        add_160: "f32[128]" = torch.ops.aten.add.Tensor(mul_320, mul_321);  mul_320 = mul_321 = None
        copy_: "f32[64, 3, 7, 7]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy_ = None
        copy__1: "f32[64]" = torch.ops.aten.copy_.default(arg3_1, add_1);  arg3_1 = add_1 = copy__1 = None
        copy__2: "f32[64]" = torch.ops.aten.copy_.default(arg5_1, add_2);  arg5_1 = add_2 = copy__2 = None
        copy__3: "f32[64, 64, 1, 1]" = torch.ops.aten.copy_.default(arg7_1, add_3);  arg7_1 = add_3 = copy__3 = None
        copy__4: "f32[64]" = torch.ops.aten.copy_.default(arg9_1, add_4);  arg9_1 = add_4 = copy__4 = None
        copy__5: "f32[64]" = torch.ops.aten.copy_.default(arg11_1, add_5);  arg11_1 = add_5 = copy__5 = None
        copy__6: "f32[64, 64, 3, 3]" = torch.ops.aten.copy_.default(arg13_1, add_6);  arg13_1 = add_6 = copy__6 = None
        copy__7: "f32[64]" = torch.ops.aten.copy_.default(arg15_1, add_7);  arg15_1 = add_7 = copy__7 = None
        copy__8: "f32[64]" = torch.ops.aten.copy_.default(arg17_1, add_8);  arg17_1 = add_8 = copy__8 = None
        copy__9: "f32[256, 64, 1, 1]" = torch.ops.aten.copy_.default(arg19_1, add_9);  arg19_1 = add_9 = copy__9 = None
        copy__10: "f32[256]" = torch.ops.aten.copy_.default(arg21_1, add_10);  arg21_1 = add_10 = copy__10 = None
        copy__11: "f32[256]" = torch.ops.aten.copy_.default(arg23_1, add_11);  arg23_1 = add_11 = copy__11 = None
        copy__12: "f32[256, 64, 1, 1]" = torch.ops.aten.copy_.default(arg25_1, add_12);  arg25_1 = add_12 = copy__12 = None
        copy__13: "f32[256]" = torch.ops.aten.copy_.default(arg27_1, add_13);  arg27_1 = add_13 = copy__13 = None
        copy__14: "f32[256]" = torch.ops.aten.copy_.default(arg29_1, add_14);  arg29_1 = add_14 = copy__14 = None
        copy__15: "f32[64, 256, 1, 1]" = torch.ops.aten.copy_.default(arg31_1, add_15);  arg31_1 = add_15 = copy__15 = None
        copy__16: "f32[64]" = torch.ops.aten.copy_.default(arg33_1, add_16);  arg33_1 = add_16 = copy__16 = None
        copy__17: "f32[64]" = torch.ops.aten.copy_.default(arg35_1, add_17);  arg35_1 = add_17 = copy__17 = None
        copy__18: "f32[64, 64, 3, 3]" = torch.ops.aten.copy_.default(arg37_1, add_18);  arg37_1 = add_18 = copy__18 = None
        copy__19: "f32[64]" = torch.ops.aten.copy_.default(arg39_1, add_19);  arg39_1 = add_19 = copy__19 = None
        copy__20: "f32[64]" = torch.ops.aten.copy_.default(arg41_1, add_20);  arg41_1 = add_20 = copy__20 = None
        copy__21: "f32[256, 64, 1, 1]" = torch.ops.aten.copy_.default(arg43_1, add_21);  arg43_1 = add_21 = copy__21 = None
        copy__22: "f32[256]" = torch.ops.aten.copy_.default(arg45_1, add_22);  arg45_1 = add_22 = copy__22 = None
        copy__23: "f32[256]" = torch.ops.aten.copy_.default(arg47_1, add_23);  arg47_1 = add_23 = copy__23 = None
        copy__24: "f32[64, 256, 1, 1]" = torch.ops.aten.copy_.default(arg49_1, add_24);  arg49_1 = add_24 = copy__24 = None
        copy__25: "f32[64]" = torch.ops.aten.copy_.default(arg51_1, add_25);  arg51_1 = add_25 = copy__25 = None
        copy__26: "f32[64]" = torch.ops.aten.copy_.default(arg53_1, add_26);  arg53_1 = add_26 = copy__26 = None
        copy__27: "f32[64, 64, 3, 3]" = torch.ops.aten.copy_.default(arg55_1, add_27);  arg55_1 = add_27 = copy__27 = None
        copy__28: "f32[64]" = torch.ops.aten.copy_.default(arg57_1, add_28);  arg57_1 = add_28 = copy__28 = None
        copy__29: "f32[64]" = torch.ops.aten.copy_.default(arg59_1, add_29);  arg59_1 = add_29 = copy__29 = None
        copy__30: "f32[256, 64, 1, 1]" = torch.ops.aten.copy_.default(arg61_1, add_30);  arg61_1 = add_30 = copy__30 = None
        copy__31: "f32[256]" = torch.ops.aten.copy_.default(arg63_1, add_31);  arg63_1 = add_31 = copy__31 = None
        copy__32: "f32[256]" = torch.ops.aten.copy_.default(arg65_1, add_32);  arg65_1 = add_32 = copy__32 = None
        copy__33: "f32[128, 256, 1, 1]" = torch.ops.aten.copy_.default(arg67_1, add_33);  arg67_1 = add_33 = copy__33 = None
        copy__34: "f32[128]" = torch.ops.aten.copy_.default(arg69_1, add_34);  arg69_1 = add_34 = copy__34 = None
        copy__35: "f32[128]" = torch.ops.aten.copy_.default(arg71_1, add_35);  arg71_1 = add_35 = copy__35 = None
        copy__36: "f32[128, 128, 3, 3]" = torch.ops.aten.copy_.default(arg73_1, add_36);  arg73_1 = add_36 = copy__36 = None
        copy__37: "f32[128]" = torch.ops.aten.copy_.default(arg75_1, add_37);  arg75_1 = add_37 = copy__37 = None
        copy__38: "f32[128]" = torch.ops.aten.copy_.default(arg77_1, add_38);  arg77_1 = add_38 = copy__38 = None
        copy__39: "f32[512, 128, 1, 1]" = torch.ops.aten.copy_.default(arg79_1, add_39);  arg79_1 = add_39 = copy__39 = None
        copy__40: "f32[512]" = torch.ops.aten.copy_.default(arg81_1, add_40);  arg81_1 = add_40 = copy__40 = None
        copy__41: "f32[512]" = torch.ops.aten.copy_.default(arg83_1, add_41);  arg83_1 = add_41 = copy__41 = None
        copy__42: "f32[512, 256, 1, 1]" = torch.ops.aten.copy_.default(arg85_1, add_42);  arg85_1 = add_42 = copy__42 = None
        copy__43: "f32[512]" = torch.ops.aten.copy_.default(arg87_1, add_43);  arg87_1 = add_43 = copy__43 = None
        copy__44: "f32[512]" = torch.ops.aten.copy_.default(arg89_1, add_44);  arg89_1 = add_44 = copy__44 = None
        copy__45: "f32[128, 512, 1, 1]" = torch.ops.aten.copy_.default(arg91_1, add_45);  arg91_1 = add_45 = copy__45 = None
        copy__46: "f32[128]" = torch.ops.aten.copy_.default(arg93_1, add_46);  arg93_1 = add_46 = copy__46 = None
        copy__47: "f32[128]" = torch.ops.aten.copy_.default(arg95_1, add_47);  arg95_1 = add_47 = copy__47 = None
        copy__48: "f32[128, 128, 3, 3]" = torch.ops.aten.copy_.default(arg97_1, add_48);  arg97_1 = add_48 = copy__48 = None
        copy__49: "f32[128]" = torch.ops.aten.copy_.default(arg99_1, add_49);  arg99_1 = add_49 = copy__49 = None
        copy__50: "f32[128]" = torch.ops.aten.copy_.default(arg101_1, add_50);  arg101_1 = add_50 = copy__50 = None
        copy__51: "f32[512, 128, 1, 1]" = torch.ops.aten.copy_.default(arg103_1, add_51);  arg103_1 = add_51 = copy__51 = None
        copy__52: "f32[512]" = torch.ops.aten.copy_.default(arg105_1, add_52);  arg105_1 = add_52 = copy__52 = None
        copy__53: "f32[512]" = torch.ops.aten.copy_.default(arg107_1, add_53);  arg107_1 = add_53 = copy__53 = None
        copy__54: "f32[128, 512, 1, 1]" = torch.ops.aten.copy_.default(arg109_1, add_54);  arg109_1 = add_54 = copy__54 = None
        copy__55: "f32[128]" = torch.ops.aten.copy_.default(arg111_1, add_55);  arg111_1 = add_55 = copy__55 = None
        copy__56: "f32[128]" = torch.ops.aten.copy_.default(arg113_1, add_56);  arg113_1 = add_56 = copy__56 = None
        copy__57: "f32[128, 128, 3, 3]" = torch.ops.aten.copy_.default(arg115_1, add_57);  arg115_1 = add_57 = copy__57 = None
        copy__58: "f32[128]" = torch.ops.aten.copy_.default(arg117_1, add_58);  arg117_1 = add_58 = copy__58 = None
        copy__59: "f32[128]" = torch.ops.aten.copy_.default(arg119_1, add_59);  arg119_1 = add_59 = copy__59 = None
        copy__60: "f32[512, 128, 1, 1]" = torch.ops.aten.copy_.default(arg121_1, add_60);  arg121_1 = add_60 = copy__60 = None
        copy__61: "f32[512]" = torch.ops.aten.copy_.default(arg123_1, add_61);  arg123_1 = add_61 = copy__61 = None
        copy__62: "f32[512]" = torch.ops.aten.copy_.default(arg125_1, add_62);  arg125_1 = add_62 = copy__62 = None
        copy__63: "f32[128, 512, 1, 1]" = torch.ops.aten.copy_.default(arg127_1, add_63);  arg127_1 = add_63 = copy__63 = None
        copy__64: "f32[128]" = torch.ops.aten.copy_.default(arg129_1, add_64);  arg129_1 = add_64 = copy__64 = None
        copy__65: "f32[128]" = torch.ops.aten.copy_.default(arg131_1, add_65);  arg131_1 = add_65 = copy__65 = None
        copy__66: "f32[128, 128, 3, 3]" = torch.ops.aten.copy_.default(arg133_1, add_66);  arg133_1 = add_66 = copy__66 = None
        copy__67: "f32[128]" = torch.ops.aten.copy_.default(arg135_1, add_67);  arg135_1 = add_67 = copy__67 = None
        copy__68: "f32[128]" = torch.ops.aten.copy_.default(arg137_1, add_68);  arg137_1 = add_68 = copy__68 = None
        copy__69: "f32[512, 128, 1, 1]" = torch.ops.aten.copy_.default(arg139_1, add_69);  arg139_1 = add_69 = copy__69 = None
        copy__70: "f32[512]" = torch.ops.aten.copy_.default(arg141_1, add_70);  arg141_1 = add_70 = copy__70 = None
        copy__71: "f32[512]" = torch.ops.aten.copy_.default(arg143_1, add_71);  arg143_1 = add_71 = copy__71 = None
        copy__72: "f32[256, 512, 1, 1]" = torch.ops.aten.copy_.default(arg145_1, add_72);  arg145_1 = add_72 = copy__72 = None
        copy__73: "f32[256]" = torch.ops.aten.copy_.default(arg147_1, add_73);  arg147_1 = add_73 = copy__73 = None
        copy__74: "f32[256]" = torch.ops.aten.copy_.default(arg149_1, add_74);  arg149_1 = add_74 = copy__74 = None
        copy__75: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg151_1, add_75);  arg151_1 = add_75 = copy__75 = None
        copy__76: "f32[256]" = torch.ops.aten.copy_.default(arg153_1, add_76);  arg153_1 = add_76 = copy__76 = None
        copy__77: "f32[256]" = torch.ops.aten.copy_.default(arg155_1, add_77);  arg155_1 = add_77 = copy__77 = None
        copy__78: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg157_1, add_78);  arg157_1 = add_78 = copy__78 = None
        copy__79: "f32[1024]" = torch.ops.aten.copy_.default(arg159_1, add_79);  arg159_1 = add_79 = copy__79 = None
        copy__80: "f32[1024]" = torch.ops.aten.copy_.default(arg161_1, add_80);  arg161_1 = add_80 = copy__80 = None
        copy__81: "f32[1024, 512, 1, 1]" = torch.ops.aten.copy_.default(arg163_1, add_81);  arg163_1 = add_81 = copy__81 = None
        copy__82: "f32[1024]" = torch.ops.aten.copy_.default(arg165_1, add_82);  arg165_1 = add_82 = copy__82 = None
        copy__83: "f32[1024]" = torch.ops.aten.copy_.default(arg167_1, add_83);  arg167_1 = add_83 = copy__83 = None
        copy__84: "f32[256, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg169_1, add_84);  arg169_1 = add_84 = copy__84 = None
        copy__85: "f32[256]" = torch.ops.aten.copy_.default(arg171_1, add_85);  arg171_1 = add_85 = copy__85 = None
        copy__86: "f32[256]" = torch.ops.aten.copy_.default(arg173_1, add_86);  arg173_1 = add_86 = copy__86 = None
        copy__87: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg175_1, add_87);  arg175_1 = add_87 = copy__87 = None
        copy__88: "f32[256]" = torch.ops.aten.copy_.default(arg177_1, add_88);  arg177_1 = add_88 = copy__88 = None
        copy__89: "f32[256]" = torch.ops.aten.copy_.default(arg179_1, add_89);  arg179_1 = add_89 = copy__89 = None
        copy__90: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg181_1, add_90);  arg181_1 = add_90 = copy__90 = None
        copy__91: "f32[1024]" = torch.ops.aten.copy_.default(arg183_1, add_91);  arg183_1 = add_91 = copy__91 = None
        copy__92: "f32[1024]" = torch.ops.aten.copy_.default(arg185_1, add_92);  arg185_1 = add_92 = copy__92 = None
        copy__93: "f32[256, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg187_1, add_93);  arg187_1 = add_93 = copy__93 = None
        copy__94: "f32[256]" = torch.ops.aten.copy_.default(arg189_1, add_94);  arg189_1 = add_94 = copy__94 = None
        copy__95: "f32[256]" = torch.ops.aten.copy_.default(arg191_1, add_95);  arg191_1 = add_95 = copy__95 = None
        copy__96: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg193_1, add_96);  arg193_1 = add_96 = copy__96 = None
        copy__97: "f32[256]" = torch.ops.aten.copy_.default(arg195_1, add_97);  arg195_1 = add_97 = copy__97 = None
        copy__98: "f32[256]" = torch.ops.aten.copy_.default(arg197_1, add_98);  arg197_1 = add_98 = copy__98 = None
        copy__99: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg199_1, add_99);  arg199_1 = add_99 = copy__99 = None
        copy__100: "f32[1024]" = torch.ops.aten.copy_.default(arg201_1, add_100);  arg201_1 = add_100 = copy__100 = None
        copy__101: "f32[1024]" = torch.ops.aten.copy_.default(arg203_1, add_101);  arg203_1 = add_101 = copy__101 = None
        copy__102: "f32[256, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg205_1, add_102);  arg205_1 = add_102 = copy__102 = None
        copy__103: "f32[256]" = torch.ops.aten.copy_.default(arg207_1, add_103);  arg207_1 = add_103 = copy__103 = None
        copy__104: "f32[256]" = torch.ops.aten.copy_.default(arg209_1, add_104);  arg209_1 = add_104 = copy__104 = None
        copy__105: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg211_1, add_105);  arg211_1 = add_105 = copy__105 = None
        copy__106: "f32[256]" = torch.ops.aten.copy_.default(arg213_1, add_106);  arg213_1 = add_106 = copy__106 = None
        copy__107: "f32[256]" = torch.ops.aten.copy_.default(arg215_1, add_107);  arg215_1 = add_107 = copy__107 = None
        copy__108: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg217_1, add_108);  arg217_1 = add_108 = copy__108 = None
        copy__109: "f32[1024]" = torch.ops.aten.copy_.default(arg219_1, add_109);  arg219_1 = add_109 = copy__109 = None
        copy__110: "f32[1024]" = torch.ops.aten.copy_.default(arg221_1, add_110);  arg221_1 = add_110 = copy__110 = None
        copy__111: "f32[256, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg223_1, add_111);  arg223_1 = add_111 = copy__111 = None
        copy__112: "f32[256]" = torch.ops.aten.copy_.default(arg225_1, add_112);  arg225_1 = add_112 = copy__112 = None
        copy__113: "f32[256]" = torch.ops.aten.copy_.default(arg227_1, add_113);  arg227_1 = add_113 = copy__113 = None
        copy__114: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg229_1, add_114);  arg229_1 = add_114 = copy__114 = None
        copy__115: "f32[256]" = torch.ops.aten.copy_.default(arg231_1, add_115);  arg231_1 = add_115 = copy__115 = None
        copy__116: "f32[256]" = torch.ops.aten.copy_.default(arg233_1, add_116);  arg233_1 = add_116 = copy__116 = None
        copy__117: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg235_1, add_117);  arg235_1 = add_117 = copy__117 = None
        copy__118: "f32[1024]" = torch.ops.aten.copy_.default(arg237_1, add_118);  arg237_1 = add_118 = copy__118 = None
        copy__119: "f32[1024]" = torch.ops.aten.copy_.default(arg239_1, add_119);  arg239_1 = add_119 = copy__119 = None
        copy__120: "f32[256, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg241_1, add_120);  arg241_1 = add_120 = copy__120 = None
        copy__121: "f32[256]" = torch.ops.aten.copy_.default(arg243_1, add_121);  arg243_1 = add_121 = copy__121 = None
        copy__122: "f32[256]" = torch.ops.aten.copy_.default(arg245_1, add_122);  arg245_1 = add_122 = copy__122 = None
        copy__123: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg247_1, add_123);  arg247_1 = add_123 = copy__123 = None
        copy__124: "f32[256]" = torch.ops.aten.copy_.default(arg249_1, add_124);  arg249_1 = add_124 = copy__124 = None
        copy__125: "f32[256]" = torch.ops.aten.copy_.default(arg251_1, add_125);  arg251_1 = add_125 = copy__125 = None
        copy__126: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg253_1, add_126);  arg253_1 = add_126 = copy__126 = None
        copy__127: "f32[1024]" = torch.ops.aten.copy_.default(arg255_1, add_127);  arg255_1 = add_127 = copy__127 = None
        copy__128: "f32[1024]" = torch.ops.aten.copy_.default(arg257_1, add_128);  arg257_1 = add_128 = copy__128 = None
        copy__129: "f32[512, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg259_1, add_129);  arg259_1 = add_129 = copy__129 = None
        copy__130: "f32[512]" = torch.ops.aten.copy_.default(arg261_1, add_130);  arg261_1 = add_130 = copy__130 = None
        copy__131: "f32[512]" = torch.ops.aten.copy_.default(arg263_1, add_131);  arg263_1 = add_131 = copy__131 = None
        copy__132: "f32[512, 512, 3, 3]" = torch.ops.aten.copy_.default(arg265_1, add_132);  arg265_1 = add_132 = copy__132 = None
        copy__133: "f32[512]" = torch.ops.aten.copy_.default(arg267_1, add_133);  arg267_1 = add_133 = copy__133 = None
        copy__134: "f32[512]" = torch.ops.aten.copy_.default(arg269_1, add_134);  arg269_1 = add_134 = copy__134 = None
        copy__135: "f32[2048, 512, 1, 1]" = torch.ops.aten.copy_.default(arg271_1, add_135);  arg271_1 = add_135 = copy__135 = None
        copy__136: "f32[2048]" = torch.ops.aten.copy_.default(arg273_1, add_136);  arg273_1 = add_136 = copy__136 = None
        copy__137: "f32[2048]" = torch.ops.aten.copy_.default(arg275_1, add_137);  arg275_1 = add_137 = copy__137 = None
        copy__138: "f32[2048, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg277_1, add_138);  arg277_1 = add_138 = copy__138 = None
        copy__139: "f32[2048]" = torch.ops.aten.copy_.default(arg279_1, add_139);  arg279_1 = add_139 = copy__139 = None
        copy__140: "f32[2048]" = torch.ops.aten.copy_.default(arg281_1, add_140);  arg281_1 = add_140 = copy__140 = None
        copy__141: "f32[512, 2048, 1, 1]" = torch.ops.aten.copy_.default(arg283_1, add_141);  arg283_1 = add_141 = copy__141 = None
        copy__142: "f32[512]" = torch.ops.aten.copy_.default(arg285_1, add_142);  arg285_1 = add_142 = copy__142 = None
        copy__143: "f32[512]" = torch.ops.aten.copy_.default(arg287_1, add_143);  arg287_1 = add_143 = copy__143 = None
        copy__144: "f32[512, 512, 3, 3]" = torch.ops.aten.copy_.default(arg289_1, add_144);  arg289_1 = add_144 = copy__144 = None
        copy__145: "f32[512]" = torch.ops.aten.copy_.default(arg291_1, add_145);  arg291_1 = add_145 = copy__145 = None
        copy__146: "f32[512]" = torch.ops.aten.copy_.default(arg293_1, add_146);  arg293_1 = add_146 = copy__146 = None
        copy__147: "f32[2048, 512, 1, 1]" = torch.ops.aten.copy_.default(arg295_1, add_147);  arg295_1 = add_147 = copy__147 = None
        copy__148: "f32[2048]" = torch.ops.aten.copy_.default(arg297_1, add_148);  arg297_1 = add_148 = copy__148 = None
        copy__149: "f32[2048]" = torch.ops.aten.copy_.default(arg299_1, add_149);  arg299_1 = add_149 = copy__149 = None
        copy__150: "f32[512, 2048, 1, 1]" = torch.ops.aten.copy_.default(arg301_1, add_150);  arg301_1 = add_150 = copy__150 = None
        copy__151: "f32[512]" = torch.ops.aten.copy_.default(arg303_1, add_151);  arg303_1 = add_151 = copy__151 = None
        copy__152: "f32[512]" = torch.ops.aten.copy_.default(arg305_1, add_152);  arg305_1 = add_152 = copy__152 = None
        copy__153: "f32[512, 512, 3, 3]" = torch.ops.aten.copy_.default(arg307_1, add_153);  arg307_1 = add_153 = copy__153 = None
        copy__154: "f32[512]" = torch.ops.aten.copy_.default(arg309_1, add_154);  arg309_1 = add_154 = copy__154 = None
        copy__155: "f32[512]" = torch.ops.aten.copy_.default(arg311_1, add_155);  arg311_1 = add_155 = copy__155 = None
        copy__156: "f32[2048, 512, 1, 1]" = torch.ops.aten.copy_.default(arg313_1, add_156);  arg313_1 = add_156 = copy__156 = None
        copy__157: "f32[2048]" = torch.ops.aten.copy_.default(arg315_1, add_157);  arg315_1 = add_157 = copy__157 = None
        copy__158: "f32[2048]" = torch.ops.aten.copy_.default(arg317_1, add_158);  arg317_1 = add_158 = copy__158 = None
        copy__159: "f32[128, 2048]" = torch.ops.aten.copy_.default(arg319_1, add_159);  arg319_1 = add_159 = copy__159 = None
        copy__160: "f32[128]" = torch.ops.aten.copy_.default(arg321_1, add_160);  arg321_1 = add_160 = copy__160 = None
        return (div,)
