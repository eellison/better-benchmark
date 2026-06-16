class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32, 3, 3, 3][27, 1, 9, 3]cuda:0", arg1_1: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", arg2_1: "bf16[32][1]cuda:0", arg3_1: "bf16[32][1]cuda:0", arg4_1: "bf16[32][1]cuda:0", arg5_1: "bf16[32][1]cuda:0", arg6_1: "bf16[32, 1, 3, 3][9, 1, 3, 1]cuda:0", arg7_1: "bf16[32][1]cuda:0", arg8_1: "bf16[32][1]cuda:0", arg9_1: "bf16[32][1]cuda:0", arg10_1: "bf16[32][1]cuda:0", arg11_1: "bf16[8, 32, 1, 1][32, 1, 32, 32]cuda:0", arg12_1: "bf16[8][1]cuda:0", arg13_1: "bf16[32, 8, 1, 1][8, 1, 8, 8]cuda:0", arg14_1: "bf16[32][1]cuda:0", arg15_1: "bf16[16, 32, 1, 1][32, 1, 32, 32]cuda:0", arg16_1: "bf16[16][1]cuda:0", arg17_1: "bf16[16][1]cuda:0", arg18_1: "bf16[16][1]cuda:0", arg19_1: "bf16[16][1]cuda:0", arg20_1: "bf16[96, 16, 1, 1][16, 1, 16, 16]cuda:0", arg21_1: "bf16[96][1]cuda:0", arg22_1: "bf16[96][1]cuda:0", arg23_1: "bf16[96][1]cuda:0", arg24_1: "bf16[96][1]cuda:0", arg25_1: "bf16[96, 1, 3, 3][9, 1, 3, 1]cuda:0", arg26_1: "bf16[96][1]cuda:0", arg27_1: "bf16[96][1]cuda:0", arg28_1: "bf16[96][1]cuda:0", arg29_1: "bf16[96][1]cuda:0", arg30_1: "bf16[4, 96, 1, 1][96, 1, 96, 96]cuda:0", arg31_1: "bf16[4][1]cuda:0", arg32_1: "bf16[96, 4, 1, 1][4, 1, 4, 4]cuda:0", arg33_1: "bf16[96][1]cuda:0", arg34_1: "bf16[24, 96, 1, 1][96, 1, 96, 96]cuda:0", arg35_1: "bf16[24][1]cuda:0", arg36_1: "bf16[24][1]cuda:0", arg37_1: "bf16[24][1]cuda:0", arg38_1: "bf16[24][1]cuda:0", arg39_1: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0", arg40_1: "bf16[144][1]cuda:0", arg41_1: "bf16[144][1]cuda:0", arg42_1: "bf16[144][1]cuda:0", arg43_1: "bf16[144][1]cuda:0", arg44_1: "bf16[144, 1, 3, 3][9, 1, 3, 1]cuda:0", arg45_1: "bf16[144][1]cuda:0", arg46_1: "bf16[144][1]cuda:0", arg47_1: "bf16[144][1]cuda:0", arg48_1: "bf16[144][1]cuda:0", arg49_1: "bf16[6, 144, 1, 1][144, 1, 144, 144]cuda:0", arg50_1: "bf16[6][1]cuda:0", arg51_1: "bf16[144, 6, 1, 1][6, 1, 6, 6]cuda:0", arg52_1: "bf16[144][1]cuda:0", arg53_1: "bf16[24, 144, 1, 1][144, 1, 144, 144]cuda:0", arg54_1: "bf16[24][1]cuda:0", arg55_1: "bf16[24][1]cuda:0", arg56_1: "bf16[24][1]cuda:0", arg57_1: "bf16[24][1]cuda:0", arg58_1: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0", arg59_1: "bf16[144][1]cuda:0", arg60_1: "bf16[144][1]cuda:0", arg61_1: "bf16[144][1]cuda:0", arg62_1: "bf16[144][1]cuda:0", arg63_1: "bf16[144, 1, 5, 5][25, 1, 5, 1]cuda:0", arg64_1: "bf16[144][1]cuda:0", arg65_1: "bf16[144][1]cuda:0", arg66_1: "bf16[144][1]cuda:0", arg67_1: "bf16[144][1]cuda:0", arg68_1: "bf16[6, 144, 1, 1][144, 1, 144, 144]cuda:0", arg69_1: "bf16[6][1]cuda:0", arg70_1: "bf16[144, 6, 1, 1][6, 1, 6, 6]cuda:0", arg71_1: "bf16[144][1]cuda:0", arg72_1: "bf16[40, 144, 1, 1][144, 1, 144, 144]cuda:0", arg73_1: "bf16[40][1]cuda:0", arg74_1: "bf16[40][1]cuda:0", arg75_1: "bf16[40][1]cuda:0", arg76_1: "bf16[40][1]cuda:0", arg77_1: "bf16[240, 40, 1, 1][40, 1, 40, 40]cuda:0", arg78_1: "bf16[240][1]cuda:0", arg79_1: "bf16[240][1]cuda:0", arg80_1: "bf16[240][1]cuda:0", arg81_1: "bf16[240][1]cuda:0", arg82_1: "bf16[240, 1, 5, 5][25, 1, 5, 1]cuda:0", arg83_1: "bf16[240][1]cuda:0", arg84_1: "bf16[240][1]cuda:0", arg85_1: "bf16[240][1]cuda:0", arg86_1: "bf16[240][1]cuda:0", arg87_1: "bf16[10, 240, 1, 1][240, 1, 240, 240]cuda:0", arg88_1: "bf16[10][1]cuda:0", arg89_1: "bf16[240, 10, 1, 1][10, 1, 10, 10]cuda:0", arg90_1: "bf16[240][1]cuda:0", arg91_1: "bf16[40, 240, 1, 1][240, 1, 240, 240]cuda:0", arg92_1: "bf16[40][1]cuda:0", arg93_1: "bf16[40][1]cuda:0", arg94_1: "bf16[40][1]cuda:0", arg95_1: "bf16[40][1]cuda:0", arg96_1: "bf16[240, 40, 1, 1][40, 1, 40, 40]cuda:0", arg97_1: "bf16[240][1]cuda:0", arg98_1: "bf16[240][1]cuda:0", arg99_1: "bf16[240][1]cuda:0", arg100_1: "bf16[240][1]cuda:0", arg101_1: "bf16[240, 1, 3, 3][9, 1, 3, 1]cuda:0", arg102_1: "bf16[240][1]cuda:0", arg103_1: "bf16[240][1]cuda:0", arg104_1: "bf16[240][1]cuda:0", arg105_1: "bf16[240][1]cuda:0", arg106_1: "bf16[10, 240, 1, 1][240, 1, 240, 240]cuda:0", arg107_1: "bf16[10][1]cuda:0", arg108_1: "bf16[240, 10, 1, 1][10, 1, 10, 10]cuda:0", arg109_1: "bf16[240][1]cuda:0", arg110_1: "bf16[80, 240, 1, 1][240, 1, 240, 240]cuda:0", arg111_1: "bf16[80][1]cuda:0", arg112_1: "bf16[80][1]cuda:0", arg113_1: "bf16[80][1]cuda:0", arg114_1: "bf16[80][1]cuda:0", arg115_1: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0", arg116_1: "bf16[480][1]cuda:0", arg117_1: "bf16[480][1]cuda:0", arg118_1: "bf16[480][1]cuda:0", arg119_1: "bf16[480][1]cuda:0", arg120_1: "bf16[480, 1, 3, 3][9, 1, 3, 1]cuda:0", arg121_1: "bf16[480][1]cuda:0", arg122_1: "bf16[480][1]cuda:0", arg123_1: "bf16[480][1]cuda:0", arg124_1: "bf16[480][1]cuda:0", arg125_1: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0", arg126_1: "bf16[20][1]cuda:0", arg127_1: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0", arg128_1: "bf16[480][1]cuda:0", arg129_1: "bf16[80, 480, 1, 1][480, 1, 480, 480]cuda:0", arg130_1: "bf16[80][1]cuda:0", arg131_1: "bf16[80][1]cuda:0", arg132_1: "bf16[80][1]cuda:0", arg133_1: "bf16[80][1]cuda:0", arg134_1: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0", arg135_1: "bf16[480][1]cuda:0", arg136_1: "bf16[480][1]cuda:0", arg137_1: "bf16[480][1]cuda:0", arg138_1: "bf16[480][1]cuda:0", arg139_1: "bf16[480, 1, 3, 3][9, 1, 3, 1]cuda:0", arg140_1: "bf16[480][1]cuda:0", arg141_1: "bf16[480][1]cuda:0", arg142_1: "bf16[480][1]cuda:0", arg143_1: "bf16[480][1]cuda:0", arg144_1: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0", arg145_1: "bf16[20][1]cuda:0", arg146_1: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0", arg147_1: "bf16[480][1]cuda:0", arg148_1: "bf16[80, 480, 1, 1][480, 1, 480, 480]cuda:0", arg149_1: "bf16[80][1]cuda:0", arg150_1: "bf16[80][1]cuda:0", arg151_1: "bf16[80][1]cuda:0", arg152_1: "bf16[80][1]cuda:0", arg153_1: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0", arg154_1: "bf16[480][1]cuda:0", arg155_1: "bf16[480][1]cuda:0", arg156_1: "bf16[480][1]cuda:0", arg157_1: "bf16[480][1]cuda:0", arg158_1: "bf16[480, 1, 5, 5][25, 1, 5, 1]cuda:0", arg159_1: "bf16[480][1]cuda:0", arg160_1: "bf16[480][1]cuda:0", arg161_1: "bf16[480][1]cuda:0", arg162_1: "bf16[480][1]cuda:0", arg163_1: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0", arg164_1: "bf16[20][1]cuda:0", arg165_1: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0", arg166_1: "bf16[480][1]cuda:0", arg167_1: "bf16[112, 480, 1, 1][480, 1, 480, 480]cuda:0", arg168_1: "bf16[112][1]cuda:0", arg169_1: "bf16[112][1]cuda:0", arg170_1: "bf16[112][1]cuda:0", arg171_1: "bf16[112][1]cuda:0", arg172_1: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0", arg173_1: "bf16[672][1]cuda:0", arg174_1: "bf16[672][1]cuda:0", arg175_1: "bf16[672][1]cuda:0", arg176_1: "bf16[672][1]cuda:0", arg177_1: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0", arg178_1: "bf16[672][1]cuda:0", arg179_1: "bf16[672][1]cuda:0", arg180_1: "bf16[672][1]cuda:0", arg181_1: "bf16[672][1]cuda:0", arg182_1: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0", arg183_1: "bf16[28][1]cuda:0", arg184_1: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0", arg185_1: "bf16[672][1]cuda:0", arg186_1: "bf16[112, 672, 1, 1][672, 1, 672, 672]cuda:0", arg187_1: "bf16[112][1]cuda:0", arg188_1: "bf16[112][1]cuda:0", arg189_1: "bf16[112][1]cuda:0", arg190_1: "bf16[112][1]cuda:0", arg191_1: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0", arg192_1: "bf16[672][1]cuda:0", arg193_1: "bf16[672][1]cuda:0", arg194_1: "bf16[672][1]cuda:0", arg195_1: "bf16[672][1]cuda:0", arg196_1: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0", arg197_1: "bf16[672][1]cuda:0", arg198_1: "bf16[672][1]cuda:0", arg199_1: "bf16[672][1]cuda:0", arg200_1: "bf16[672][1]cuda:0", arg201_1: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0", arg202_1: "bf16[28][1]cuda:0", arg203_1: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0", arg204_1: "bf16[672][1]cuda:0", arg205_1: "bf16[112, 672, 1, 1][672, 1, 672, 672]cuda:0", arg206_1: "bf16[112][1]cuda:0", arg207_1: "bf16[112][1]cuda:0", arg208_1: "bf16[112][1]cuda:0", arg209_1: "bf16[112][1]cuda:0", arg210_1: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0", arg211_1: "bf16[672][1]cuda:0", arg212_1: "bf16[672][1]cuda:0", arg213_1: "bf16[672][1]cuda:0", arg214_1: "bf16[672][1]cuda:0", arg215_1: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0", arg216_1: "bf16[672][1]cuda:0", arg217_1: "bf16[672][1]cuda:0", arg218_1: "bf16[672][1]cuda:0", arg219_1: "bf16[672][1]cuda:0", arg220_1: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0", arg221_1: "bf16[28][1]cuda:0", arg222_1: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0", arg223_1: "bf16[672][1]cuda:0", arg224_1: "bf16[192, 672, 1, 1][672, 1, 672, 672]cuda:0", arg225_1: "bf16[192][1]cuda:0", arg226_1: "bf16[192][1]cuda:0", arg227_1: "bf16[192][1]cuda:0", arg228_1: "bf16[192][1]cuda:0", arg229_1: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", arg230_1: "bf16[1152][1]cuda:0", arg231_1: "bf16[1152][1]cuda:0", arg232_1: "bf16[1152][1]cuda:0", arg233_1: "bf16[1152][1]cuda:0", arg234_1: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0", arg235_1: "bf16[1152][1]cuda:0", arg236_1: "bf16[1152][1]cuda:0", arg237_1: "bf16[1152][1]cuda:0", arg238_1: "bf16[1152][1]cuda:0", arg239_1: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", arg240_1: "bf16[48][1]cuda:0", arg241_1: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", arg242_1: "bf16[1152][1]cuda:0", arg243_1: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", arg244_1: "bf16[192][1]cuda:0", arg245_1: "bf16[192][1]cuda:0", arg246_1: "bf16[192][1]cuda:0", arg247_1: "bf16[192][1]cuda:0", arg248_1: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", arg249_1: "bf16[1152][1]cuda:0", arg250_1: "bf16[1152][1]cuda:0", arg251_1: "bf16[1152][1]cuda:0", arg252_1: "bf16[1152][1]cuda:0", arg253_1: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0", arg254_1: "bf16[1152][1]cuda:0", arg255_1: "bf16[1152][1]cuda:0", arg256_1: "bf16[1152][1]cuda:0", arg257_1: "bf16[1152][1]cuda:0", arg258_1: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", arg259_1: "bf16[48][1]cuda:0", arg260_1: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", arg261_1: "bf16[1152][1]cuda:0", arg262_1: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", arg263_1: "bf16[192][1]cuda:0", arg264_1: "bf16[192][1]cuda:0", arg265_1: "bf16[192][1]cuda:0", arg266_1: "bf16[192][1]cuda:0", arg267_1: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", arg268_1: "bf16[1152][1]cuda:0", arg269_1: "bf16[1152][1]cuda:0", arg270_1: "bf16[1152][1]cuda:0", arg271_1: "bf16[1152][1]cuda:0", arg272_1: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0", arg273_1: "bf16[1152][1]cuda:0", arg274_1: "bf16[1152][1]cuda:0", arg275_1: "bf16[1152][1]cuda:0", arg276_1: "bf16[1152][1]cuda:0", arg277_1: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", arg278_1: "bf16[48][1]cuda:0", arg279_1: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", arg280_1: "bf16[1152][1]cuda:0", arg281_1: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", arg282_1: "bf16[192][1]cuda:0", arg283_1: "bf16[192][1]cuda:0", arg284_1: "bf16[192][1]cuda:0", arg285_1: "bf16[192][1]cuda:0", arg286_1: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", arg287_1: "bf16[1152][1]cuda:0", arg288_1: "bf16[1152][1]cuda:0", arg289_1: "bf16[1152][1]cuda:0", arg290_1: "bf16[1152][1]cuda:0", arg291_1: "bf16[1152, 1, 3, 3][9, 1, 3, 1]cuda:0", arg292_1: "bf16[1152][1]cuda:0", arg293_1: "bf16[1152][1]cuda:0", arg294_1: "bf16[1152][1]cuda:0", arg295_1: "bf16[1152][1]cuda:0", arg296_1: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", arg297_1: "bf16[48][1]cuda:0", arg298_1: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", arg299_1: "bf16[1152][1]cuda:0", arg300_1: "bf16[320, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", arg301_1: "bf16[320][1]cuda:0", arg302_1: "bf16[320][1]cuda:0", arg303_1: "bf16[320][1]cuda:0", arg304_1: "bf16[320][1]cuda:0", arg305_1: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", arg306_1: "bf16[1280][1]cuda:0", arg307_1: "bf16[1280][1]cuda:0", arg308_1: "bf16[1280][1]cuda:0", arg309_1: "bf16[1280][1]cuda:0", arg310_1: "bf16[1000, 1280][1280, 1]cuda:0", arg311_1: "bf16[1000][1]cuda:0"):
        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "bf16[128, 3, 225, 225][151875, 1, 675, 3]cuda:0" = torch.ops.aten.constant_pad_nd.default(arg1_1, [0, 1, 0, 1], 0.0);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.convolution.default(constant_pad_nd, arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  constant_pad_nd = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        unsqueeze: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        convert_element_type_1: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1, 0.001);  convert_element_type_1 = None
        sqrt: "f32[32][1]cuda:0" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[32][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "bf16[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5: "bf16[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "bf16[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7: "bf16[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_3: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        neg: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.neg.default(convert_element_type_3)
        exp: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.exp.default(neg);  neg = None
        add_2: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_3, add_2);  convert_element_type_3 = add_2 = None
        convert_element_type_4: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:223 in forward, code: x = self.conv_dw(x)
        convolution_1: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_4, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  convert_element_type_4 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_5: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        unsqueeze_8: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, -1);  convert_element_type_5 = None
        unsqueeze_9: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        convert_element_type_6: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        add_3: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_6, 0.001);  convert_element_type_6 = None
        sqrt_1: "f32[32][1]cuda:0" = torch.ops.aten.sqrt.default(add_3);  add_3 = None
        reciprocal_1: "f32[32][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_4: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "bf16[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_13: "bf16[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14: "bf16[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_15: "bf16[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_4: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
        convert_element_type_7: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_8: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        neg_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.neg.default(convert_element_type_8)
        exp_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_5: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_8, add_5);  convert_element_type_8 = add_5 = None
        convert_element_type_9: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean: "bf16[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_9, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_2: "bf16[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.convolution.default(mean, arg11_1, arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg11_1 = arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_10: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        neg_2: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.neg.default(convert_element_type_10)
        exp_2: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_6: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_10, add_6);  convert_element_type_10 = add_6 = None
        convert_element_type_11: "bf16[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_3: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_11, arg13_1, arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_11 = arg13_1 = arg14_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.sigmoid.default(convolution_3);  convolution_3 = None
        mul_6: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_9, sigmoid);  convert_element_type_9 = sigmoid = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:227 in forward, code: x = self.conv_pw(x)
        convolution_4: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.convolution.default(mul_6, arg15_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_6 = arg15_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_12: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg16_1, torch.float32);  arg16_1 = None
        unsqueeze_16: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
        unsqueeze_17: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_2: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_17);  convolution_4 = unsqueeze_17 = None
        convert_element_type_13: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        add_7: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_13, 0.001);  convert_element_type_13 = None
        sqrt_2: "f32[16][1]cuda:0" = torch.ops.aten.sqrt.default(add_7);  add_7 = None
        reciprocal_2: "f32[16][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_7: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_7, -1);  mul_7 = None
        unsqueeze_19: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_8: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_21: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_9: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_21);  mul_8 = unsqueeze_21 = None
        unsqueeze_22: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_23: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_8: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_9, unsqueeze_23);  mul_9 = unsqueeze_23 = None
        convert_element_type_14: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_5: "bf16[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_14, arg20_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_14 = arg20_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_15: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg21_1, torch.float32);  arg21_1 = None
        unsqueeze_24: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_15, -1);  convert_element_type_15 = None
        unsqueeze_25: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_3: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_25);  convolution_5 = unsqueeze_25 = None
        convert_element_type_16: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg22_1, torch.float32);  arg22_1 = None
        add_9: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_16, 0.001);  convert_element_type_16 = None
        sqrt_3: "f32[96][1]cuda:0" = torch.ops.aten.sqrt.default(add_9);  add_9 = None
        reciprocal_3: "f32[96][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_10: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_10, -1);  mul_10 = None
        unsqueeze_27: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_11: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_29: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_12: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_29);  mul_11 = unsqueeze_29 = None
        unsqueeze_30: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_31: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_10: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_12, unsqueeze_31);  mul_12 = unsqueeze_31 = None
        convert_element_type_17: "bf16[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_18: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_17, torch.float32);  convert_element_type_17 = None
        neg_3: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.neg.default(convert_element_type_18)
        exp_3: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_11: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_18, add_11);  convert_element_type_18 = add_11 = None
        convert_element_type_19: "bf16[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "bf16[128, 96, 113, 113][1225824, 1, 10848, 96]cuda:0" = torch.ops.aten.constant_pad_nd.default(convert_element_type_19, [0, 1, 0, 1], 0.0);  convert_element_type_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_6: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.convolution.default(constant_pad_nd_1, arg25_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 96);  constant_pad_nd_1 = arg25_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_20: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg26_1, torch.float32);  arg26_1 = None
        unsqueeze_32: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
        unsqueeze_33: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        sub_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_33);  convolution_6 = unsqueeze_33 = None
        convert_element_type_21: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg27_1, torch.float32);  arg27_1 = None
        add_12: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_21, 0.001);  convert_element_type_21 = None
        sqrt_4: "f32[96][1]cuda:0" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
        reciprocal_4: "f32[96][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_13: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_34: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_13, -1);  mul_13 = None
        unsqueeze_35: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_14: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
        unsqueeze_36: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_37: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_15: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_37);  mul_14 = unsqueeze_37 = None
        unsqueeze_38: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_39: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_13: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_15, unsqueeze_39);  mul_15 = unsqueeze_39 = None
        convert_element_type_22: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_23: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        neg_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.neg.default(convert_element_type_23)
        exp_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_14: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_23, add_14);  convert_element_type_23 = add_14 = None
        convert_element_type_24: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_1: "bf16[128, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_24, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_7: "bf16[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.convolution.default(mean_1, arg30_1, arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg30_1 = arg31_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_25: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        neg_5: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.neg.default(convert_element_type_25)
        exp_5: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_15: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_25, add_15);  convert_element_type_25 = add_15 = None
        convert_element_type_26: "bf16[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_8: "bf16[128, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_26, arg32_1, arg33_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_26 = arg32_1 = arg33_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_1: "bf16[128, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.aten.sigmoid.default(convolution_8);  convolution_8 = None
        mul_16: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, sigmoid_1);  convert_element_type_24 = sigmoid_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_9: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.convolution.default(mul_16, arg34_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_16 = arg34_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_27: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg35_1, torch.float32);  arg35_1 = None
        unsqueeze_40: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_27, -1);  convert_element_type_27 = None
        unsqueeze_41: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        sub_5: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_41);  convolution_9 = unsqueeze_41 = None
        convert_element_type_28: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg36_1, torch.float32);  arg36_1 = None
        add_16: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_28, 0.001);  convert_element_type_28 = None
        sqrt_5: "f32[24][1]cuda:0" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
        reciprocal_5: "f32[24][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_17: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_42: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_17, -1);  mul_17 = None
        unsqueeze_43: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_18: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
        unsqueeze_44: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_45: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_19: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, unsqueeze_45);  mul_18 = unsqueeze_45 = None
        unsqueeze_46: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_47: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_17: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(mul_19, unsqueeze_47);  mul_19 = unsqueeze_47 = None
        convert_element_type_29: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_10: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_29, arg39_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg39_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_30: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg40_1, torch.float32);  arg40_1 = None
        unsqueeze_48: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
        unsqueeze_49: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        sub_6: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_49);  convolution_10 = unsqueeze_49 = None
        convert_element_type_31: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg41_1, torch.float32);  arg41_1 = None
        add_18: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_31, 0.001);  convert_element_type_31 = None
        sqrt_6: "f32[144][1]cuda:0" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
        reciprocal_6: "f32[144][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_20: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_50: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_20, -1);  mul_20 = None
        unsqueeze_51: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_21: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
        unsqueeze_52: "bf16[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_53: "bf16[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_22: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_53);  mul_21 = unsqueeze_53 = None
        unsqueeze_54: "bf16[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_55: "bf16[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_19: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_55);  mul_22 = unsqueeze_55 = None
        convert_element_type_32: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_33: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_32, torch.float32);  convert_element_type_32 = None
        neg_6: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.neg.default(convert_element_type_33)
        exp_6: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_20: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_33, add_20);  convert_element_type_33 = add_20 = None
        convert_element_type_34: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_11: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_34, arg44_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144);  convert_element_type_34 = arg44_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_35: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg45_1, torch.float32);  arg45_1 = None
        unsqueeze_56: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_35, -1);  convert_element_type_35 = None
        unsqueeze_57: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        sub_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_57);  convolution_11 = unsqueeze_57 = None
        convert_element_type_36: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg46_1, torch.float32);  arg46_1 = None
        add_21: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_36, 0.001);  convert_element_type_36 = None
        sqrt_7: "f32[144][1]cuda:0" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
        reciprocal_7: "f32[144][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_23: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_58: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_23, -1);  mul_23 = None
        unsqueeze_59: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_24: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
        unsqueeze_60: "bf16[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_61: "bf16[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_25: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_61);  mul_24 = unsqueeze_61 = None
        unsqueeze_62: "bf16[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_63: "bf16[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_22: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_25, unsqueeze_63);  mul_25 = unsqueeze_63 = None
        convert_element_type_37: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_38: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_37, torch.float32);  convert_element_type_37 = None
        neg_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.neg.default(convert_element_type_38)
        exp_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_23: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_38, add_23);  convert_element_type_38 = add_23 = None
        convert_element_type_39: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_2: "bf16[128, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_39, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_12: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.convolution.default(mean_2, arg49_1, arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg49_1 = arg50_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_40: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        neg_8: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.neg.default(convert_element_type_40)
        exp_8: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_24: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_40, add_24);  convert_element_type_40 = add_24 = None
        convert_element_type_41: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_13: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_41, arg51_1, arg52_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_41 = arg51_1 = arg52_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_2: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.sigmoid.default(convolution_13);  convolution_13 = None
        mul_26: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, sigmoid_2);  convert_element_type_39 = sigmoid_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_14: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.convolution.default(mul_26, arg53_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_26 = arg53_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_42: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg54_1, torch.float32);  arg54_1 = None
        unsqueeze_64: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
        unsqueeze_65: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        sub_8: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_65);  convolution_14 = unsqueeze_65 = None
        convert_element_type_43: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg55_1, torch.float32);  arg55_1 = None
        add_25: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_43, 0.001);  convert_element_type_43 = None
        sqrt_8: "f32[24][1]cuda:0" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
        reciprocal_8: "f32[24][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_27: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_66: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
        unsqueeze_67: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_28: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
        unsqueeze_68: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_69: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_29: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_69);  mul_28 = unsqueeze_69 = None
        unsqueeze_70: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_71: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_26: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_71);  mul_29 = unsqueeze_71 = None
        convert_element_type_44: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_27: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_44, convert_element_type_29);  convert_element_type_44 = convert_element_type_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_15: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.convolution.default(add_27, arg58_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_27 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_45: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg59_1, torch.float32);  arg59_1 = None
        unsqueeze_72: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_45, -1);  convert_element_type_45 = None
        unsqueeze_73: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        sub_9: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_73);  convolution_15 = unsqueeze_73 = None
        convert_element_type_46: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg60_1, torch.float32);  arg60_1 = None
        add_28: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_46, 0.001);  convert_element_type_46 = None
        sqrt_9: "f32[144][1]cuda:0" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        reciprocal_9: "f32[144][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_30: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_74: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
        unsqueeze_75: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_31: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
        unsqueeze_76: "bf16[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_77: "bf16[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_32: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_77);  mul_31 = unsqueeze_77 = None
        unsqueeze_78: "bf16[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_79: "bf16[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_29: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_79);  mul_32 = unsqueeze_79 = None
        convert_element_type_47: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_48: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_47, torch.float32);  convert_element_type_47 = None
        neg_9: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.neg.default(convert_element_type_48)
        exp_9: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_30: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_48, add_30);  convert_element_type_48 = add_30 = None
        convert_element_type_49: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "bf16[128, 144, 59, 59][501264, 1, 8496, 144]cuda:0" = torch.ops.aten.constant_pad_nd.default(convert_element_type_49, [1, 2, 1, 2], 0.0);  convert_element_type_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_16: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.convolution.default(constant_pad_nd_2, arg63_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 144);  constant_pad_nd_2 = arg63_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_50: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg64_1, torch.float32);  arg64_1 = None
        unsqueeze_80: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
        unsqueeze_81: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        sub_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_81);  convolution_16 = unsqueeze_81 = None
        convert_element_type_51: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg65_1, torch.float32);  arg65_1 = None
        add_31: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_51, 0.001);  convert_element_type_51 = None
        sqrt_10: "f32[144][1]cuda:0" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
        reciprocal_10: "f32[144][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_33: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_82: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
        unsqueeze_83: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_34: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
        unsqueeze_84: "bf16[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_85: "bf16[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_35: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_85);  mul_34 = unsqueeze_85 = None
        unsqueeze_86: "bf16[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_87: "bf16[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_32: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_87);  mul_35 = unsqueeze_87 = None
        convert_element_type_52: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_53: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_52, torch.float32);  convert_element_type_52 = None
        neg_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.neg.default(convert_element_type_53)
        exp_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_33: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_53, add_33);  convert_element_type_53 = add_33 = None
        convert_element_type_54: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_3: "bf16[128, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_54, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_17: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.convolution.default(mean_3, arg68_1, arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg68_1 = arg69_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_55: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        neg_11: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.neg.default(convert_element_type_55)
        exp_11: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_34: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_55, add_34);  convert_element_type_55 = add_34 = None
        convert_element_type_56: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_18: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_56, arg70_1, arg71_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_56 = arg70_1 = arg71_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_3: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.sigmoid.default(convolution_18);  convolution_18 = None
        mul_36: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_54, sigmoid_3);  convert_element_type_54 = sigmoid_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_19: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.convolution.default(mul_36, arg72_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_36 = arg72_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_57: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg73_1, torch.float32);  arg73_1 = None
        unsqueeze_88: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_57, -1);  convert_element_type_57 = None
        unsqueeze_89: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        sub_11: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_89);  convolution_19 = unsqueeze_89 = None
        convert_element_type_58: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg74_1, torch.float32);  arg74_1 = None
        add_35: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_58, 0.001);  convert_element_type_58 = None
        sqrt_11: "f32[40][1]cuda:0" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
        reciprocal_11: "f32[40][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_37: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_90: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_37, -1);  mul_37 = None
        unsqueeze_91: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_38: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
        unsqueeze_92: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_93: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_39: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, unsqueeze_93);  mul_38 = unsqueeze_93 = None
        unsqueeze_94: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_95: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_36: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(mul_39, unsqueeze_95);  mul_39 = unsqueeze_95 = None
        convert_element_type_59: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_20: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_59, arg77_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg77_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_60: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg78_1, torch.float32);  arg78_1 = None
        unsqueeze_96: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
        unsqueeze_97: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        sub_12: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_97);  convolution_20 = unsqueeze_97 = None
        convert_element_type_61: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg79_1, torch.float32);  arg79_1 = None
        add_37: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_61, 0.001);  convert_element_type_61 = None
        sqrt_12: "f32[240][1]cuda:0" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
        reciprocal_12: "f32[240][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_40: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_98: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_40, -1);  mul_40 = None
        unsqueeze_99: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_41: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
        unsqueeze_100: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_101: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_42: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, unsqueeze_101);  mul_41 = unsqueeze_101 = None
        unsqueeze_102: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_103: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_38: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_42, unsqueeze_103);  mul_42 = unsqueeze_103 = None
        convert_element_type_62: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_63: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_62, torch.float32);  convert_element_type_62 = None
        neg_12: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.neg.default(convert_element_type_63)
        exp_12: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_39: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_63, add_39);  convert_element_type_63 = add_39 = None
        convert_element_type_64: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_21: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_64, arg82_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 240);  convert_element_type_64 = arg82_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_65: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg83_1, torch.float32);  arg83_1 = None
        unsqueeze_104: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_65, -1);  convert_element_type_65 = None
        unsqueeze_105: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        sub_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_105);  convolution_21 = unsqueeze_105 = None
        convert_element_type_66: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg84_1, torch.float32);  arg84_1 = None
        add_40: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_66, 0.001);  convert_element_type_66 = None
        sqrt_13: "f32[240][1]cuda:0" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
        reciprocal_13: "f32[240][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_43: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_106: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_43, -1);  mul_43 = None
        unsqueeze_107: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_44: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
        unsqueeze_108: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_109: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_45: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_109);  mul_44 = unsqueeze_109 = None
        unsqueeze_110: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_111: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_41: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_45, unsqueeze_111);  mul_45 = unsqueeze_111 = None
        convert_element_type_67: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_68: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_67, torch.float32);  convert_element_type_67 = None
        neg_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.neg.default(convert_element_type_68)
        exp_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_42: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_68, add_42);  convert_element_type_68 = add_42 = None
        convert_element_type_69: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_4: "bf16[128, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_69, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_22: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.convolution.default(mean_4, arg87_1, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_4 = arg87_1 = arg88_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_70: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        neg_14: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.neg.default(convert_element_type_70)
        exp_14: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_43: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_70, add_43);  convert_element_type_70 = add_43 = None
        convert_element_type_71: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_23: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_71, arg89_1, arg90_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_71 = arg89_1 = arg90_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_4: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
        mul_46: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_69, sigmoid_4);  convert_element_type_69 = sigmoid_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_24: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.convolution.default(mul_46, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_46 = arg91_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_72: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg92_1, torch.float32);  arg92_1 = None
        unsqueeze_112: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
        unsqueeze_113: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        sub_14: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_113);  convolution_24 = unsqueeze_113 = None
        convert_element_type_73: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg93_1, torch.float32);  arg93_1 = None
        add_44: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_73, 0.001);  convert_element_type_73 = None
        sqrt_14: "f32[40][1]cuda:0" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
        reciprocal_14: "f32[40][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_47: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_114: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_47, -1);  mul_47 = None
        unsqueeze_115: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_48: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
        unsqueeze_116: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_117: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_49: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_117);  mul_48 = unsqueeze_117 = None
        unsqueeze_118: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_119: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_45: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_119);  mul_49 = unsqueeze_119 = None
        convert_element_type_74: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_46: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_74, convert_element_type_59);  convert_element_type_74 = convert_element_type_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_25: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.convolution.default(add_46, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_46 = arg96_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_75: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg97_1, torch.float32);  arg97_1 = None
        unsqueeze_120: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_75, -1);  convert_element_type_75 = None
        unsqueeze_121: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        sub_15: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_121);  convolution_25 = unsqueeze_121 = None
        convert_element_type_76: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg98_1, torch.float32);  arg98_1 = None
        add_47: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_76, 0.001);  convert_element_type_76 = None
        sqrt_15: "f32[240][1]cuda:0" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
        reciprocal_15: "f32[240][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_50: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_122: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_50, -1);  mul_50 = None
        unsqueeze_123: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_51: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
        unsqueeze_124: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_125: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_52: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_125);  mul_51 = unsqueeze_125 = None
        unsqueeze_126: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_127: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_48: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_52, unsqueeze_127);  mul_52 = unsqueeze_127 = None
        convert_element_type_77: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_78: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_77, torch.float32);  convert_element_type_77 = None
        neg_15: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.neg.default(convert_element_type_78)
        exp_15: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_49: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_78, add_49);  convert_element_type_78 = add_49 = None
        convert_element_type_79: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_3: "bf16[128, 240, 29, 29][201840, 1, 6960, 240]cuda:0" = torch.ops.aten.constant_pad_nd.default(convert_element_type_79, [0, 1, 0, 1], 0.0);  convert_element_type_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_26: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.convolution.default(constant_pad_nd_3, arg101_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240);  constant_pad_nd_3 = arg101_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_80: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg102_1, torch.float32);  arg102_1 = None
        unsqueeze_128: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
        unsqueeze_129: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        sub_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_129);  convolution_26 = unsqueeze_129 = None
        convert_element_type_81: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg103_1, torch.float32);  arg103_1 = None
        add_50: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_81, 0.001);  convert_element_type_81 = None
        sqrt_16: "f32[240][1]cuda:0" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
        reciprocal_16: "f32[240][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_53: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_130: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_53, -1);  mul_53 = None
        unsqueeze_131: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_54: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
        unsqueeze_132: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_133: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_55: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, unsqueeze_133);  mul_54 = unsqueeze_133 = None
        unsqueeze_134: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_135: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_51: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_135);  mul_55 = unsqueeze_135 = None
        convert_element_type_82: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_83: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_82, torch.float32);  convert_element_type_82 = None
        neg_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.neg.default(convert_element_type_83)
        exp_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_52: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_83, add_52);  convert_element_type_83 = add_52 = None
        convert_element_type_84: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_5: "bf16[128, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_84, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_27: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.convolution.default(mean_5, arg106_1, arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg106_1 = arg107_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_85: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        neg_17: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.neg.default(convert_element_type_85)
        exp_17: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_53: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        div_17: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_85, add_53);  convert_element_type_85 = add_53 = None
        convert_element_type_86: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_28: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_86, arg108_1, arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_86 = arg108_1 = arg109_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_5: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.sigmoid.default(convolution_28);  convolution_28 = None
        mul_56: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_84, sigmoid_5);  convert_element_type_84 = sigmoid_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_29: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.convolution.default(mul_56, arg110_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_56 = arg110_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_87: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg111_1, torch.float32);  arg111_1 = None
        unsqueeze_136: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_87, -1);  convert_element_type_87 = None
        unsqueeze_137: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        sub_17: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_137);  convolution_29 = unsqueeze_137 = None
        convert_element_type_88: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg112_1, torch.float32);  arg112_1 = None
        add_54: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_88, 0.001);  convert_element_type_88 = None
        sqrt_17: "f32[80][1]cuda:0" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
        reciprocal_17: "f32[80][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_57: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        unsqueeze_138: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
        unsqueeze_139: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        mul_58: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
        unsqueeze_140: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_141: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_59: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_141);  mul_58 = unsqueeze_141 = None
        unsqueeze_142: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_143: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_55: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_143);  mul_59 = unsqueeze_143 = None
        convert_element_type_89: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_30: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_89, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg115_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_90: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg116_1, torch.float32);  arg116_1 = None
        unsqueeze_144: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
        unsqueeze_145: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        sub_18: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_145);  convolution_30 = unsqueeze_145 = None
        convert_element_type_91: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg117_1, torch.float32);  arg117_1 = None
        add_56: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_91, 0.001);  convert_element_type_91 = None
        sqrt_18: "f32[480][1]cuda:0" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
        reciprocal_18: "f32[480][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_60: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        unsqueeze_146: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
        unsqueeze_147: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        mul_61: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
        unsqueeze_148: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_149: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_62: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_149);  mul_61 = unsqueeze_149 = None
        unsqueeze_150: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_151: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_57: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_151);  mul_62 = unsqueeze_151 = None
        convert_element_type_92: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_93: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_92, torch.float32);  convert_element_type_92 = None
        neg_18: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_93)
        exp_18: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_58: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        div_18: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_93, add_58);  convert_element_type_93 = add_58 = None
        convert_element_type_94: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_31: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_94, arg120_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  convert_element_type_94 = arg120_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_95: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg121_1, torch.float32);  arg121_1 = None
        unsqueeze_152: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_95, -1);  convert_element_type_95 = None
        unsqueeze_153: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        sub_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_153);  convolution_31 = unsqueeze_153 = None
        convert_element_type_96: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg122_1, torch.float32);  arg122_1 = None
        add_59: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_96, 0.001);  convert_element_type_96 = None
        sqrt_19: "f32[480][1]cuda:0" = torch.ops.aten.sqrt.default(add_59);  add_59 = None
        reciprocal_19: "f32[480][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_63: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        unsqueeze_154: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
        unsqueeze_155: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        mul_64: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
        unsqueeze_156: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_157: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_65: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_157);  mul_64 = unsqueeze_157 = None
        unsqueeze_158: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_159: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_60: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_159);  mul_65 = unsqueeze_159 = None
        convert_element_type_97: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_60, torch.bfloat16);  add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_98: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_97, torch.float32);  convert_element_type_97 = None
        neg_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_98)
        exp_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_61: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_98, add_61);  convert_element_type_98 = add_61 = None
        convert_element_type_99: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_6: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_99, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_32: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.convolution.default(mean_6, arg125_1, arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_6 = arg125_1 = arg126_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_100: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        neg_20: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.neg.default(convert_element_type_100)
        exp_20: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_62: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        div_20: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_100, add_62);  convert_element_type_100 = add_62 = None
        convert_element_type_101: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_33: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_101, arg127_1, arg128_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_101 = arg127_1 = arg128_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_6: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sigmoid.default(convolution_33);  convolution_33 = None
        mul_66: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_99, sigmoid_6);  convert_element_type_99 = sigmoid_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_34: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.convolution.default(mul_66, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_66 = arg129_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_102: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg130_1, torch.float32);  arg130_1 = None
        unsqueeze_160: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
        unsqueeze_161: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        sub_20: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_161);  convolution_34 = unsqueeze_161 = None
        convert_element_type_103: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg131_1, torch.float32);  arg131_1 = None
        add_63: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_103, 0.001);  convert_element_type_103 = None
        sqrt_20: "f32[80][1]cuda:0" = torch.ops.aten.sqrt.default(add_63);  add_63 = None
        reciprocal_20: "f32[80][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        mul_67: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        unsqueeze_162: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_67, -1);  mul_67 = None
        unsqueeze_163: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        mul_68: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
        unsqueeze_164: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_165: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_69: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, unsqueeze_165);  mul_68 = unsqueeze_165 = None
        unsqueeze_166: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_167: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_64: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_167);  mul_69 = unsqueeze_167 = None
        convert_element_type_104: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_65: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_104, convert_element_type_89);  convert_element_type_104 = convert_element_type_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_35: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(add_65, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg134_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_105: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg135_1, torch.float32);  arg135_1 = None
        unsqueeze_168: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_105, -1);  convert_element_type_105 = None
        unsqueeze_169: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        sub_21: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_169);  convolution_35 = unsqueeze_169 = None
        convert_element_type_106: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg136_1, torch.float32);  arg136_1 = None
        add_66: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_106, 0.001);  convert_element_type_106 = None
        sqrt_21: "f32[480][1]cuda:0" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
        reciprocal_21: "f32[480][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        mul_70: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        unsqueeze_170: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_70, -1);  mul_70 = None
        unsqueeze_171: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        mul_71: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
        unsqueeze_172: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_173: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_72: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_173);  mul_71 = unsqueeze_173 = None
        unsqueeze_174: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_175: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_67: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_175);  mul_72 = unsqueeze_175 = None
        convert_element_type_107: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_108: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_107, torch.float32);  convert_element_type_107 = None
        neg_21: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_108)
        exp_21: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_68: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        div_21: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_108, add_68);  convert_element_type_108 = add_68 = None
        convert_element_type_109: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_36: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_109, arg139_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  convert_element_type_109 = arg139_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_110: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float32);  arg140_1 = None
        unsqueeze_176: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_110, -1);  convert_element_type_110 = None
        unsqueeze_177: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        sub_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_177);  convolution_36 = unsqueeze_177 = None
        convert_element_type_111: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg141_1, torch.float32);  arg141_1 = None
        add_69: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_111, 0.001);  convert_element_type_111 = None
        sqrt_22: "f32[480][1]cuda:0" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
        reciprocal_22: "f32[480][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        mul_73: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        unsqueeze_178: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_73, -1);  mul_73 = None
        unsqueeze_179: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        mul_74: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
        unsqueeze_180: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_181: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_75: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, unsqueeze_181);  mul_74 = unsqueeze_181 = None
        unsqueeze_182: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_183: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_70: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_75, unsqueeze_183);  mul_75 = unsqueeze_183 = None
        convert_element_type_112: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_113: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_112, torch.float32);  convert_element_type_112 = None
        neg_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_113)
        exp_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_71: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_113, add_71);  convert_element_type_113 = add_71 = None
        convert_element_type_114: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_7: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_114, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_37: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.convolution.default(mean_7, arg144_1, arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg144_1 = arg145_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_115: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        neg_23: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.neg.default(convert_element_type_115)
        exp_23: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_72: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        div_23: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_115, add_72);  convert_element_type_115 = add_72 = None
        convert_element_type_116: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_38: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_116, arg146_1, arg147_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_116 = arg146_1 = arg147_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_7: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sigmoid.default(convolution_38);  convolution_38 = None
        mul_76: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_114, sigmoid_7);  convert_element_type_114 = sigmoid_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_39: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.convolution.default(mul_76, arg148_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_76 = arg148_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_117: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg149_1, torch.float32);  arg149_1 = None
        unsqueeze_184: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_117, -1);  convert_element_type_117 = None
        unsqueeze_185: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        sub_23: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_185);  convolution_39 = unsqueeze_185 = None
        convert_element_type_118: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg150_1, torch.float32);  arg150_1 = None
        add_73: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_118, 0.001);  convert_element_type_118 = None
        sqrt_23: "f32[80][1]cuda:0" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
        reciprocal_23: "f32[80][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_77: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        unsqueeze_186: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_77, -1);  mul_77 = None
        unsqueeze_187: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        mul_78: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
        unsqueeze_188: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_189: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_79: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, unsqueeze_189);  mul_78 = unsqueeze_189 = None
        unsqueeze_190: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_191: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_74: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_191);  mul_79 = unsqueeze_191 = None
        convert_element_type_119: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_75: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_119, add_65);  convert_element_type_119 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_40: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(add_75, arg153_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_75 = arg153_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_120: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg154_1, torch.float32);  arg154_1 = None
        unsqueeze_192: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_120, -1);  convert_element_type_120 = None
        unsqueeze_193: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        sub_24: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_193);  convolution_40 = unsqueeze_193 = None
        convert_element_type_121: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg155_1, torch.float32);  arg155_1 = None
        add_76: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_121, 0.001);  convert_element_type_121 = None
        sqrt_24: "f32[480][1]cuda:0" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
        reciprocal_24: "f32[480][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_80: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        unsqueeze_194: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_80, -1);  mul_80 = None
        unsqueeze_195: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        mul_81: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
        unsqueeze_196: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_197: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_82: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_197);  mul_81 = unsqueeze_197 = None
        unsqueeze_198: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_199: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_77: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_82, unsqueeze_199);  mul_82 = unsqueeze_199 = None
        convert_element_type_122: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_123: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_122, torch.float32);  convert_element_type_122 = None
        neg_24: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_123)
        exp_24: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_78: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        div_24: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_123, add_78);  convert_element_type_123 = add_78 = None
        convert_element_type_124: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_41: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_124, arg158_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480);  convert_element_type_124 = arg158_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_125: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg159_1, torch.float32);  arg159_1 = None
        unsqueeze_200: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_125, -1);  convert_element_type_125 = None
        unsqueeze_201: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        sub_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_201);  convolution_41 = unsqueeze_201 = None
        convert_element_type_126: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg160_1, torch.float32);  arg160_1 = None
        add_79: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_126, 0.001);  convert_element_type_126 = None
        sqrt_25: "f32[480][1]cuda:0" = torch.ops.aten.sqrt.default(add_79);  add_79 = None
        reciprocal_25: "f32[480][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_83: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        unsqueeze_202: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_83, -1);  mul_83 = None
        unsqueeze_203: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        mul_84: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
        unsqueeze_204: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_205: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_85: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_205);  mul_84 = unsqueeze_205 = None
        unsqueeze_206: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_207: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_80: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_207);  mul_85 = unsqueeze_207 = None
        convert_element_type_127: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_128: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_127, torch.float32);  convert_element_type_127 = None
        neg_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_128)
        exp_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_81: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_128, add_81);  convert_element_type_128 = add_81 = None
        convert_element_type_129: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_8: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_129, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_42: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.convolution.default(mean_8, arg163_1, arg164_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg163_1 = arg164_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_130: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32);  convolution_42 = None
        neg_26: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.neg.default(convert_element_type_130)
        exp_26: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_82: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        div_26: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_130, add_82);  convert_element_type_130 = add_82 = None
        convert_element_type_131: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_43: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_131, arg165_1, arg166_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_131 = arg165_1 = arg166_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_8: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sigmoid.default(convolution_43);  convolution_43 = None
        mul_86: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_129, sigmoid_8);  convert_element_type_129 = sigmoid_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_44: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.convolution.default(mul_86, arg167_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_86 = arg167_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_132: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg168_1, torch.float32);  arg168_1 = None
        unsqueeze_208: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_132, -1);  convert_element_type_132 = None
        unsqueeze_209: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        sub_26: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_209);  convolution_44 = unsqueeze_209 = None
        convert_element_type_133: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg169_1, torch.float32);  arg169_1 = None
        add_83: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_133, 0.001);  convert_element_type_133 = None
        sqrt_26: "f32[112][1]cuda:0" = torch.ops.aten.sqrt.default(add_83);  add_83 = None
        reciprocal_26: "f32[112][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_87: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        unsqueeze_210: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
        unsqueeze_211: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        mul_88: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
        unsqueeze_212: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_213: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_89: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_213);  mul_88 = unsqueeze_213 = None
        unsqueeze_214: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_215: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_84: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_215);  mul_89 = unsqueeze_215 = None
        convert_element_type_134: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_45: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_134, arg172_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg172_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_135: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg173_1, torch.float32);  arg173_1 = None
        unsqueeze_216: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_135, -1);  convert_element_type_135 = None
        unsqueeze_217: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        sub_27: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_217);  convolution_45 = unsqueeze_217 = None
        convert_element_type_136: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg174_1, torch.float32);  arg174_1 = None
        add_85: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_136, 0.001);  convert_element_type_136 = None
        sqrt_27: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
        reciprocal_27: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_90: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_218: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
        unsqueeze_219: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        mul_91: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
        unsqueeze_220: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_221: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_92: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_221);  mul_91 = unsqueeze_221 = None
        unsqueeze_222: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_223: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_86: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_223);  mul_92 = unsqueeze_223 = None
        convert_element_type_137: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_138: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_137, torch.float32);  convert_element_type_137 = None
        neg_27: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_138)
        exp_27: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_87: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        div_27: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_138, add_87);  convert_element_type_138 = add_87 = None
        convert_element_type_139: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_46: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_139, arg177_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672);  convert_element_type_139 = arg177_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_140: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg178_1, torch.float32);  arg178_1 = None
        unsqueeze_224: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_140, -1);  convert_element_type_140 = None
        unsqueeze_225: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        sub_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_225);  convolution_46 = unsqueeze_225 = None
        convert_element_type_141: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg179_1, torch.float32);  arg179_1 = None
        add_88: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_141, 0.001);  convert_element_type_141 = None
        sqrt_28: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
        reciprocal_28: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_93: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_226: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
        unsqueeze_227: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        mul_94: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
        unsqueeze_228: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_229: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_95: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_229);  mul_94 = unsqueeze_229 = None
        unsqueeze_230: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_231: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_89: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_231);  mul_95 = unsqueeze_231 = None
        convert_element_type_142: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_143: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_142, torch.float32);  convert_element_type_142 = None
        neg_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_143)
        exp_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_90: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        div_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_143, add_90);  convert_element_type_143 = add_90 = None
        convert_element_type_144: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_9: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_144, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_47: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.convolution.default(mean_9, arg182_1, arg183_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_9 = arg182_1 = arg183_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_145: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32);  convolution_47 = None
        neg_29: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.neg.default(convert_element_type_145)
        exp_29: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_91: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        div_29: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_145, add_91);  convert_element_type_145 = add_91 = None
        convert_element_type_146: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(div_29, torch.bfloat16);  div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_48: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_146, arg184_1, arg185_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_146 = arg184_1 = arg185_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_9: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
        mul_96: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_144, sigmoid_9);  convert_element_type_144 = sigmoid_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_49: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.convolution.default(mul_96, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_96 = arg186_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_147: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg187_1, torch.float32);  arg187_1 = None
        unsqueeze_232: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_147, -1);  convert_element_type_147 = None
        unsqueeze_233: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        sub_29: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_233);  convolution_49 = unsqueeze_233 = None
        convert_element_type_148: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg188_1, torch.float32);  arg188_1 = None
        add_92: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_148, 0.001);  convert_element_type_148 = None
        sqrt_29: "f32[112][1]cuda:0" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
        reciprocal_29: "f32[112][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_97: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_234: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_97, -1);  mul_97 = None
        unsqueeze_235: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        mul_98: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
        unsqueeze_236: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_237: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_99: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_237);  mul_98 = unsqueeze_237 = None
        unsqueeze_238: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_239: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_93: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_239);  mul_99 = unsqueeze_239 = None
        convert_element_type_149: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_94: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_149, convert_element_type_134);  convert_element_type_149 = convert_element_type_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_50: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(add_94, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg191_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_150: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg192_1, torch.float32);  arg192_1 = None
        unsqueeze_240: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_150, -1);  convert_element_type_150 = None
        unsqueeze_241: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        sub_30: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_241);  convolution_50 = unsqueeze_241 = None
        convert_element_type_151: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg193_1, torch.float32);  arg193_1 = None
        add_95: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_151, 0.001);  convert_element_type_151 = None
        sqrt_30: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_95);  add_95 = None
        reciprocal_30: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_100: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_242: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_100, -1);  mul_100 = None
        unsqueeze_243: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        mul_101: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
        unsqueeze_244: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_245: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_102: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, unsqueeze_245);  mul_101 = unsqueeze_245 = None
        unsqueeze_246: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_247: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_96: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_102, unsqueeze_247);  mul_102 = unsqueeze_247 = None
        convert_element_type_152: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16);  add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_153: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_152, torch.float32);  convert_element_type_152 = None
        neg_30: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_153)
        exp_30: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_30);  neg_30 = None
        add_97: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        div_30: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_153, add_97);  convert_element_type_153 = add_97 = None
        convert_element_type_154: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_30, torch.bfloat16);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_51: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_154, arg196_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672);  convert_element_type_154 = arg196_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_155: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg197_1, torch.float32);  arg197_1 = None
        unsqueeze_248: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_155, -1);  convert_element_type_155 = None
        unsqueeze_249: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        sub_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_249);  convolution_51 = unsqueeze_249 = None
        convert_element_type_156: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg198_1, torch.float32);  arg198_1 = None
        add_98: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_156, 0.001);  convert_element_type_156 = None
        sqrt_31: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
        reciprocal_31: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_103: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_250: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_103, -1);  mul_103 = None
        unsqueeze_251: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        mul_104: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
        unsqueeze_252: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_253: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_105: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, unsqueeze_253);  mul_104 = unsqueeze_253 = None
        unsqueeze_254: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_255: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_99: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_105, unsqueeze_255);  mul_105 = unsqueeze_255 = None
        convert_element_type_157: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_158: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_157, torch.float32);  convert_element_type_157 = None
        neg_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_158)
        exp_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_100: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        div_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_158, add_100);  convert_element_type_158 = add_100 = None
        convert_element_type_159: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16);  div_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_10: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_159, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_52: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.convolution.default(mean_10, arg201_1, arg202_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_10 = arg201_1 = arg202_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_160: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32);  convolution_52 = None
        neg_32: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.neg.default(convert_element_type_160)
        exp_32: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_101: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.add.Tensor(exp_32, 1);  exp_32 = None
        div_32: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_160, add_101);  convert_element_type_160 = add_101 = None
        convert_element_type_161: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(div_32, torch.bfloat16);  div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_53: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_161, arg203_1, arg204_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_161 = arg203_1 = arg204_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_10: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
        mul_106: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_159, sigmoid_10);  convert_element_type_159 = sigmoid_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_54: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.convolution.default(mul_106, arg205_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_106 = arg205_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_162: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg206_1, torch.float32);  arg206_1 = None
        unsqueeze_256: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_162, -1);  convert_element_type_162 = None
        unsqueeze_257: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        sub_32: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_257);  convolution_54 = unsqueeze_257 = None
        convert_element_type_163: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg207_1, torch.float32);  arg207_1 = None
        add_102: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_163, 0.001);  convert_element_type_163 = None
        sqrt_32: "f32[112][1]cuda:0" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
        reciprocal_32: "f32[112][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_107: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_258: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_107, -1);  mul_107 = None
        unsqueeze_259: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        mul_108: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
        unsqueeze_260: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_261: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_109: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_261);  mul_108 = unsqueeze_261 = None
        unsqueeze_262: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_263: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_103: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(mul_109, unsqueeze_263);  mul_109 = unsqueeze_263 = None
        convert_element_type_164: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_104: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_164, add_94);  convert_element_type_164 = add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_55: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(add_104, arg210_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_104 = arg210_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_165: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg211_1, torch.float32);  arg211_1 = None
        unsqueeze_264: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_165, -1);  convert_element_type_165 = None
        unsqueeze_265: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        sub_33: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_265);  convolution_55 = unsqueeze_265 = None
        convert_element_type_166: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg212_1, torch.float32);  arg212_1 = None
        add_105: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_166, 0.001);  convert_element_type_166 = None
        sqrt_33: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
        reciprocal_33: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_110: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_266: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_110, -1);  mul_110 = None
        unsqueeze_267: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        mul_111: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
        unsqueeze_268: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_269: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_112: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, unsqueeze_269);  mul_111 = unsqueeze_269 = None
        unsqueeze_270: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_271: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_106: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_112, unsqueeze_271);  mul_112 = unsqueeze_271 = None
        convert_element_type_167: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16);  add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_168: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_167, torch.float32);  convert_element_type_167 = None
        neg_33: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_168)
        exp_33: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_33);  neg_33 = None
        add_107: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_33, 1);  exp_33 = None
        div_33: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_168, add_107);  convert_element_type_168 = add_107 = None
        convert_element_type_169: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16);  div_33 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_4: "bf16[128, 672, 17, 17][194208, 1, 11424, 672]cuda:0" = torch.ops.aten.constant_pad_nd.default(convert_element_type_169, [1, 2, 1, 2], 0.0);  convert_element_type_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_56: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.convolution.default(constant_pad_nd_4, arg215_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 672);  constant_pad_nd_4 = arg215_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_170: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg216_1, torch.float32);  arg216_1 = None
        unsqueeze_272: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_170, -1);  convert_element_type_170 = None
        unsqueeze_273: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        sub_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_273);  convolution_56 = unsqueeze_273 = None
        convert_element_type_171: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg217_1, torch.float32);  arg217_1 = None
        add_108: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_171, 0.001);  convert_element_type_171 = None
        sqrt_34: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
        reciprocal_34: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_113: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_274: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_113, -1);  mul_113 = None
        unsqueeze_275: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        mul_114: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
        unsqueeze_276: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_277: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_115: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, unsqueeze_277);  mul_114 = unsqueeze_277 = None
        unsqueeze_278: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_279: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_109: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_115, unsqueeze_279);  mul_115 = unsqueeze_279 = None
        convert_element_type_172: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_173: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_172, torch.float32);  convert_element_type_172 = None
        neg_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_173)
        exp_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.exp.default(neg_34);  neg_34 = None
        add_110: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_34, 1);  exp_34 = None
        div_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_173, add_110);  convert_element_type_173 = add_110 = None
        convert_element_type_174: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_34, torch.bfloat16);  div_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_11: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_174, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_57: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.convolution.default(mean_11, arg220_1, arg221_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_11 = arg220_1 = arg221_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_175: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32);  convolution_57 = None
        neg_35: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.neg.default(convert_element_type_175)
        exp_35: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.exp.default(neg_35);  neg_35 = None
        add_111: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.add.Tensor(exp_35, 1);  exp_35 = None
        div_35: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_175, add_111);  convert_element_type_175 = add_111 = None
        convert_element_type_176: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(div_35, torch.bfloat16);  div_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_58: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_176, arg222_1, arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_176 = arg222_1 = arg223_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_11: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
        mul_116: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_174, sigmoid_11);  convert_element_type_174 = sigmoid_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_59: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.convolution.default(mul_116, arg224_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_116 = arg224_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_177: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg225_1, torch.float32);  arg225_1 = None
        unsqueeze_280: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_177, -1);  convert_element_type_177 = None
        unsqueeze_281: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        sub_35: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_281);  convolution_59 = unsqueeze_281 = None
        convert_element_type_178: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg226_1, torch.float32);  arg226_1 = None
        add_112: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_178, 0.001);  convert_element_type_178 = None
        sqrt_35: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
        reciprocal_35: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_117: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_282: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
        unsqueeze_283: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        mul_118: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
        unsqueeze_284: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_285: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_119: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_285);  mul_118 = unsqueeze_285 = None
        unsqueeze_286: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_287: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_113: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_287);  mul_119 = unsqueeze_287 = None
        convert_element_type_179: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_113, torch.bfloat16);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_60: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_179, arg229_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg229_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_180: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg230_1, torch.float32);  arg230_1 = None
        unsqueeze_288: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_180, -1);  convert_element_type_180 = None
        unsqueeze_289: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        sub_36: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_289);  convolution_60 = unsqueeze_289 = None
        convert_element_type_181: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg231_1, torch.float32);  arg231_1 = None
        add_114: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_181, 0.001);  convert_element_type_181 = None
        sqrt_36: "f32[1152][1]cuda:0" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_36: "f32[1152][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_120: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_290: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
        unsqueeze_291: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        mul_121: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
        unsqueeze_292: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_293: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_122: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_293);  mul_121 = unsqueeze_293 = None
        unsqueeze_294: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_295: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_115: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_295);  mul_122 = unsqueeze_295 = None
        convert_element_type_182: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16);  add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_183: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_182, torch.float32);  convert_element_type_182 = None
        neg_36: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_183)
        exp_36: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_36);  neg_36 = None
        add_116: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_36, 1);  exp_36 = None
        div_36: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_183, add_116);  convert_element_type_183 = add_116 = None
        convert_element_type_184: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_36, torch.bfloat16);  div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_61: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_184, arg234_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  convert_element_type_184 = arg234_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_185: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg235_1, torch.float32);  arg235_1 = None
        unsqueeze_296: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_185, -1);  convert_element_type_185 = None
        unsqueeze_297: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        sub_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_297);  convolution_61 = unsqueeze_297 = None
        convert_element_type_186: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg236_1, torch.float32);  arg236_1 = None
        add_117: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_186, 0.001);  convert_element_type_186 = None
        sqrt_37: "f32[1152][1]cuda:0" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
        reciprocal_37: "f32[1152][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_123: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_298: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_299: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        mul_124: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
        unsqueeze_300: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_301: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_125: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_301);  mul_124 = unsqueeze_301 = None
        unsqueeze_302: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_303: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_118: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_303);  mul_125 = unsqueeze_303 = None
        convert_element_type_187: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_118, torch.bfloat16);  add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_188: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_187, torch.float32);  convert_element_type_187 = None
        neg_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_188)
        exp_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_37);  neg_37 = None
        add_119: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_37, 1);  exp_37 = None
        div_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_188, add_119);  convert_element_type_188 = add_119 = None
        convert_element_type_189: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_37, torch.bfloat16);  div_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_12: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_189, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_62: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.convolution.default(mean_12, arg239_1, arg240_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_12 = arg239_1 = arg240_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_190: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_62, torch.float32);  convolution_62 = None
        neg_38: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.neg.default(convert_element_type_190)
        exp_38: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.exp.default(neg_38);  neg_38 = None
        add_120: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(exp_38, 1);  exp_38 = None
        div_38: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_190, add_120);  convert_element_type_190 = add_120 = None
        convert_element_type_191: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(div_38, torch.bfloat16);  div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_63: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_191, arg241_1, arg242_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_191 = arg241_1 = arg242_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_12: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_63);  convolution_63 = None
        mul_126: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_189, sigmoid_12);  convert_element_type_189 = sigmoid_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_64: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.convolution.default(mul_126, arg243_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_126 = arg243_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_192: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg244_1, torch.float32);  arg244_1 = None
        unsqueeze_304: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_192, -1);  convert_element_type_192 = None
        unsqueeze_305: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        sub_38: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_305);  convolution_64 = unsqueeze_305 = None
        convert_element_type_193: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg245_1, torch.float32);  arg245_1 = None
        add_121: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_193, 0.001);  convert_element_type_193 = None
        sqrt_38: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_121);  add_121 = None
        reciprocal_38: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_127: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_306: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_127, -1);  mul_127 = None
        unsqueeze_307: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        mul_128: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
        unsqueeze_308: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
        unsqueeze_309: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_129: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, unsqueeze_309);  mul_128 = unsqueeze_309 = None
        unsqueeze_310: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_311: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_122: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_311);  mul_129 = unsqueeze_311 = None
        convert_element_type_194: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_122, torch.bfloat16);  add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_123: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_194, convert_element_type_179);  convert_element_type_194 = convert_element_type_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_65: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(add_123, arg248_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg248_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_195: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg249_1, torch.float32);  arg249_1 = None
        unsqueeze_312: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_195, -1);  convert_element_type_195 = None
        unsqueeze_313: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        sub_39: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_313);  convolution_65 = unsqueeze_313 = None
        convert_element_type_196: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg250_1, torch.float32);  arg250_1 = None
        add_124: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_196, 0.001);  convert_element_type_196 = None
        sqrt_39: "f32[1152][1]cuda:0" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
        reciprocal_39: "f32[1152][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_130: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_314: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_130, -1);  mul_130 = None
        unsqueeze_315: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        mul_131: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
        unsqueeze_316: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_317: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_132: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_317);  mul_131 = unsqueeze_317 = None
        unsqueeze_318: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_319: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_125: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_319);  mul_132 = unsqueeze_319 = None
        convert_element_type_197: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_198: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_197, torch.float32);  convert_element_type_197 = None
        neg_39: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_198)
        exp_39: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_39);  neg_39 = None
        add_126: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_39, 1);  exp_39 = None
        div_39: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_198, add_126);  convert_element_type_198 = add_126 = None
        convert_element_type_199: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_39, torch.bfloat16);  div_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_66: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_199, arg253_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  convert_element_type_199 = arg253_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_200: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg254_1, torch.float32);  arg254_1 = None
        unsqueeze_320: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_200, -1);  convert_element_type_200 = None
        unsqueeze_321: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        sub_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_321);  convolution_66 = unsqueeze_321 = None
        convert_element_type_201: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg255_1, torch.float32);  arg255_1 = None
        add_127: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_201, 0.001);  convert_element_type_201 = None
        sqrt_40: "f32[1152][1]cuda:0" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
        reciprocal_40: "f32[1152][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_133: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_322: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_133, -1);  mul_133 = None
        unsqueeze_323: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        mul_134: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
        unsqueeze_324: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_325: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_135: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, unsqueeze_325);  mul_134 = unsqueeze_325 = None
        unsqueeze_326: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_327: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_128: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_135, unsqueeze_327);  mul_135 = unsqueeze_327 = None
        convert_element_type_202: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_128, torch.bfloat16);  add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_203: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_202, torch.float32);  convert_element_type_202 = None
        neg_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_203)
        exp_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_40);  neg_40 = None
        add_129: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_40, 1);  exp_40 = None
        div_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_203, add_129);  convert_element_type_203 = add_129 = None
        convert_element_type_204: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_40, torch.bfloat16);  div_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_13: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_204, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_67: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.convolution.default(mean_13, arg258_1, arg259_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_13 = arg258_1 = arg259_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_205: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_67, torch.float32);  convolution_67 = None
        neg_41: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.neg.default(convert_element_type_205)
        exp_41: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.exp.default(neg_41);  neg_41 = None
        add_130: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(exp_41, 1);  exp_41 = None
        div_41: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_205, add_130);  convert_element_type_205 = add_130 = None
        convert_element_type_206: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(div_41, torch.bfloat16);  div_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_68: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_206, arg260_1, arg261_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_206 = arg260_1 = arg261_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_13: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_68);  convolution_68 = None
        mul_136: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_204, sigmoid_13);  convert_element_type_204 = sigmoid_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_69: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.convolution.default(mul_136, arg262_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_136 = arg262_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_207: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg263_1, torch.float32);  arg263_1 = None
        unsqueeze_328: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_207, -1);  convert_element_type_207 = None
        unsqueeze_329: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        sub_41: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_329);  convolution_69 = unsqueeze_329 = None
        convert_element_type_208: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg264_1, torch.float32);  arg264_1 = None
        add_131: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_208, 0.001);  convert_element_type_208 = None
        sqrt_41: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_131);  add_131 = None
        reciprocal_41: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_137: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_330: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_137, -1);  mul_137 = None
        unsqueeze_331: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        mul_138: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
        unsqueeze_332: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_333: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_139: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_333);  mul_138 = unsqueeze_333 = None
        unsqueeze_334: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_335: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_132: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_335);  mul_139 = unsqueeze_335 = None
        convert_element_type_209: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_132, torch.bfloat16);  add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_133: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_209, add_123);  convert_element_type_209 = add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_70: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(add_133, arg267_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg267_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_210: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg268_1, torch.float32);  arg268_1 = None
        unsqueeze_336: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_210, -1);  convert_element_type_210 = None
        unsqueeze_337: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        sub_42: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_337);  convolution_70 = unsqueeze_337 = None
        convert_element_type_211: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg269_1, torch.float32);  arg269_1 = None
        add_134: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_211, 0.001);  convert_element_type_211 = None
        sqrt_42: "f32[1152][1]cuda:0" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_42: "f32[1152][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_140: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_338: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_140, -1);  mul_140 = None
        unsqueeze_339: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        mul_141: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
        unsqueeze_340: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_341: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_142: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, unsqueeze_341);  mul_141 = unsqueeze_341 = None
        unsqueeze_342: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_343: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_135: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_343);  mul_142 = unsqueeze_343 = None
        convert_element_type_212: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_213: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_212, torch.float32);  convert_element_type_212 = None
        neg_42: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_213)
        exp_42: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_42);  neg_42 = None
        add_136: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_42, 1);  exp_42 = None
        div_42: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_213, add_136);  convert_element_type_213 = add_136 = None
        convert_element_type_214: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_42, torch.bfloat16);  div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_71: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_214, arg272_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  convert_element_type_214 = arg272_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_215: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg273_1, torch.float32);  arg273_1 = None
        unsqueeze_344: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_215, -1);  convert_element_type_215 = None
        unsqueeze_345: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        sub_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_345);  convolution_71 = unsqueeze_345 = None
        convert_element_type_216: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg274_1, torch.float32);  arg274_1 = None
        add_137: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_216, 0.001);  convert_element_type_216 = None
        sqrt_43: "f32[1152][1]cuda:0" = torch.ops.aten.sqrt.default(add_137);  add_137 = None
        reciprocal_43: "f32[1152][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_143: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_346: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_143, -1);  mul_143 = None
        unsqueeze_347: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        mul_144: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
        unsqueeze_348: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_349: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_145: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_349);  mul_144 = unsqueeze_349 = None
        unsqueeze_350: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_351: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_138: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_351);  mul_145 = unsqueeze_351 = None
        convert_element_type_217: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16);  add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_218: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_217, torch.float32);  convert_element_type_217 = None
        neg_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_218)
        exp_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_43);  neg_43 = None
        add_139: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_43, 1);  exp_43 = None
        div_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_218, add_139);  convert_element_type_218 = add_139 = None
        convert_element_type_219: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_43, torch.bfloat16);  div_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_14: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_219, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_72: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.convolution.default(mean_14, arg277_1, arg278_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg277_1 = arg278_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_220: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_72, torch.float32);  convolution_72 = None
        neg_44: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.neg.default(convert_element_type_220)
        exp_44: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.exp.default(neg_44);  neg_44 = None
        add_140: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(exp_44, 1);  exp_44 = None
        div_44: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_220, add_140);  convert_element_type_220 = add_140 = None
        convert_element_type_221: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(div_44, torch.bfloat16);  div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_73: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_221, arg279_1, arg280_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_221 = arg279_1 = arg280_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_14: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
        mul_146: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_219, sigmoid_14);  convert_element_type_219 = sigmoid_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_74: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.convolution.default(mul_146, arg281_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_146 = arg281_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_222: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg282_1, torch.float32);  arg282_1 = None
        unsqueeze_352: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_222, -1);  convert_element_type_222 = None
        unsqueeze_353: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        sub_44: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_353);  convolution_74 = unsqueeze_353 = None
        convert_element_type_223: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg283_1, torch.float32);  arg283_1 = None
        add_141: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_223, 0.001);  convert_element_type_223 = None
        sqrt_44: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_44: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_147: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_354: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
        unsqueeze_355: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        mul_148: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
        unsqueeze_356: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_357: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_149: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_357);  mul_148 = unsqueeze_357 = None
        unsqueeze_358: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_359: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_142: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_359);  mul_149 = unsqueeze_359 = None
        convert_element_type_224: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16);  add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_143: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_224, add_133);  convert_element_type_224 = add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_75: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(add_143, arg286_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_143 = arg286_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_225: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg287_1, torch.float32);  arg287_1 = None
        unsqueeze_360: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_225, -1);  convert_element_type_225 = None
        unsqueeze_361: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        sub_45: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_361);  convolution_75 = unsqueeze_361 = None
        convert_element_type_226: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg288_1, torch.float32);  arg288_1 = None
        add_144: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_226, 0.001);  convert_element_type_226 = None
        sqrt_45: "f32[1152][1]cuda:0" = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_45: "f32[1152][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_150: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_362: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
        unsqueeze_363: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        mul_151: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
        unsqueeze_364: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_365: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_152: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_365);  mul_151 = unsqueeze_365 = None
        unsqueeze_366: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_367: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_145: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_367);  mul_152 = unsqueeze_367 = None
        convert_element_type_227: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.bfloat16);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_228: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_227, torch.float32);  convert_element_type_227 = None
        neg_45: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_228)
        exp_45: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_45);  neg_45 = None
        add_146: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_45, 1);  exp_45 = None
        div_45: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_228, add_146);  convert_element_type_228 = add_146 = None
        convert_element_type_229: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_45, torch.bfloat16);  div_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_76: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_229, arg291_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152);  convert_element_type_229 = arg291_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_230: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg292_1, torch.float32);  arg292_1 = None
        unsqueeze_368: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_230, -1);  convert_element_type_230 = None
        unsqueeze_369: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        sub_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_369);  convolution_76 = unsqueeze_369 = None
        convert_element_type_231: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg293_1, torch.float32);  arg293_1 = None
        add_147: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_231, 0.001);  convert_element_type_231 = None
        sqrt_46: "f32[1152][1]cuda:0" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_46: "f32[1152][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_153: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_370: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
        unsqueeze_371: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        mul_154: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
        unsqueeze_372: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_373: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_155: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_373);  mul_154 = unsqueeze_373 = None
        unsqueeze_374: "bf16[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_375: "bf16[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_148: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_375);  mul_155 = unsqueeze_375 = None
        convert_element_type_232: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_148, torch.bfloat16);  add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_233: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_232, torch.float32);  convert_element_type_232 = None
        neg_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_233)
        exp_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_46);  neg_46 = None
        add_149: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_46, 1);  exp_46 = None
        div_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_233, add_149);  convert_element_type_233 = add_149 = None
        convert_element_type_234: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_46, torch.bfloat16);  div_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_15: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_234, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_77: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.convolution.default(mean_15, arg296_1, arg297_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg296_1 = arg297_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_235: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_77, torch.float32);  convolution_77 = None
        neg_47: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.neg.default(convert_element_type_235)
        exp_47: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.exp.default(neg_47);  neg_47 = None
        add_150: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(exp_47, 1);  exp_47 = None
        div_47: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_235, add_150);  convert_element_type_235 = add_150 = None
        convert_element_type_236: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(div_47, torch.bfloat16);  div_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convolution_78: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_236, arg298_1, arg299_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_236 = arg298_1 = arg299_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_15: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_78);  convolution_78 = None
        mul_156: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_234, sigmoid_15);  convert_element_type_234 = sigmoid_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_79: "bf16[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.convolution.default(mul_156, arg300_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_156 = arg300_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_237: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg301_1, torch.float32);  arg301_1 = None
        unsqueeze_376: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_237, -1);  convert_element_type_237 = None
        unsqueeze_377: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        sub_47: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_377);  convolution_79 = unsqueeze_377 = None
        convert_element_type_238: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg302_1, torch.float32);  arg302_1 = None
        add_151: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_238, 0.001);  convert_element_type_238 = None
        sqrt_47: "f32[320][1]cuda:0" = torch.ops.aten.sqrt.default(add_151);  add_151 = None
        reciprocal_47: "f32[320][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_157: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_378: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_157, -1);  mul_157 = None
        unsqueeze_379: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        mul_158: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
        unsqueeze_380: "bf16[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_381: "bf16[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_159: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, unsqueeze_381);  mul_158 = unsqueeze_381 = None
        unsqueeze_382: "bf16[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_383: "bf16[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_152: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_159, unsqueeze_383);  mul_159 = unsqueeze_383 = None
        convert_element_type_239: "bf16[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/efficientnet.py:327 in forward_features, code: x = self.conv_head(x)
        convolution_80: "bf16[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_239, arg305_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_239 = arg305_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_240: "f32[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg306_1, torch.float32);  arg306_1 = None
        unsqueeze_384: "f32[1280, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_240, -1);  convert_element_type_240 = None
        unsqueeze_385: "f32[1280, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        sub_48: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_385);  convolution_80 = unsqueeze_385 = None
        convert_element_type_241: "f32[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg307_1, torch.float32);  arg307_1 = None
        add_153: "f32[1280][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_241, 0.001);  convert_element_type_241 = None
        sqrt_48: "f32[1280][1]cuda:0" = torch.ops.aten.sqrt.default(add_153);  add_153 = None
        reciprocal_48: "f32[1280][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_160: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_386: "f32[1280, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_160, -1);  mul_160 = None
        unsqueeze_387: "f32[1280, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        mul_161: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
        unsqueeze_388: "bf16[1280, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_389: "bf16[1280, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_162: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_389);  mul_161 = unsqueeze_389 = None
        unsqueeze_390: "bf16[1280, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_391: "bf16[1280, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_154: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_391);  mul_162 = unsqueeze_391 = None
        convert_element_type_242: "bf16[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_243: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_242, torch.float32);  convert_element_type_242 = None
        neg_48: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.neg.default(convert_element_type_243)
        exp_48: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.exp.default(neg_48);  neg_48 = None
        add_155: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.add.Tensor(exp_48, 1);  exp_48 = None
        div_48: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_243, add_155);  convert_element_type_243 = add_155 = None
        convert_element_type_244: "bf16[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(div_48, torch.bfloat16);  div_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_16: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_244, [-1, -2], True);  convert_element_type_244 = None
        as_strided: "bf16[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.as_strided.default(mean_16, [128, 1280, 1, 1], [1280, 1, 1280, 1280]);  mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view: "bf16[128, 1280][1280, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided, [128, 1280]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/efficientnet.py:344 in forward_head, code: return x if pre_logits else self.classifier(x)
        permute: "bf16[1280, 1000][1, 1280]cuda:0" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        addmm: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg311_1, view, permute);  arg311_1 = view = permute = None
        return (addmm,)
