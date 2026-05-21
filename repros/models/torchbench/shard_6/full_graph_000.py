class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[16, 3, 3, 3][27, 9, 3, 1]cuda:0", arg1_1: "Sym(s21)", arg2_1: "bf16[s21, 3, 224, 224][150528, 50176, 224, 1]cuda:0", arg3_1: "bf16[16][1]cuda:0", arg4_1: "bf16[16][1]cuda:0", arg5_1: "bf16[16][1]cuda:0", arg6_1: "bf16[16][1]cuda:0", arg7_1: "bf16[16, 1, 3, 3][9, 9, 3, 1]cuda:0", arg8_1: "bf16[16][1]cuda:0", arg9_1: "bf16[16][1]cuda:0", arg10_1: "bf16[16][1]cuda:0", arg11_1: "bf16[16][1]cuda:0", arg12_1: "bf16[16, 16, 1, 1][16, 1, 1, 1]cuda:0", arg13_1: "bf16[16][1]cuda:0", arg14_1: "bf16[16][1]cuda:0", arg15_1: "bf16[16][1]cuda:0", arg16_1: "bf16[16][1]cuda:0", arg17_1: "bf16[64, 16, 1, 1][16, 1, 1, 1]cuda:0", arg18_1: "bf16[64][1]cuda:0", arg19_1: "bf16[64][1]cuda:0", arg20_1: "bf16[64][1]cuda:0", arg21_1: "bf16[64][1]cuda:0", arg22_1: "bf16[64, 1, 3, 3][9, 9, 3, 1]cuda:0", arg23_1: "bf16[64][1]cuda:0", arg24_1: "bf16[64][1]cuda:0", arg25_1: "bf16[64][1]cuda:0", arg26_1: "bf16[64][1]cuda:0", arg27_1: "bf16[24, 64, 1, 1][64, 1, 1, 1]cuda:0", arg28_1: "bf16[24][1]cuda:0", arg29_1: "bf16[24][1]cuda:0", arg30_1: "bf16[24][1]cuda:0", arg31_1: "bf16[24][1]cuda:0", arg32_1: "bf16[72, 24, 1, 1][24, 1, 1, 1]cuda:0", arg33_1: "bf16[72][1]cuda:0", arg34_1: "bf16[72][1]cuda:0", arg35_1: "bf16[72][1]cuda:0", arg36_1: "bf16[72][1]cuda:0", arg37_1: "bf16[72, 1, 3, 3][9, 9, 3, 1]cuda:0", arg38_1: "bf16[72][1]cuda:0", arg39_1: "bf16[72][1]cuda:0", arg40_1: "bf16[72][1]cuda:0", arg41_1: "bf16[72][1]cuda:0", arg42_1: "bf16[24, 72, 1, 1][72, 1, 1, 1]cuda:0", arg43_1: "bf16[24][1]cuda:0", arg44_1: "bf16[24][1]cuda:0", arg45_1: "bf16[24][1]cuda:0", arg46_1: "bf16[24][1]cuda:0", arg47_1: "bf16[72, 24, 1, 1][24, 1, 1, 1]cuda:0", arg48_1: "bf16[72][1]cuda:0", arg49_1: "bf16[72][1]cuda:0", arg50_1: "bf16[72][1]cuda:0", arg51_1: "bf16[72][1]cuda:0", arg52_1: "bf16[72, 1, 5, 5][25, 25, 5, 1]cuda:0", arg53_1: "bf16[72][1]cuda:0", arg54_1: "bf16[72][1]cuda:0", arg55_1: "bf16[72][1]cuda:0", arg56_1: "bf16[72][1]cuda:0", arg57_1: "bf16[24, 72, 1, 1][72, 1, 1, 1]cuda:0", arg58_1: "bf16[24][1]cuda:0", arg59_1: "bf16[72, 24, 1, 1][24, 1, 1, 1]cuda:0", arg60_1: "bf16[72][1]cuda:0", arg61_1: "bf16[40, 72, 1, 1][72, 1, 1, 1]cuda:0", arg62_1: "bf16[40][1]cuda:0", arg63_1: "bf16[40][1]cuda:0", arg64_1: "bf16[40][1]cuda:0", arg65_1: "bf16[40][1]cuda:0", arg66_1: "bf16[120, 40, 1, 1][40, 1, 1, 1]cuda:0", arg67_1: "bf16[120][1]cuda:0", arg68_1: "bf16[120][1]cuda:0", arg69_1: "bf16[120][1]cuda:0", arg70_1: "bf16[120][1]cuda:0", arg71_1: "bf16[120, 1, 5, 5][25, 25, 5, 1]cuda:0", arg72_1: "bf16[120][1]cuda:0", arg73_1: "bf16[120][1]cuda:0", arg74_1: "bf16[120][1]cuda:0", arg75_1: "bf16[120][1]cuda:0", arg76_1: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0", arg77_1: "bf16[32][1]cuda:0", arg78_1: "bf16[120, 32, 1, 1][32, 1, 1, 1]cuda:0", arg79_1: "bf16[120][1]cuda:0", arg80_1: "bf16[40, 120, 1, 1][120, 1, 1, 1]cuda:0", arg81_1: "bf16[40][1]cuda:0", arg82_1: "bf16[40][1]cuda:0", arg83_1: "bf16[40][1]cuda:0", arg84_1: "bf16[40][1]cuda:0", arg85_1: "bf16[120, 40, 1, 1][40, 1, 1, 1]cuda:0", arg86_1: "bf16[120][1]cuda:0", arg87_1: "bf16[120][1]cuda:0", arg88_1: "bf16[120][1]cuda:0", arg89_1: "bf16[120][1]cuda:0", arg90_1: "bf16[120, 1, 5, 5][25, 25, 5, 1]cuda:0", arg91_1: "bf16[120][1]cuda:0", arg92_1: "bf16[120][1]cuda:0", arg93_1: "bf16[120][1]cuda:0", arg94_1: "bf16[120][1]cuda:0", arg95_1: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0", arg96_1: "bf16[32][1]cuda:0", arg97_1: "bf16[120, 32, 1, 1][32, 1, 1, 1]cuda:0", arg98_1: "bf16[120][1]cuda:0", arg99_1: "bf16[40, 120, 1, 1][120, 1, 1, 1]cuda:0", arg100_1: "bf16[40][1]cuda:0", arg101_1: "bf16[40][1]cuda:0", arg102_1: "bf16[40][1]cuda:0", arg103_1: "bf16[40][1]cuda:0", arg104_1: "bf16[240, 40, 1, 1][40, 1, 1, 1]cuda:0", arg105_1: "bf16[240][1]cuda:0", arg106_1: "bf16[240][1]cuda:0", arg107_1: "bf16[240][1]cuda:0", arg108_1: "bf16[240][1]cuda:0", arg109_1: "bf16[240, 1, 3, 3][9, 9, 3, 1]cuda:0", arg110_1: "bf16[240][1]cuda:0", arg111_1: "bf16[240][1]cuda:0", arg112_1: "bf16[240][1]cuda:0", arg113_1: "bf16[240][1]cuda:0", arg114_1: "bf16[80, 240, 1, 1][240, 1, 1, 1]cuda:0", arg115_1: "bf16[80][1]cuda:0", arg116_1: "bf16[80][1]cuda:0", arg117_1: "bf16[80][1]cuda:0", arg118_1: "bf16[80][1]cuda:0", arg119_1: "bf16[200, 80, 1, 1][80, 1, 1, 1]cuda:0", arg120_1: "bf16[200][1]cuda:0", arg121_1: "bf16[200][1]cuda:0", arg122_1: "bf16[200][1]cuda:0", arg123_1: "bf16[200][1]cuda:0", arg124_1: "bf16[200, 1, 3, 3][9, 9, 3, 1]cuda:0", arg125_1: "bf16[200][1]cuda:0", arg126_1: "bf16[200][1]cuda:0", arg127_1: "bf16[200][1]cuda:0", arg128_1: "bf16[200][1]cuda:0", arg129_1: "bf16[80, 200, 1, 1][200, 1, 1, 1]cuda:0", arg130_1: "bf16[80][1]cuda:0", arg131_1: "bf16[80][1]cuda:0", arg132_1: "bf16[80][1]cuda:0", arg133_1: "bf16[80][1]cuda:0", arg134_1: "bf16[184, 80, 1, 1][80, 1, 1, 1]cuda:0", arg135_1: "bf16[184][1]cuda:0", arg136_1: "bf16[184][1]cuda:0", arg137_1: "bf16[184][1]cuda:0", arg138_1: "bf16[184][1]cuda:0", arg139_1: "bf16[184, 1, 3, 3][9, 9, 3, 1]cuda:0", arg140_1: "bf16[184][1]cuda:0", arg141_1: "bf16[184][1]cuda:0", arg142_1: "bf16[184][1]cuda:0", arg143_1: "bf16[184][1]cuda:0", arg144_1: "bf16[80, 184, 1, 1][184, 1, 1, 1]cuda:0", arg145_1: "bf16[80][1]cuda:0", arg146_1: "bf16[80][1]cuda:0", arg147_1: "bf16[80][1]cuda:0", arg148_1: "bf16[80][1]cuda:0", arg149_1: "bf16[184, 80, 1, 1][80, 1, 1, 1]cuda:0", arg150_1: "bf16[184][1]cuda:0", arg151_1: "bf16[184][1]cuda:0", arg152_1: "bf16[184][1]cuda:0", arg153_1: "bf16[184][1]cuda:0", arg154_1: "bf16[184, 1, 3, 3][9, 9, 3, 1]cuda:0", arg155_1: "bf16[184][1]cuda:0", arg156_1: "bf16[184][1]cuda:0", arg157_1: "bf16[184][1]cuda:0", arg158_1: "bf16[184][1]cuda:0", arg159_1: "bf16[80, 184, 1, 1][184, 1, 1, 1]cuda:0", arg160_1: "bf16[80][1]cuda:0", arg161_1: "bf16[80][1]cuda:0", arg162_1: "bf16[80][1]cuda:0", arg163_1: "bf16[80][1]cuda:0", arg164_1: "bf16[480, 80, 1, 1][80, 1, 1, 1]cuda:0", arg165_1: "bf16[480][1]cuda:0", arg166_1: "bf16[480][1]cuda:0", arg167_1: "bf16[480][1]cuda:0", arg168_1: "bf16[480][1]cuda:0", arg169_1: "bf16[480, 1, 3, 3][9, 9, 3, 1]cuda:0", arg170_1: "bf16[480][1]cuda:0", arg171_1: "bf16[480][1]cuda:0", arg172_1: "bf16[480][1]cuda:0", arg173_1: "bf16[480][1]cuda:0", arg174_1: "bf16[120, 480, 1, 1][480, 1, 1, 1]cuda:0", arg175_1: "bf16[120][1]cuda:0", arg176_1: "bf16[480, 120, 1, 1][120, 1, 1, 1]cuda:0", arg177_1: "bf16[480][1]cuda:0", arg178_1: "bf16[112, 480, 1, 1][480, 1, 1, 1]cuda:0", arg179_1: "bf16[112][1]cuda:0", arg180_1: "bf16[112][1]cuda:0", arg181_1: "bf16[112][1]cuda:0", arg182_1: "bf16[112][1]cuda:0", arg183_1: "bf16[672, 112, 1, 1][112, 1, 1, 1]cuda:0", arg184_1: "bf16[672][1]cuda:0", arg185_1: "bf16[672][1]cuda:0", arg186_1: "bf16[672][1]cuda:0", arg187_1: "bf16[672][1]cuda:0", arg188_1: "bf16[672, 1, 3, 3][9, 9, 3, 1]cuda:0", arg189_1: "bf16[672][1]cuda:0", arg190_1: "bf16[672][1]cuda:0", arg191_1: "bf16[672][1]cuda:0", arg192_1: "bf16[672][1]cuda:0", arg193_1: "bf16[168, 672, 1, 1][672, 1, 1, 1]cuda:0", arg194_1: "bf16[168][1]cuda:0", arg195_1: "bf16[672, 168, 1, 1][168, 1, 1, 1]cuda:0", arg196_1: "bf16[672][1]cuda:0", arg197_1: "bf16[112, 672, 1, 1][672, 1, 1, 1]cuda:0", arg198_1: "bf16[112][1]cuda:0", arg199_1: "bf16[112][1]cuda:0", arg200_1: "bf16[112][1]cuda:0", arg201_1: "bf16[112][1]cuda:0", arg202_1: "bf16[672, 112, 1, 1][112, 1, 1, 1]cuda:0", arg203_1: "bf16[672][1]cuda:0", arg204_1: "bf16[672][1]cuda:0", arg205_1: "bf16[672][1]cuda:0", arg206_1: "bf16[672][1]cuda:0", arg207_1: "bf16[672, 1, 5, 5][25, 25, 5, 1]cuda:0", arg208_1: "bf16[672][1]cuda:0", arg209_1: "bf16[672][1]cuda:0", arg210_1: "bf16[672][1]cuda:0", arg211_1: "bf16[672][1]cuda:0", arg212_1: "bf16[168, 672, 1, 1][672, 1, 1, 1]cuda:0", arg213_1: "bf16[168][1]cuda:0", arg214_1: "bf16[672, 168, 1, 1][168, 1, 1, 1]cuda:0", arg215_1: "bf16[672][1]cuda:0", arg216_1: "bf16[160, 672, 1, 1][672, 1, 1, 1]cuda:0", arg217_1: "bf16[160][1]cuda:0", arg218_1: "bf16[160][1]cuda:0", arg219_1: "bf16[160][1]cuda:0", arg220_1: "bf16[160][1]cuda:0", arg221_1: "bf16[960, 160, 1, 1][160, 1, 1, 1]cuda:0", arg222_1: "bf16[960][1]cuda:0", arg223_1: "bf16[960][1]cuda:0", arg224_1: "bf16[960][1]cuda:0", arg225_1: "bf16[960][1]cuda:0", arg226_1: "bf16[960, 1, 5, 5][25, 25, 5, 1]cuda:0", arg227_1: "bf16[960][1]cuda:0", arg228_1: "bf16[960][1]cuda:0", arg229_1: "bf16[960][1]cuda:0", arg230_1: "bf16[960][1]cuda:0", arg231_1: "bf16[240, 960, 1, 1][960, 1, 1, 1]cuda:0", arg232_1: "bf16[240][1]cuda:0", arg233_1: "bf16[960, 240, 1, 1][240, 1, 1, 1]cuda:0", arg234_1: "bf16[960][1]cuda:0", arg235_1: "bf16[160, 960, 1, 1][960, 1, 1, 1]cuda:0", arg236_1: "bf16[160][1]cuda:0", arg237_1: "bf16[160][1]cuda:0", arg238_1: "bf16[160][1]cuda:0", arg239_1: "bf16[160][1]cuda:0", arg240_1: "bf16[960, 160, 1, 1][160, 1, 1, 1]cuda:0", arg241_1: "bf16[960][1]cuda:0", arg242_1: "bf16[960][1]cuda:0", arg243_1: "bf16[960][1]cuda:0", arg244_1: "bf16[960][1]cuda:0", arg245_1: "bf16[960, 1, 5, 5][25, 25, 5, 1]cuda:0", arg246_1: "bf16[960][1]cuda:0", arg247_1: "bf16[960][1]cuda:0", arg248_1: "bf16[960][1]cuda:0", arg249_1: "bf16[960][1]cuda:0", arg250_1: "bf16[240, 960, 1, 1][960, 1, 1, 1]cuda:0", arg251_1: "bf16[240][1]cuda:0", arg252_1: "bf16[960, 240, 1, 1][240, 1, 1, 1]cuda:0", arg253_1: "bf16[960][1]cuda:0", arg254_1: "bf16[160, 960, 1, 1][960, 1, 1, 1]cuda:0", arg255_1: "bf16[160][1]cuda:0", arg256_1: "bf16[160][1]cuda:0", arg257_1: "bf16[160][1]cuda:0", arg258_1: "bf16[160][1]cuda:0", arg259_1: "bf16[960, 160, 1, 1][160, 1, 1, 1]cuda:0", arg260_1: "bf16[960][1]cuda:0", arg261_1: "bf16[960][1]cuda:0", arg262_1: "bf16[960][1]cuda:0", arg263_1: "bf16[960][1]cuda:0", arg264_1: "bf16[1280, 960][960, 1]cuda:0", arg265_1: "bf16[1280][1]cuda:0", arg266_1: "bf16[1000, 1280][1280, 1]cuda:0", arg267_1: "bf16[1000][1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:211 in _forward_impl, code: x = self.features(x)
        convolution: "bf16[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(arg2_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = None
        convert_element_type: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        unsqueeze: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub_1: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        convert_element_type_1: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        add_5: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1, 0.001);  convert_element_type_1 = None
        sqrt: "f32[16][1]cuda:0" = torch.ops.aten.sqrt.default(add_5);  add_5 = None
        reciprocal: "f32[16][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul_5: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_5, -1);  mul_5 = None
        unsqueeze_3: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_6: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_3);  sub_1 = unsqueeze_3 = None
        unsqueeze_4: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_5: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_7: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, unsqueeze_5);  mul_6 = unsqueeze_5 = None
        unsqueeze_6: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_7: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_6: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_7, unsqueeze_7);  mul_7 = unsqueeze_7 = None
        add_38: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(add_6, 3)
        clamp_min: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_38, 0);  add_38 = None
        clamp_max: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul_138: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_6, clamp_max);  add_6 = clamp_max = None
        div: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_138, 6);  mul_138 = None
        convert_element_type_4: "bf16[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_1: "bf16[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_4, arg7_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  arg7_1 = None
        convert_element_type_5: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        unsqueeze_8: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, -1);  convert_element_type_5 = None
        unsqueeze_9: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_8: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        convert_element_type_6: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        add_49: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_6, 0.001);  convert_element_type_6 = None
        sqrt_1: "f32[16][1]cuda:0" = torch.ops.aten.sqrt.default(add_49);  add_49 = None
        reciprocal_1: "f32[16][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_146: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_146, -1);  mul_146 = None
        unsqueeze_11: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_147: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_11);  sub_8 = unsqueeze_11 = None
        unsqueeze_12: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_13: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_148: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_13);  mul_147 = unsqueeze_13 = None
        unsqueeze_14: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_15: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_50: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_148, unsqueeze_15);  mul_148 = unsqueeze_15 = None
        convert_element_type_7: "bf16[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None
        relu: "bf16[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_7);  convert_element_type_7 = None
        convolution_2: "bf16[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(relu, arg12_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu = arg12_1 = None
        convert_element_type_8: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        unsqueeze_16: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
        unsqueeze_17: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_13: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        convert_element_type_9: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg14_1, torch.float32);  arg14_1 = None
        add_71: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_9, 0.001);  convert_element_type_9 = None
        sqrt_2: "f32[16][1]cuda:0" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
        reciprocal_2: "f32[16][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_160: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_160, -1);  mul_160 = None
        unsqueeze_19: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_161: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_19);  sub_13 = unsqueeze_19 = None
        unsqueeze_20: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_21: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_162: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_21);  mul_161 = unsqueeze_21 = None
        unsqueeze_22: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_23: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_72: "f32[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_23);  mul_162 = unsqueeze_23 = None
        convert_element_type_10: "bf16[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_88: "bf16[s21, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_10, convert_element_type_4);  convert_element_type_10 = convert_element_type_4 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_3: "bf16[s21, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(add_88, arg17_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_88 = arg17_1 = None
        convert_element_type_11: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg18_1, torch.float32);  arg18_1 = None
        unsqueeze_24: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_11, -1);  convert_element_type_11 = None
        unsqueeze_25: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_19: "f32[s21, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        convert_element_type_12: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg19_1, torch.float32);  arg19_1 = None
        add_99: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_12, 0.001);  convert_element_type_12 = None
        sqrt_3: "f32[64][1]cuda:0" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
        reciprocal_3: "f32[64][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_176: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_176, -1);  mul_176 = None
        unsqueeze_27: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_177: "f32[s21, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_27);  sub_19 = unsqueeze_27 = None
        unsqueeze_28: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_29: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_178: "f32[s21, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, unsqueeze_29);  mul_177 = unsqueeze_29 = None
        unsqueeze_30: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_31: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_100: "f32[s21, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_178, unsqueeze_31);  mul_178 = unsqueeze_31 = None
        convert_element_type_13: "bf16[s21, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16);  add_100 = None
        relu_1: "bf16[s21, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_13);  convert_element_type_13 = None
        convolution_4: "bf16[s21, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, arg22_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  relu_1 = arg22_1 = None
        convert_element_type_14: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg23_1, torch.float32);  arg23_1 = None
        unsqueeze_32: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
        unsqueeze_33: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        sub_24: "f32[s21, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        convert_element_type_15: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg24_1, torch.float32);  arg24_1 = None
        add_121: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_15, 0.001);  convert_element_type_15 = None
        sqrt_4: "f32[64][1]cuda:0" = torch.ops.aten.sqrt.default(add_121);  add_121 = None
        reciprocal_4: "f32[64][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_190: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_34: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_190, -1);  mul_190 = None
        unsqueeze_35: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_191: "f32[s21, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_35);  sub_24 = unsqueeze_35 = None
        unsqueeze_36: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_37: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_192: "f32[s21, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_37);  mul_191 = unsqueeze_37 = None
        unsqueeze_38: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_39: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_122: "f32[s21, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_192, unsqueeze_39);  mul_192 = unsqueeze_39 = None
        convert_element_type_16: "bf16[s21, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_122, torch.bfloat16);  add_122 = None
        relu_2: "bf16[s21, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_16);  convert_element_type_16 = None
        convolution_5: "bf16[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, arg27_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg27_1 = None
        convert_element_type_17: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg28_1, torch.float32);  arg28_1 = None
        unsqueeze_40: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_17, -1);  convert_element_type_17 = None
        unsqueeze_41: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        sub_29: "f32[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        convert_element_type_18: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg29_1, torch.float32);  arg29_1 = None
        add_143: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_18, 0.001);  convert_element_type_18 = None
        sqrt_5: "f32[24][1]cuda:0" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_5: "f32[24][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_204: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_42: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
        unsqueeze_43: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_205: "f32[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_43);  sub_29 = unsqueeze_43 = None
        unsqueeze_44: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_45: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_206: "f32[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_45);  mul_205 = unsqueeze_45 = None
        unsqueeze_46: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_47: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_144: "f32[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_47);  mul_206 = unsqueeze_47 = None
        convert_element_type_19: "bf16[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16);  add_144 = None
        convolution_6: "bf16[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_19, arg32_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg32_1 = None
        convert_element_type_20: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg33_1, torch.float32);  arg33_1 = None
        unsqueeze_48: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
        unsqueeze_49: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        sub_32: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        convert_element_type_21: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg34_1, torch.float32);  arg34_1 = None
        add_155: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_21, 0.001);  convert_element_type_21 = None
        sqrt_6: "f32[72][1]cuda:0" = torch.ops.aten.sqrt.default(add_155);  add_155 = None
        reciprocal_6: "f32[72][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_214: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_50: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_214, -1);  mul_214 = None
        unsqueeze_51: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_215: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_51);  sub_32 = unsqueeze_51 = None
        unsqueeze_52: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_53: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_216: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_215, unsqueeze_53);  mul_215 = unsqueeze_53 = None
        unsqueeze_54: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_55: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_156: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_55);  mul_216 = unsqueeze_55 = None
        convert_element_type_22: "bf16[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_156, torch.bfloat16);  add_156 = None
        relu_3: "bf16[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_22);  convert_element_type_22 = None
        convolution_7: "bf16[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, arg37_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72);  relu_3 = arg37_1 = None
        convert_element_type_23: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg38_1, torch.float32);  arg38_1 = None
        unsqueeze_56: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_23, -1);  convert_element_type_23 = None
        unsqueeze_57: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        sub_37: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        convert_element_type_24: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg39_1, torch.float32);  arg39_1 = None
        add_177: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_24, 0.001);  convert_element_type_24 = None
        sqrt_7: "f32[72][1]cuda:0" = torch.ops.aten.sqrt.default(add_177);  add_177 = None
        reciprocal_7: "f32[72][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_228: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_58: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
        unsqueeze_59: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_229: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_59);  sub_37 = unsqueeze_59 = None
        unsqueeze_60: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_61: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_230: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_61);  mul_229 = unsqueeze_61 = None
        unsqueeze_62: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_63: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_178: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_63);  mul_230 = unsqueeze_63 = None
        convert_element_type_25: "bf16[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_178, torch.bfloat16);  add_178 = None
        relu_4: "bf16[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_25);  convert_element_type_25 = None
        convolution_8: "bf16[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, arg42_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_4 = arg42_1 = None
        convert_element_type_26: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg43_1, torch.float32);  arg43_1 = None
        unsqueeze_64: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
        unsqueeze_65: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        sub_42: "f32[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
        convert_element_type_27: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg44_1, torch.float32);  arg44_1 = None
        add_199: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_27, 0.001);  convert_element_type_27 = None
        sqrt_8: "f32[24][1]cuda:0" = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_8: "f32[24][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_242: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_66: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_242, -1);  mul_242 = None
        unsqueeze_67: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_243: "f32[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_67);  sub_42 = unsqueeze_67 = None
        unsqueeze_68: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_69: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_244: "f32[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, unsqueeze_69);  mul_243 = unsqueeze_69 = None
        unsqueeze_70: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
        unsqueeze_71: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_200: "f32[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_71);  mul_244 = unsqueeze_71 = None
        convert_element_type_28: "bf16[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_200, torch.bfloat16);  add_200 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_216: "bf16[s21, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_28, convert_element_type_19);  convert_element_type_28 = convert_element_type_19 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_9: "bf16[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(add_216, arg47_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_216 = arg47_1 = None
        convert_element_type_29: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg48_1, torch.float32);  arg48_1 = None
        unsqueeze_72: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_29, -1);  convert_element_type_29 = None
        unsqueeze_73: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        sub_48: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
        convert_element_type_30: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg49_1, torch.float32);  arg49_1 = None
        add_227: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_30, 0.001);  convert_element_type_30 = None
        sqrt_9: "f32[72][1]cuda:0" = torch.ops.aten.sqrt.default(add_227);  add_227 = None
        reciprocal_9: "f32[72][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_258: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_74: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
        unsqueeze_75: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_259: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_75);  sub_48 = unsqueeze_75 = None
        unsqueeze_76: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_77: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_260: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_77);  mul_259 = unsqueeze_77 = None
        unsqueeze_78: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_79: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_228: "f32[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_79);  mul_260 = unsqueeze_79 = None
        convert_element_type_31: "bf16[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_228, torch.bfloat16);  add_228 = None
        relu_5: "bf16[s21, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_31);  convert_element_type_31 = None
        convolution_10: "bf16[s21, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, arg52_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72);  relu_5 = arg52_1 = None
        convert_element_type_32: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg53_1, torch.float32);  arg53_1 = None
        unsqueeze_80: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
        unsqueeze_81: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        sub_53: "f32[s21, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
        convert_element_type_33: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg54_1, torch.float32);  arg54_1 = None
        add_249: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_33, 0.001);  convert_element_type_33 = None
        sqrt_10: "f32[72][1]cuda:0" = torch.ops.aten.sqrt.default(add_249);  add_249 = None
        reciprocal_10: "f32[72][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_272: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_82: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_272, -1);  mul_272 = None
        unsqueeze_83: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_273: "f32[s21, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_83);  sub_53 = unsqueeze_83 = None
        unsqueeze_84: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_85: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_274: "f32[s21, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_85);  mul_273 = unsqueeze_85 = None
        unsqueeze_86: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_87: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_250: "f32[s21, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_274, unsqueeze_87);  mul_274 = unsqueeze_87 = None
        convert_element_type_34: "bf16[s21, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_250, torch.bfloat16);  add_250 = None
        relu_6: "bf16[s21, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_34);  convert_element_type_34 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean: "bf16[s21, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_6, [-1, -2], True)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_11: "bf16[s21, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean, arg57_1, arg58_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg57_1 = arg58_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_7: "bf16[s21, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_12: "bf16[s21, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, arg59_1, arg60_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_7 = arg59_1 = arg60_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_35: "f32[s21, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        add_286: "f32[s21, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_35, 3);  convert_element_type_35 = None
        clamp_min_1: "f32[s21, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_286, 0);  add_286 = None
        clamp_max_1: "f32[s21, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
        div_1: "f32[s21, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
        convert_element_type_36: "bf16[s21, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_291: "bf16[s21, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_36, relu_6);  convert_element_type_36 = relu_6 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_13: "bf16[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(mul_291, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_291 = arg61_1 = None
        convert_element_type_37: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg62_1, torch.float32);  arg62_1 = None
        unsqueeze_88: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_37, -1);  convert_element_type_37 = None
        unsqueeze_89: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        sub_64: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_89);  convolution_13 = unsqueeze_89 = None
        convert_element_type_38: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg63_1, torch.float32);  arg63_1 = None
        add_302: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_38, 0.001);  convert_element_type_38 = None
        sqrt_11: "f32[40][1]cuda:0" = torch.ops.aten.sqrt.default(add_302);  add_302 = None
        reciprocal_11: "f32[40][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_299: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_90: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_299, -1);  mul_299 = None
        unsqueeze_91: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_300: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_91);  sub_64 = unsqueeze_91 = None
        unsqueeze_92: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_93: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_301: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_300, unsqueeze_93);  mul_300 = unsqueeze_93 = None
        unsqueeze_94: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_95: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_303: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_301, unsqueeze_95);  mul_301 = unsqueeze_95 = None
        convert_element_type_39: "bf16[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_303, torch.bfloat16);  add_303 = None
        convolution_14: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_39, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg66_1 = None
        convert_element_type_40: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg67_1, torch.float32);  arg67_1 = None
        unsqueeze_96: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
        unsqueeze_97: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        sub_67: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_97);  convolution_14 = unsqueeze_97 = None
        convert_element_type_41: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg68_1, torch.float32);  arg68_1 = None
        add_314: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_41, 0.001);  convert_element_type_41 = None
        sqrt_12: "f32[120][1]cuda:0" = torch.ops.aten.sqrt.default(add_314);  add_314 = None
        reciprocal_12: "f32[120][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_309: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_98: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
        unsqueeze_99: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_310: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_99);  sub_67 = unsqueeze_99 = None
        unsqueeze_100: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_101: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_311: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_101);  mul_310 = unsqueeze_101 = None
        unsqueeze_102: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_103: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_315: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_311, unsqueeze_103);  mul_311 = unsqueeze_103 = None
        convert_element_type_42: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_315, torch.bfloat16);  add_315 = None
        relu_8: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_42);  convert_element_type_42 = None
        convolution_15: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, arg71_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  relu_8 = arg71_1 = None
        convert_element_type_43: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg72_1, torch.float32);  arg72_1 = None
        unsqueeze_104: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_43, -1);  convert_element_type_43 = None
        unsqueeze_105: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        sub_72: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_105);  convolution_15 = unsqueeze_105 = None
        convert_element_type_44: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg73_1, torch.float32);  arg73_1 = None
        add_336: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_44, 0.001);  convert_element_type_44 = None
        sqrt_13: "f32[120][1]cuda:0" = torch.ops.aten.sqrt.default(add_336);  add_336 = None
        reciprocal_13: "f32[120][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_323: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_106: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_323, -1);  mul_323 = None
        unsqueeze_107: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_324: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_107);  sub_72 = unsqueeze_107 = None
        unsqueeze_108: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_109: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_325: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_324, unsqueeze_109);  mul_324 = unsqueeze_109 = None
        unsqueeze_110: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_111: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_337: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_325, unsqueeze_111);  mul_325 = unsqueeze_111 = None
        convert_element_type_45: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_337, torch.bfloat16);  add_337 = None
        relu_9: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_45);  convert_element_type_45 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_1: "bf16[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_9, [-1, -2], True)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_16: "bf16[s21, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_1, arg76_1, arg77_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg76_1 = arg77_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_10: "bf16[s21, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_17: "bf16[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, arg78_1, arg79_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_10 = arg78_1 = arg79_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_46: "f32[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        add_373: "f32[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_46, 3);  convert_element_type_46 = None
        clamp_min_2: "f32[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_373, 0);  add_373 = None
        clamp_max_2: "f32[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
        div_2: "f32[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
        convert_element_type_47: "bf16[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_342: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_47, relu_9);  convert_element_type_47 = relu_9 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_18: "bf16[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(mul_342, arg80_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_342 = arg80_1 = None
        convert_element_type_48: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg81_1, torch.float32);  arg81_1 = None
        unsqueeze_112: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
        unsqueeze_113: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        sub_83: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_113);  convolution_18 = unsqueeze_113 = None
        convert_element_type_49: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg82_1, torch.float32);  arg82_1 = None
        add_389: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_49, 0.001);  convert_element_type_49 = None
        sqrt_14: "f32[40][1]cuda:0" = torch.ops.aten.sqrt.default(add_389);  add_389 = None
        reciprocal_14: "f32[40][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_350: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_114: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_350, -1);  mul_350 = None
        unsqueeze_115: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_351: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_115);  sub_83 = unsqueeze_115 = None
        unsqueeze_116: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_117: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_352: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_351, unsqueeze_117);  mul_351 = unsqueeze_117 = None
        unsqueeze_118: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_119: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_390: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_352, unsqueeze_119);  mul_352 = unsqueeze_119 = None
        convert_element_type_50: "bf16[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_390, torch.bfloat16);  add_390 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_406: "bf16[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_50, convert_element_type_39);  convert_element_type_50 = convert_element_type_39 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_19: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(add_406, arg85_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg85_1 = None
        convert_element_type_51: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg86_1, torch.float32);  arg86_1 = None
        unsqueeze_120: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_51, -1);  convert_element_type_51 = None
        unsqueeze_121: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        sub_89: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_121);  convolution_19 = unsqueeze_121 = None
        convert_element_type_52: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg87_1, torch.float32);  arg87_1 = None
        add_417: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_52, 0.001);  convert_element_type_52 = None
        sqrt_15: "f32[120][1]cuda:0" = torch.ops.aten.sqrt.default(add_417);  add_417 = None
        reciprocal_15: "f32[120][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_366: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_122: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
        unsqueeze_123: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_367: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_123);  sub_89 = unsqueeze_123 = None
        unsqueeze_124: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_125: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_368: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_125);  mul_367 = unsqueeze_125 = None
        unsqueeze_126: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_127: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_418: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_368, unsqueeze_127);  mul_368 = unsqueeze_127 = None
        convert_element_type_53: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_418, torch.bfloat16);  add_418 = None
        relu_11: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_53);  convert_element_type_53 = None
        convolution_20: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_11, arg90_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  relu_11 = arg90_1 = None
        convert_element_type_54: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg91_1, torch.float32);  arg91_1 = None
        unsqueeze_128: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
        unsqueeze_129: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        sub_94: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_129);  convolution_20 = unsqueeze_129 = None
        convert_element_type_55: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg92_1, torch.float32);  arg92_1 = None
        add_439: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_55, 0.001);  convert_element_type_55 = None
        sqrt_16: "f32[120][1]cuda:0" = torch.ops.aten.sqrt.default(add_439);  add_439 = None
        reciprocal_16: "f32[120][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_380: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_130: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_380, -1);  mul_380 = None
        unsqueeze_131: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_381: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_131);  sub_94 = unsqueeze_131 = None
        unsqueeze_132: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_133: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_382: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, unsqueeze_133);  mul_381 = unsqueeze_133 = None
        unsqueeze_134: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_135: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_440: "f32[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_382, unsqueeze_135);  mul_382 = unsqueeze_135 = None
        convert_element_type_56: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_440, torch.bfloat16);  add_440 = None
        relu_12: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_56);  convert_element_type_56 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_2: "bf16[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_12, [-1, -2], True)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_21: "bf16[s21, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_2, arg95_1, arg96_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg95_1 = arg96_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_13: "bf16[s21, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_22: "bf16[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, arg97_1, arg98_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_13 = arg97_1 = arg98_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_57: "f32[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        add_476: "f32[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_57, 3);  convert_element_type_57 = None
        clamp_min_3: "f32[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_476, 0);  add_476 = None
        clamp_max_3: "f32[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
        div_3: "f32[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
        convert_element_type_58: "bf16[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_399: "bf16[s21, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, relu_12);  convert_element_type_58 = relu_12 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_23: "bf16[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(mul_399, arg99_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_399 = arg99_1 = None
        convert_element_type_59: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg100_1, torch.float32);  arg100_1 = None
        unsqueeze_136: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_59, -1);  convert_element_type_59 = None
        unsqueeze_137: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        sub_105: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_137);  convolution_23 = unsqueeze_137 = None
        convert_element_type_60: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg101_1, torch.float32);  arg101_1 = None
        add_492: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_60, 0.001);  convert_element_type_60 = None
        sqrt_17: "f32[40][1]cuda:0" = torch.ops.aten.sqrt.default(add_492);  add_492 = None
        reciprocal_17: "f32[40][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_407: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        unsqueeze_138: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_407, -1);  mul_407 = None
        unsqueeze_139: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        mul_408: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_139);  sub_105 = unsqueeze_139 = None
        unsqueeze_140: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_141: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_409: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_141);  mul_408 = unsqueeze_141 = None
        unsqueeze_142: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_143: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_493: "f32[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_409, unsqueeze_143);  mul_409 = unsqueeze_143 = None
        convert_element_type_61: "bf16[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_493, torch.bfloat16);  add_493 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_509: "bf16[s21, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_61, add_406);  convert_element_type_61 = add_406 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_24: "bf16[s21, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(add_509, arg104_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_509 = arg104_1 = None
        convert_element_type_62: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg105_1, torch.float32);  arg105_1 = None
        unsqueeze_144: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
        unsqueeze_145: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        sub_111: "f32[s21, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_145);  convolution_24 = unsqueeze_145 = None
        convert_element_type_63: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg106_1, torch.float32);  arg106_1 = None
        add_520: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_63, 0.001);  convert_element_type_63 = None
        sqrt_18: "f32[240][1]cuda:0" = torch.ops.aten.sqrt.default(add_520);  add_520 = None
        reciprocal_18: "f32[240][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_423: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        unsqueeze_146: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_423, -1);  mul_423 = None
        unsqueeze_147: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        mul_424: "f32[s21, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_147);  sub_111 = unsqueeze_147 = None
        unsqueeze_148: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_149: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_425: "f32[s21, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_424, unsqueeze_149);  mul_424 = unsqueeze_149 = None
        unsqueeze_150: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_151: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_521: "f32[s21, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_425, unsqueeze_151);  mul_425 = unsqueeze_151 = None
        add_553: "f32[s21, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_521, 3)
        clamp_min_4: "f32[s21, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_553, 0);  add_553 = None
        clamp_max_4: "f32[s21, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
        mul_556: "f32[s21, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_521, clamp_max_4);  add_521 = clamp_max_4 = None
        div_4: "f32[s21, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_556, 6);  mul_556 = None
        convert_element_type_66: "bf16[s21, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        convolution_25: "bf16[s21, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_66, arg109_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240);  convert_element_type_66 = arg109_1 = None
        convert_element_type_67: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg110_1, torch.float32);  arg110_1 = None
        unsqueeze_152: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_67, -1);  convert_element_type_67 = None
        unsqueeze_153: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        sub_118: "f32[s21, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_153);  convolution_25 = unsqueeze_153 = None
        convert_element_type_68: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg111_1, torch.float32);  arg111_1 = None
        add_564: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_68, 0.001);  convert_element_type_68 = None
        sqrt_19: "f32[240][1]cuda:0" = torch.ops.aten.sqrt.default(add_564);  add_564 = None
        reciprocal_19: "f32[240][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_564: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        unsqueeze_154: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_564, -1);  mul_564 = None
        unsqueeze_155: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        mul_565: "f32[s21, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_155);  sub_118 = unsqueeze_155 = None
        unsqueeze_156: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_157: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_566: "f32[s21, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_565, unsqueeze_157);  mul_565 = unsqueeze_157 = None
        unsqueeze_158: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_159: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_565: "f32[s21, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_159);  mul_566 = unsqueeze_159 = None
        add_597: "f32[s21, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_565, 3)
        clamp_min_5: "f32[s21, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_597, 0);  add_597 = None
        clamp_max_5: "f32[s21, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
        mul_697: "f32[s21, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_565, clamp_max_5);  add_565 = clamp_max_5 = None
        div_5: "f32[s21, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_697, 6);  mul_697 = None
        convert_element_type_71: "bf16[s21, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        convolution_26: "bf16[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_71, arg114_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_71 = arg114_1 = None
        convert_element_type_72: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg115_1, torch.float32);  arg115_1 = None
        unsqueeze_160: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
        unsqueeze_161: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        sub_125: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_161);  convolution_26 = unsqueeze_161 = None
        convert_element_type_73: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg116_1, torch.float32);  arg116_1 = None
        add_608: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_73, 0.001);  convert_element_type_73 = None
        sqrt_20: "f32[80][1]cuda:0" = torch.ops.aten.sqrt.default(add_608);  add_608 = None
        reciprocal_20: "f32[80][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        mul_705: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        unsqueeze_162: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_705, -1);  mul_705 = None
        unsqueeze_163: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        mul_706: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_163);  sub_125 = unsqueeze_163 = None
        unsqueeze_164: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_165: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_707: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_706, unsqueeze_165);  mul_706 = unsqueeze_165 = None
        unsqueeze_166: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_167: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_609: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_707, unsqueeze_167);  mul_707 = unsqueeze_167 = None
        convert_element_type_74: "bf16[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_609, torch.bfloat16);  add_609 = None
        convolution_27: "bf16[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_74, arg119_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg119_1 = None
        convert_element_type_75: "f32[200][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg120_1, torch.float32);  arg120_1 = None
        unsqueeze_168: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_75, -1);  convert_element_type_75 = None
        unsqueeze_169: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        sub_128: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_169);  convolution_27 = unsqueeze_169 = None
        convert_element_type_76: "f32[200][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg121_1, torch.float32);  arg121_1 = None
        add_620: "f32[200][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_76, 0.001);  convert_element_type_76 = None
        sqrt_21: "f32[200][1]cuda:0" = torch.ops.aten.sqrt.default(add_620);  add_620 = None
        reciprocal_21: "f32[200][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        mul_715: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        unsqueeze_170: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_715, -1);  mul_715 = None
        unsqueeze_171: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        mul_716: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_171);  sub_128 = unsqueeze_171 = None
        unsqueeze_172: "bf16[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_173: "bf16[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_717: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_716, unsqueeze_173);  mul_716 = unsqueeze_173 = None
        unsqueeze_174: "bf16[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_175: "bf16[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_621: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_717, unsqueeze_175);  mul_717 = unsqueeze_175 = None
        add_653: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_621, 3)
        clamp_min_6: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_653, 0);  add_653 = None
        clamp_max_6: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
        mul_848: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_621, clamp_max_6);  add_621 = clamp_max_6 = None
        div_6: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_848, 6);  mul_848 = None
        convert_element_type_79: "bf16[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        convolution_28: "bf16[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_79, arg124_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 200);  convert_element_type_79 = arg124_1 = None
        convert_element_type_80: "f32[200][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg125_1, torch.float32);  arg125_1 = None
        unsqueeze_176: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
        unsqueeze_177: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        sub_135: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_177);  convolution_28 = unsqueeze_177 = None
        convert_element_type_81: "f32[200][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg126_1, torch.float32);  arg126_1 = None
        add_664: "f32[200][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_81, 0.001);  convert_element_type_81 = None
        sqrt_22: "f32[200][1]cuda:0" = torch.ops.aten.sqrt.default(add_664);  add_664 = None
        reciprocal_22: "f32[200][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        mul_856: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        unsqueeze_178: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_856, -1);  mul_856 = None
        unsqueeze_179: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        mul_857: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_179);  sub_135 = unsqueeze_179 = None
        unsqueeze_180: "bf16[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_181: "bf16[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_858: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_857, unsqueeze_181);  mul_857 = unsqueeze_181 = None
        unsqueeze_182: "bf16[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_183: "bf16[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_665: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_858, unsqueeze_183);  mul_858 = unsqueeze_183 = None
        add_697: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_665, 3)
        clamp_min_7: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_697, 0);  add_697 = None
        clamp_max_7: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
        mul_989: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_665, clamp_max_7);  add_665 = clamp_max_7 = None
        div_7: "f32[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_989, 6);  mul_989 = None
        convert_element_type_84: "bf16[s21, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        convolution_29: "bf16[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_84, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_84 = arg129_1 = None
        convert_element_type_85: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg130_1, torch.float32);  arg130_1 = None
        unsqueeze_184: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_85, -1);  convert_element_type_85 = None
        unsqueeze_185: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        sub_142: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_185);  convolution_29 = unsqueeze_185 = None
        convert_element_type_86: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg131_1, torch.float32);  arg131_1 = None
        add_708: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_86, 0.001);  convert_element_type_86 = None
        sqrt_23: "f32[80][1]cuda:0" = torch.ops.aten.sqrt.default(add_708);  add_708 = None
        reciprocal_23: "f32[80][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_997: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        unsqueeze_186: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_997, -1);  mul_997 = None
        unsqueeze_187: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        mul_998: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_187);  sub_142 = unsqueeze_187 = None
        unsqueeze_188: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_189: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_999: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_998, unsqueeze_189);  mul_998 = unsqueeze_189 = None
        unsqueeze_190: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_191: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_709: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_999, unsqueeze_191);  mul_999 = unsqueeze_191 = None
        convert_element_type_87: "bf16[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_709, torch.bfloat16);  add_709 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_725: "bf16[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_87, convert_element_type_74);  convert_element_type_87 = convert_element_type_74 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_30: "bf16[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(add_725, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg134_1 = None
        convert_element_type_88: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg135_1, torch.float32);  arg135_1 = None
        unsqueeze_192: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
        unsqueeze_193: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        sub_148: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_193);  convolution_30 = unsqueeze_193 = None
        convert_element_type_89: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg136_1, torch.float32);  arg136_1 = None
        add_736: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_89, 0.001);  convert_element_type_89 = None
        sqrt_24: "f32[184][1]cuda:0" = torch.ops.aten.sqrt.default(add_736);  add_736 = None
        reciprocal_24: "f32[184][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_1013: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        unsqueeze_194: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1013, -1);  mul_1013 = None
        unsqueeze_195: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        mul_1014: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_195);  sub_148 = unsqueeze_195 = None
        unsqueeze_196: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_197: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_1015: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1014, unsqueeze_197);  mul_1014 = unsqueeze_197 = None
        unsqueeze_198: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_199: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_737: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1015, unsqueeze_199);  mul_1015 = unsqueeze_199 = None
        add_769: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_737, 3)
        clamp_min_8: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_769, 0);  add_769 = None
        clamp_max_8: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
        mul_1146: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_737, clamp_max_8);  add_737 = clamp_max_8 = None
        div_8: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_1146, 6);  mul_1146 = None
        convert_element_type_92: "bf16[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        convolution_31: "bf16[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_92, arg139_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184);  convert_element_type_92 = arg139_1 = None
        convert_element_type_93: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float32);  arg140_1 = None
        unsqueeze_200: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_93, -1);  convert_element_type_93 = None
        unsqueeze_201: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        sub_155: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_201);  convolution_31 = unsqueeze_201 = None
        convert_element_type_94: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg141_1, torch.float32);  arg141_1 = None
        add_780: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_94, 0.001);  convert_element_type_94 = None
        sqrt_25: "f32[184][1]cuda:0" = torch.ops.aten.sqrt.default(add_780);  add_780 = None
        reciprocal_25: "f32[184][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_1154: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        unsqueeze_202: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1154, -1);  mul_1154 = None
        unsqueeze_203: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        mul_1155: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_203);  sub_155 = unsqueeze_203 = None
        unsqueeze_204: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_205: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_1156: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1155, unsqueeze_205);  mul_1155 = unsqueeze_205 = None
        unsqueeze_206: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_207: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_781: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1156, unsqueeze_207);  mul_1156 = unsqueeze_207 = None
        add_813: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_781, 3)
        clamp_min_9: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_813, 0);  add_813 = None
        clamp_max_9: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
        mul_1287: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_781, clamp_max_9);  add_781 = clamp_max_9 = None
        div_9: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_1287, 6);  mul_1287 = None
        convert_element_type_97: "bf16[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        convolution_32: "bf16[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_97, arg144_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_97 = arg144_1 = None
        convert_element_type_98: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg145_1, torch.float32);  arg145_1 = None
        unsqueeze_208: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
        unsqueeze_209: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        sub_162: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_209);  convolution_32 = unsqueeze_209 = None
        convert_element_type_99: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg146_1, torch.float32);  arg146_1 = None
        add_824: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_99, 0.001);  convert_element_type_99 = None
        sqrt_26: "f32[80][1]cuda:0" = torch.ops.aten.sqrt.default(add_824);  add_824 = None
        reciprocal_26: "f32[80][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_1295: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        unsqueeze_210: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1295, -1);  mul_1295 = None
        unsqueeze_211: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        mul_1296: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_211);  sub_162 = unsqueeze_211 = None
        unsqueeze_212: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_213: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_1297: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1296, unsqueeze_213);  mul_1296 = unsqueeze_213 = None
        unsqueeze_214: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_215: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_825: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1297, unsqueeze_215);  mul_1297 = unsqueeze_215 = None
        convert_element_type_100: "bf16[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_825, torch.bfloat16);  add_825 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_841: "bf16[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_100, add_725);  convert_element_type_100 = add_725 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_33: "bf16[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(add_841, arg149_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg149_1 = None
        convert_element_type_101: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg150_1, torch.float32);  arg150_1 = None
        unsqueeze_216: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_101, -1);  convert_element_type_101 = None
        unsqueeze_217: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        sub_168: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_217);  convolution_33 = unsqueeze_217 = None
        convert_element_type_102: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg151_1, torch.float32);  arg151_1 = None
        add_852: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_102, 0.001);  convert_element_type_102 = None
        sqrt_27: "f32[184][1]cuda:0" = torch.ops.aten.sqrt.default(add_852);  add_852 = None
        reciprocal_27: "f32[184][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_1311: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_218: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1311, -1);  mul_1311 = None
        unsqueeze_219: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        mul_1312: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_219);  sub_168 = unsqueeze_219 = None
        unsqueeze_220: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_221: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_1313: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1312, unsqueeze_221);  mul_1312 = unsqueeze_221 = None
        unsqueeze_222: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_223: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_853: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1313, unsqueeze_223);  mul_1313 = unsqueeze_223 = None
        add_885: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_853, 3)
        clamp_min_10: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_885, 0);  add_885 = None
        clamp_max_10: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
        mul_1444: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_853, clamp_max_10);  add_853 = clamp_max_10 = None
        div_10: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_1444, 6);  mul_1444 = None
        convert_element_type_105: "bf16[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        convolution_34: "bf16[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_105, arg154_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184);  convert_element_type_105 = arg154_1 = None
        convert_element_type_106: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg155_1, torch.float32);  arg155_1 = None
        unsqueeze_224: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_106, -1);  convert_element_type_106 = None
        unsqueeze_225: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        sub_175: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_225);  convolution_34 = unsqueeze_225 = None
        convert_element_type_107: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg156_1, torch.float32);  arg156_1 = None
        add_896: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_107, 0.001);  convert_element_type_107 = None
        sqrt_28: "f32[184][1]cuda:0" = torch.ops.aten.sqrt.default(add_896);  add_896 = None
        reciprocal_28: "f32[184][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_1452: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_226: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1452, -1);  mul_1452 = None
        unsqueeze_227: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        mul_1453: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_227);  sub_175 = unsqueeze_227 = None
        unsqueeze_228: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_229: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_1454: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1453, unsqueeze_229);  mul_1453 = unsqueeze_229 = None
        unsqueeze_230: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_231: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_897: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1454, unsqueeze_231);  mul_1454 = unsqueeze_231 = None
        add_929: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_897, 3)
        clamp_min_11: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_929, 0);  add_929 = None
        clamp_max_11: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
        mul_1585: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_897, clamp_max_11);  add_897 = clamp_max_11 = None
        div_11: "f32[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_1585, 6);  mul_1585 = None
        convert_element_type_110: "bf16[s21, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        convolution_35: "bf16[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_110, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_110 = arg159_1 = None
        convert_element_type_111: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg160_1, torch.float32);  arg160_1 = None
        unsqueeze_232: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_111, -1);  convert_element_type_111 = None
        unsqueeze_233: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        sub_182: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_233);  convolution_35 = unsqueeze_233 = None
        convert_element_type_112: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg161_1, torch.float32);  arg161_1 = None
        add_940: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_112, 0.001);  convert_element_type_112 = None
        sqrt_29: "f32[80][1]cuda:0" = torch.ops.aten.sqrt.default(add_940);  add_940 = None
        reciprocal_29: "f32[80][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_1593: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_234: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1593, -1);  mul_1593 = None
        unsqueeze_235: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        mul_1594: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_235);  sub_182 = unsqueeze_235 = None
        unsqueeze_236: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_237: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_1595: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1594, unsqueeze_237);  mul_1594 = unsqueeze_237 = None
        unsqueeze_238: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_239: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_941: "f32[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1595, unsqueeze_239);  mul_1595 = unsqueeze_239 = None
        convert_element_type_113: "bf16[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_941, torch.bfloat16);  add_941 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_957: "bf16[s21, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_113, add_841);  convert_element_type_113 = add_841 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_36: "bf16[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(add_957, arg164_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_957 = arg164_1 = None
        convert_element_type_114: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg165_1, torch.float32);  arg165_1 = None
        unsqueeze_240: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_114, -1);  convert_element_type_114 = None
        unsqueeze_241: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        sub_188: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_241);  convolution_36 = unsqueeze_241 = None
        convert_element_type_115: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg166_1, torch.float32);  arg166_1 = None
        add_968: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_115, 0.001);  convert_element_type_115 = None
        sqrt_30: "f32[480][1]cuda:0" = torch.ops.aten.sqrt.default(add_968);  add_968 = None
        reciprocal_30: "f32[480][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_1609: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_242: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1609, -1);  mul_1609 = None
        unsqueeze_243: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        mul_1610: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_243);  sub_188 = unsqueeze_243 = None
        unsqueeze_244: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_245: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_1611: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1610, unsqueeze_245);  mul_1610 = unsqueeze_245 = None
        unsqueeze_246: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_247: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_969: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1611, unsqueeze_247);  mul_1611 = unsqueeze_247 = None
        add_1001: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_969, 3)
        clamp_min_12: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1001, 0);  add_1001 = None
        clamp_max_12: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
        mul_1742: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_969, clamp_max_12);  add_969 = clamp_max_12 = None
        div_12: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_1742, 6);  mul_1742 = None
        convert_element_type_118: "bf16[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        convolution_37: "bf16[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_118, arg169_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  convert_element_type_118 = arg169_1 = None
        convert_element_type_119: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg170_1, torch.float32);  arg170_1 = None
        unsqueeze_248: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_119, -1);  convert_element_type_119 = None
        unsqueeze_249: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        sub_195: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_249);  convolution_37 = unsqueeze_249 = None
        convert_element_type_120: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg171_1, torch.float32);  arg171_1 = None
        add_1012: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_120, 0.001);  convert_element_type_120 = None
        sqrt_31: "f32[480][1]cuda:0" = torch.ops.aten.sqrt.default(add_1012);  add_1012 = None
        reciprocal_31: "f32[480][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_1750: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_250: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1750, -1);  mul_1750 = None
        unsqueeze_251: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        mul_1751: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_251);  sub_195 = unsqueeze_251 = None
        unsqueeze_252: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_253: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_1752: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1751, unsqueeze_253);  mul_1751 = unsqueeze_253 = None
        unsqueeze_254: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_255: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_1013: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1752, unsqueeze_255);  mul_1752 = unsqueeze_255 = None
        add_1045: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1013, 3)
        clamp_min_13: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1045, 0);  add_1045 = None
        clamp_max_13: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
        mul_1883: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_1013, clamp_max_13);  add_1013 = clamp_max_13 = None
        div_13: "f32[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_1883, 6);  mul_1883 = None
        convert_element_type_123: "bf16[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_3: "bf16[s21, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_123, [-1, -2], True)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_38: "bf16[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_3, arg174_1, arg175_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg174_1 = arg175_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_14: "bf16[s21, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_38);  convolution_38 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_39: "bf16[s21, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_14, arg176_1, arg177_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = arg176_1 = arg177_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_124: "f32[s21, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        add_1071: "f32[s21, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_124, 3);  convert_element_type_124 = None
        clamp_min_14: "f32[s21, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1071, 0);  add_1071 = None
        clamp_max_14: "f32[s21, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
        div_14: "f32[s21, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_14, 6);  clamp_max_14 = None
        convert_element_type_125: "bf16[s21, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_1896: "bf16[s21, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_125, convert_element_type_123);  convert_element_type_125 = convert_element_type_123 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_40: "bf16[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(mul_1896, arg178_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_1896 = arg178_1 = None
        convert_element_type_126: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg179_1, torch.float32);  arg179_1 = None
        unsqueeze_256: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_126, -1);  convert_element_type_126 = None
        unsqueeze_257: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        sub_208: "f32[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_257);  convolution_40 = unsqueeze_257 = None
        convert_element_type_127: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg180_1, torch.float32);  arg180_1 = None
        add_1087: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_127, 0.001);  convert_element_type_127 = None
        sqrt_32: "f32[112][1]cuda:0" = torch.ops.aten.sqrt.default(add_1087);  add_1087 = None
        reciprocal_32: "f32[112][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_1904: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_258: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1904, -1);  mul_1904 = None
        unsqueeze_259: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        mul_1905: "f32[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_259);  sub_208 = unsqueeze_259 = None
        unsqueeze_260: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_261: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_1906: "f32[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1905, unsqueeze_261);  mul_1905 = unsqueeze_261 = None
        unsqueeze_262: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_263: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_1088: "f32[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1906, unsqueeze_263);  mul_1906 = unsqueeze_263 = None
        convert_element_type_128: "bf16[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1088, torch.bfloat16);  add_1088 = None
        convolution_41: "bf16[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_128, arg183_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg183_1 = None
        convert_element_type_129: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg184_1, torch.float32);  arg184_1 = None
        unsqueeze_264: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_129, -1);  convert_element_type_129 = None
        unsqueeze_265: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        sub_211: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_265);  convolution_41 = unsqueeze_265 = None
        convert_element_type_130: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg185_1, torch.float32);  arg185_1 = None
        add_1099: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_130, 0.001);  convert_element_type_130 = None
        sqrt_33: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_1099);  add_1099 = None
        reciprocal_33: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_1914: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_266: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1914, -1);  mul_1914 = None
        unsqueeze_267: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        mul_1915: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_267);  sub_211 = unsqueeze_267 = None
        unsqueeze_268: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_269: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_1916: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1915, unsqueeze_269);  mul_1915 = unsqueeze_269 = None
        unsqueeze_270: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_271: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_1100: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1916, unsqueeze_271);  mul_1916 = unsqueeze_271 = None
        add_1132: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1100, 3)
        clamp_min_15: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1132, 0);  add_1132 = None
        clamp_max_15: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
        mul_2047: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_1100, clamp_max_15);  add_1100 = clamp_max_15 = None
        div_15: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_2047, 6);  mul_2047 = None
        convert_element_type_133: "bf16[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        convolution_42: "bf16[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_133, arg188_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 672);  convert_element_type_133 = arg188_1 = None
        convert_element_type_134: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg189_1, torch.float32);  arg189_1 = None
        unsqueeze_272: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_134, -1);  convert_element_type_134 = None
        unsqueeze_273: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        sub_218: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_273);  convolution_42 = unsqueeze_273 = None
        convert_element_type_135: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg190_1, torch.float32);  arg190_1 = None
        add_1143: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_135, 0.001);  convert_element_type_135 = None
        sqrt_34: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_1143);  add_1143 = None
        reciprocal_34: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_2055: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_274: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2055, -1);  mul_2055 = None
        unsqueeze_275: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        mul_2056: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_275);  sub_218 = unsqueeze_275 = None
        unsqueeze_276: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
        unsqueeze_277: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_2057: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2056, unsqueeze_277);  mul_2056 = unsqueeze_277 = None
        unsqueeze_278: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_279: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_1144: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2057, unsqueeze_279);  mul_2057 = unsqueeze_279 = None
        add_1176: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1144, 3)
        clamp_min_16: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1176, 0);  add_1176 = None
        clamp_max_16: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
        mul_2188: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_1144, clamp_max_16);  add_1144 = clamp_max_16 = None
        div_16: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_2188, 6);  mul_2188 = None
        convert_element_type_138: "bf16[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_4: "bf16[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_138, [-1, -2], True)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_43: "bf16[s21, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_4, arg193_1, arg194_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_4 = arg193_1 = arg194_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_15: "bf16[s21, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_43);  convolution_43 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_44: "bf16[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_15, arg195_1, arg196_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_15 = arg195_1 = arg196_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_139: "f32[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        add_1202: "f32[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_139, 3);  convert_element_type_139 = None
        clamp_min_17: "f32[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1202, 0);  add_1202 = None
        clamp_max_17: "f32[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
        div_17: "f32[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_17, 6);  clamp_max_17 = None
        convert_element_type_140: "bf16[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_2201: "bf16[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, convert_element_type_138);  convert_element_type_140 = convert_element_type_138 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_45: "bf16[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(mul_2201, arg197_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_2201 = arg197_1 = None
        convert_element_type_141: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg198_1, torch.float32);  arg198_1 = None
        unsqueeze_280: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_141, -1);  convert_element_type_141 = None
        unsqueeze_281: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        sub_231: "f32[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_281);  convolution_45 = unsqueeze_281 = None
        convert_element_type_142: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg199_1, torch.float32);  arg199_1 = None
        add_1218: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_142, 0.001);  convert_element_type_142 = None
        sqrt_35: "f32[112][1]cuda:0" = torch.ops.aten.sqrt.default(add_1218);  add_1218 = None
        reciprocal_35: "f32[112][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_2209: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_282: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2209, -1);  mul_2209 = None
        unsqueeze_283: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        mul_2210: "f32[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_283);  sub_231 = unsqueeze_283 = None
        unsqueeze_284: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_285: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_2211: "f32[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2210, unsqueeze_285);  mul_2210 = unsqueeze_285 = None
        unsqueeze_286: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg201_1, -1);  arg201_1 = None
        unsqueeze_287: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_1219: "f32[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2211, unsqueeze_287);  mul_2211 = unsqueeze_287 = None
        convert_element_type_143: "bf16[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1219, torch.bfloat16);  add_1219 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_1235: "bf16[s21, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_143, convert_element_type_128);  convert_element_type_143 = convert_element_type_128 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_46: "bf16[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(add_1235, arg202_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_1235 = arg202_1 = None
        convert_element_type_144: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg203_1, torch.float32);  arg203_1 = None
        unsqueeze_288: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_144, -1);  convert_element_type_144 = None
        unsqueeze_289: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        sub_237: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_289);  convolution_46 = unsqueeze_289 = None
        convert_element_type_145: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg204_1, torch.float32);  arg204_1 = None
        add_1246: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_145, 0.001);  convert_element_type_145 = None
        sqrt_36: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_1246);  add_1246 = None
        reciprocal_36: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_2225: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_290: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2225, -1);  mul_2225 = None
        unsqueeze_291: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        mul_2226: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_291);  sub_237 = unsqueeze_291 = None
        unsqueeze_292: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_293: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_2227: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2226, unsqueeze_293);  mul_2226 = unsqueeze_293 = None
        unsqueeze_294: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_295: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_1247: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2227, unsqueeze_295);  mul_2227 = unsqueeze_295 = None
        add_1279: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1247, 3)
        clamp_min_18: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1279, 0);  add_1279 = None
        clamp_max_18: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
        mul_2358: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_1247, clamp_max_18);  add_1247 = clamp_max_18 = None
        div_18: "f32[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_2358, 6);  mul_2358 = None
        convert_element_type_148: "bf16[s21, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        convolution_47: "bf16[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_148, arg207_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672);  convert_element_type_148 = arg207_1 = None
        convert_element_type_149: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg208_1, torch.float32);  arg208_1 = None
        unsqueeze_296: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_149, -1);  convert_element_type_149 = None
        unsqueeze_297: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        sub_244: "f32[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_297);  convolution_47 = unsqueeze_297 = None
        convert_element_type_150: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg209_1, torch.float32);  arg209_1 = None
        add_1290: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_150, 0.001);  convert_element_type_150 = None
        sqrt_37: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_1290);  add_1290 = None
        reciprocal_37: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_2366: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_298: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2366, -1);  mul_2366 = None
        unsqueeze_299: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        mul_2367: "f32[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_299);  sub_244 = unsqueeze_299 = None
        unsqueeze_300: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_301: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_2368: "f32[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2367, unsqueeze_301);  mul_2367 = unsqueeze_301 = None
        unsqueeze_302: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_303: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_1291: "f32[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2368, unsqueeze_303);  mul_2368 = unsqueeze_303 = None
        add_1323: "f32[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1291, 3)
        clamp_min_19: "f32[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1323, 0);  add_1323 = None
        clamp_max_19: "f32[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
        mul_2499: "f32[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_1291, clamp_max_19);  add_1291 = clamp_max_19 = None
        div_19: "f32[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_2499, 6);  mul_2499 = None
        convert_element_type_153: "bf16[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_5: "bf16[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_153, [-1, -2], True)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_48: "bf16[s21, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_5, arg212_1, arg213_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg212_1 = arg213_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_16: "bf16[s21, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_48);  convolution_48 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_49: "bf16[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_16, arg214_1, arg215_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_16 = arg214_1 = arg215_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_154: "f32[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        add_1349: "f32[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_154, 3);  convert_element_type_154 = None
        clamp_min_20: "f32[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1349, 0);  add_1349 = None
        clamp_max_20: "f32[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
        div_20: "f32[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_20, 6);  clamp_max_20 = None
        convert_element_type_155: "bf16[s21, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_2512: "bf16[s21, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_155, convert_element_type_153);  convert_element_type_155 = convert_element_type_153 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_50: "bf16[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(mul_2512, arg216_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_2512 = arg216_1 = None
        convert_element_type_156: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg217_1, torch.float32);  arg217_1 = None
        unsqueeze_304: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_156, -1);  convert_element_type_156 = None
        unsqueeze_305: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        sub_257: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_305);  convolution_50 = unsqueeze_305 = None
        convert_element_type_157: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg218_1, torch.float32);  arg218_1 = None
        add_1365: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_157, 0.001);  convert_element_type_157 = None
        sqrt_38: "f32[160][1]cuda:0" = torch.ops.aten.sqrt.default(add_1365);  add_1365 = None
        reciprocal_38: "f32[160][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_2520: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_306: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2520, -1);  mul_2520 = None
        unsqueeze_307: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        mul_2521: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_307);  sub_257 = unsqueeze_307 = None
        unsqueeze_308: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_309: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_2522: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2521, unsqueeze_309);  mul_2521 = unsqueeze_309 = None
        unsqueeze_310: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_311: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_1366: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2522, unsqueeze_311);  mul_2522 = unsqueeze_311 = None
        convert_element_type_158: "bf16[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1366, torch.bfloat16);  add_1366 = None
        convolution_51: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_158, arg221_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg221_1 = None
        convert_element_type_159: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg222_1, torch.float32);  arg222_1 = None
        unsqueeze_312: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_159, -1);  convert_element_type_159 = None
        unsqueeze_313: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        sub_260: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_313);  convolution_51 = unsqueeze_313 = None
        convert_element_type_160: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg223_1, torch.float32);  arg223_1 = None
        add_1377: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_160, 0.001);  convert_element_type_160 = None
        sqrt_39: "f32[960][1]cuda:0" = torch.ops.aten.sqrt.default(add_1377);  add_1377 = None
        reciprocal_39: "f32[960][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_2530: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_314: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2530, -1);  mul_2530 = None
        unsqueeze_315: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        mul_2531: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_315);  sub_260 = unsqueeze_315 = None
        unsqueeze_316: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_317: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_2532: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2531, unsqueeze_317);  mul_2531 = unsqueeze_317 = None
        unsqueeze_318: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_319: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_1378: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2532, unsqueeze_319);  mul_2532 = unsqueeze_319 = None
        add_1410: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1378, 3)
        clamp_min_21: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1410, 0);  add_1410 = None
        clamp_max_21: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
        mul_2663: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_1378, clamp_max_21);  add_1378 = clamp_max_21 = None
        div_21: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_2663, 6);  mul_2663 = None
        convert_element_type_163: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        convolution_52: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_163, arg226_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960);  convert_element_type_163 = arg226_1 = None
        convert_element_type_164: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg227_1, torch.float32);  arg227_1 = None
        unsqueeze_320: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_164, -1);  convert_element_type_164 = None
        unsqueeze_321: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        sub_267: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_321);  convolution_52 = unsqueeze_321 = None
        convert_element_type_165: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg228_1, torch.float32);  arg228_1 = None
        add_1421: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_165, 0.001);  convert_element_type_165 = None
        sqrt_40: "f32[960][1]cuda:0" = torch.ops.aten.sqrt.default(add_1421);  add_1421 = None
        reciprocal_40: "f32[960][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_2671: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_322: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2671, -1);  mul_2671 = None
        unsqueeze_323: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        mul_2672: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_323);  sub_267 = unsqueeze_323 = None
        unsqueeze_324: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_325: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_2673: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2672, unsqueeze_325);  mul_2672 = unsqueeze_325 = None
        unsqueeze_326: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_327: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_1422: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2673, unsqueeze_327);  mul_2673 = unsqueeze_327 = None
        add_1454: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1422, 3)
        clamp_min_22: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1454, 0);  add_1454 = None
        clamp_max_22: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
        mul_2804: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_1422, clamp_max_22);  add_1422 = clamp_max_22 = None
        div_22: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_2804, 6);  mul_2804 = None
        convert_element_type_168: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_6: "bf16[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_168, [-1, -2], True)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_53: "bf16[s21, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_6, arg231_1, arg232_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_6 = arg231_1 = arg232_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_17: "bf16[s21, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_54: "bf16[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_17, arg233_1, arg234_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg233_1 = arg234_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_169: "f32[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_54, torch.float32);  convolution_54 = None
        add_1480: "f32[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_169, 3);  convert_element_type_169 = None
        clamp_min_23: "f32[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1480, 0);  add_1480 = None
        clamp_max_23: "f32[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
        div_23: "f32[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_23, 6);  clamp_max_23 = None
        convert_element_type_170: "bf16[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_2817: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_170, convert_element_type_168);  convert_element_type_170 = convert_element_type_168 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_55: "bf16[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(mul_2817, arg235_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_2817 = arg235_1 = None
        convert_element_type_171: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg236_1, torch.float32);  arg236_1 = None
        unsqueeze_328: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_171, -1);  convert_element_type_171 = None
        unsqueeze_329: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        sub_280: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_329);  convolution_55 = unsqueeze_329 = None
        convert_element_type_172: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg237_1, torch.float32);  arg237_1 = None
        add_1496: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_172, 0.001);  convert_element_type_172 = None
        sqrt_41: "f32[160][1]cuda:0" = torch.ops.aten.sqrt.default(add_1496);  add_1496 = None
        reciprocal_41: "f32[160][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_2825: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_330: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2825, -1);  mul_2825 = None
        unsqueeze_331: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        mul_2826: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_331);  sub_280 = unsqueeze_331 = None
        unsqueeze_332: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_333: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_2827: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2826, unsqueeze_333);  mul_2826 = unsqueeze_333 = None
        unsqueeze_334: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_335: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_1497: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2827, unsqueeze_335);  mul_2827 = unsqueeze_335 = None
        convert_element_type_173: "bf16[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1497, torch.bfloat16);  add_1497 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_1513: "bf16[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_173, convert_element_type_158);  convert_element_type_173 = convert_element_type_158 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_56: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(add_1513, arg240_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg240_1 = None
        convert_element_type_174: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg241_1, torch.float32);  arg241_1 = None
        unsqueeze_336: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_174, -1);  convert_element_type_174 = None
        unsqueeze_337: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        sub_286: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_337);  convolution_56 = unsqueeze_337 = None
        convert_element_type_175: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg242_1, torch.float32);  arg242_1 = None
        add_1524: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_175, 0.001);  convert_element_type_175 = None
        sqrt_42: "f32[960][1]cuda:0" = torch.ops.aten.sqrt.default(add_1524);  add_1524 = None
        reciprocal_42: "f32[960][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_2841: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_338: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2841, -1);  mul_2841 = None
        unsqueeze_339: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        mul_2842: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_339);  sub_286 = unsqueeze_339 = None
        unsqueeze_340: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
        unsqueeze_341: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_2843: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2842, unsqueeze_341);  mul_2842 = unsqueeze_341 = None
        unsqueeze_342: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_343: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_1525: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2843, unsqueeze_343);  mul_2843 = unsqueeze_343 = None
        add_1557: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1525, 3)
        clamp_min_24: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1557, 0);  add_1557 = None
        clamp_max_24: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
        mul_2974: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_1525, clamp_max_24);  add_1525 = clamp_max_24 = None
        div_24: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_2974, 6);  mul_2974 = None
        convert_element_type_178: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None
        convolution_57: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_178, arg245_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960);  convert_element_type_178 = arg245_1 = None
        convert_element_type_179: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg246_1, torch.float32);  arg246_1 = None
        unsqueeze_344: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_179, -1);  convert_element_type_179 = None
        unsqueeze_345: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        sub_293: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_345);  convolution_57 = unsqueeze_345 = None
        convert_element_type_180: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg247_1, torch.float32);  arg247_1 = None
        add_1568: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_180, 0.001);  convert_element_type_180 = None
        sqrt_43: "f32[960][1]cuda:0" = torch.ops.aten.sqrt.default(add_1568);  add_1568 = None
        reciprocal_43: "f32[960][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_2982: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_346: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2982, -1);  mul_2982 = None
        unsqueeze_347: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        mul_2983: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_347);  sub_293 = unsqueeze_347 = None
        unsqueeze_348: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_349: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_2984: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2983, unsqueeze_349);  mul_2983 = unsqueeze_349 = None
        unsqueeze_350: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_351: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_1569: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2984, unsqueeze_351);  mul_2984 = unsqueeze_351 = None
        add_1601: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1569, 3)
        clamp_min_25: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1601, 0);  add_1601 = None
        clamp_max_25: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
        mul_3115: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_1569, clamp_max_25);  add_1569 = clamp_max_25 = None
        div_25: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_3115, 6);  mul_3115 = None
        convert_element_type_183: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_7: "bf16[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_183, [-1, -2], True)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_58: "bf16[s21, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_7, arg250_1, arg251_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg250_1 = arg251_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_18: "bf16[s21, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_58);  convolution_58 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_59: "bf16[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_18, arg252_1, arg253_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_18 = arg252_1 = arg253_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_184: "f32[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32);  convolution_59 = None
        add_1627: "f32[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_184, 3);  convert_element_type_184 = None
        clamp_min_26: "f32[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1627, 0);  add_1627 = None
        clamp_max_26: "f32[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
        div_26: "f32[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_26, 6);  clamp_max_26 = None
        convert_element_type_185: "bf16[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_3128: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_185, convert_element_type_183);  convert_element_type_185 = convert_element_type_183 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_60: "bf16[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(mul_3128, arg254_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_3128 = arg254_1 = None
        convert_element_type_186: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg255_1, torch.float32);  arg255_1 = None
        unsqueeze_352: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_186, -1);  convert_element_type_186 = None
        unsqueeze_353: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        sub_306: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_353);  convolution_60 = unsqueeze_353 = None
        convert_element_type_187: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg256_1, torch.float32);  arg256_1 = None
        add_1643: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_187, 0.001);  convert_element_type_187 = None
        sqrt_44: "f32[160][1]cuda:0" = torch.ops.aten.sqrt.default(add_1643);  add_1643 = None
        reciprocal_44: "f32[160][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_3136: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_354: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3136, -1);  mul_3136 = None
        unsqueeze_355: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        mul_3137: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_355);  sub_306 = unsqueeze_355 = None
        unsqueeze_356: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_357: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_3138: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3137, unsqueeze_357);  mul_3137 = unsqueeze_357 = None
        unsqueeze_358: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
        unsqueeze_359: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_1644: "f32[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3138, unsqueeze_359);  mul_3138 = unsqueeze_359 = None
        convert_element_type_188: "bf16[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1644, torch.bfloat16);  add_1644 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_1660: "bf16[s21, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_188, add_1513);  convert_element_type_188 = add_1513 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:211 in _forward_impl, code: x = self.features(x)
        convolution_61: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(add_1660, arg259_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_1660 = arg259_1 = None
        convert_element_type_189: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg260_1, torch.float32);  arg260_1 = None
        unsqueeze_360: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_189, -1);  convert_element_type_189 = None
        unsqueeze_361: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        sub_312: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_361);  convolution_61 = unsqueeze_361 = None
        convert_element_type_190: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg261_1, torch.float32);  arg261_1 = None
        add_1671: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_190, 0.001);  convert_element_type_190 = None
        sqrt_45: "f32[960][1]cuda:0" = torch.ops.aten.sqrt.default(add_1671);  add_1671 = None
        reciprocal_45: "f32[960][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_3152: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_362: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3152, -1);  mul_3152 = None
        unsqueeze_363: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        mul_3153: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_363);  sub_312 = unsqueeze_363 = None
        unsqueeze_364: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_365: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_3154: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3153, unsqueeze_365);  mul_3153 = unsqueeze_365 = None
        unsqueeze_366: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_367: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_1672: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3154, unsqueeze_367);  mul_3154 = unsqueeze_367 = None
        add_1704: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1672, 3)
        clamp_min_27: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1704, 0);  add_1704 = None
        clamp_max_27: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
        mul_3285: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_1672, clamp_max_27);  add_1672 = clamp_max_27 = None
        div_27: "f32[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_3285, 6);  mul_3285 = None
        convert_element_type_193: "bf16[s21, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:213 in _forward_impl, code: x = self.avgpool(x)
        mean_8: "bf16[s21, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_193, [-1, -2], True);  convert_element_type_193 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:214 in _forward_impl, code: x = torch.flatten(x, 1)
        view: "bf16[s21, 960][960, 1]cuda:0" = torch.ops.aten.reshape.default(mean_8, [arg1_1, 960]);  mean_8 = arg1_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/mobilenetv3.py:216 in _forward_impl, code: x = self.classifier(x)
        permute: "bf16[960, 1280][1, 960]cuda:0" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm: "bf16[s21, 1280][1280, 1]cuda:0" = torch.ops.aten.addmm.default(arg265_1, view, permute);  arg265_1 = view = permute = None
        convert_element_type_197: "f32[s21, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        add_1741: "f32[s21, 1280][1280, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_197, 3)
        clamp_min_28: "f32[s21, 1280][1280, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_1741, 0);  add_1741 = None
        clamp_max_28: "f32[s21, 1280][1280, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
        mul_3375: "f32[s21, 1280][1280, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_197, clamp_max_28);  convert_element_type_197 = clamp_max_28 = None
        div_28: "f32[s21, 1280][1280, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_3375, 6);  mul_3375 = None
        convert_element_type_198: "bf16[s21, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None
        permute_1: "bf16[1280, 1000][1, 1280]cuda:0" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_1: "bf16[s21, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg267_1, convert_element_type_198, permute_1);  arg267_1 = convert_element_type_198 = permute_1 = None
        return (addmm_1,)
