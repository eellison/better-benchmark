class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[16, 3, 3, 3][27, 9, 3, 1]cuda:0", arg1_1: "bf16[32, 3, 224, 224][150528, 50176, 224, 1]cuda:0", arg2_1: "bf16[16][1]cuda:0", arg3_1: "bf16[16][1]cuda:0", arg4_1: "bf16[16][1]cuda:0", arg5_1: "bf16[16][1]cuda:0", arg6_1: "bf16[16, 1, 3, 3][9, 9, 3, 1]cuda:0", arg7_1: "bf16[16][1]cuda:0", arg8_1: "bf16[16][1]cuda:0", arg9_1: "bf16[16][1]cuda:0", arg10_1: "bf16[16][1]cuda:0", arg11_1: "bf16[16, 16, 1, 1][16, 1, 1, 1]cuda:0", arg12_1: "bf16[16][1]cuda:0", arg13_1: "bf16[16][1]cuda:0", arg14_1: "bf16[16][1]cuda:0", arg15_1: "bf16[16][1]cuda:0", arg16_1: "bf16[64, 16, 1, 1][16, 1, 1, 1]cuda:0", arg17_1: "bf16[64][1]cuda:0", arg18_1: "bf16[64][1]cuda:0", arg19_1: "bf16[64][1]cuda:0", arg20_1: "bf16[64][1]cuda:0", arg21_1: "bf16[64, 1, 3, 3][9, 9, 3, 1]cuda:0", arg22_1: "bf16[64][1]cuda:0", arg23_1: "bf16[64][1]cuda:0", arg24_1: "bf16[64][1]cuda:0", arg25_1: "bf16[64][1]cuda:0", arg26_1: "bf16[24, 64, 1, 1][64, 1, 1, 1]cuda:0", arg27_1: "bf16[24][1]cuda:0", arg28_1: "bf16[24][1]cuda:0", arg29_1: "bf16[24][1]cuda:0", arg30_1: "bf16[24][1]cuda:0", arg31_1: "bf16[72, 24, 1, 1][24, 1, 1, 1]cuda:0", arg32_1: "bf16[72][1]cuda:0", arg33_1: "bf16[72][1]cuda:0", arg34_1: "bf16[72][1]cuda:0", arg35_1: "bf16[72][1]cuda:0", arg36_1: "bf16[72, 1, 3, 3][9, 9, 3, 1]cuda:0", arg37_1: "bf16[72][1]cuda:0", arg38_1: "bf16[72][1]cuda:0", arg39_1: "bf16[72][1]cuda:0", arg40_1: "bf16[72][1]cuda:0", arg41_1: "bf16[24, 72, 1, 1][72, 1, 1, 1]cuda:0", arg42_1: "bf16[24][1]cuda:0", arg43_1: "bf16[24][1]cuda:0", arg44_1: "bf16[24][1]cuda:0", arg45_1: "bf16[24][1]cuda:0", arg46_1: "bf16[72, 24, 1, 1][24, 1, 1, 1]cuda:0", arg47_1: "bf16[72][1]cuda:0", arg48_1: "bf16[72][1]cuda:0", arg49_1: "bf16[72][1]cuda:0", arg50_1: "bf16[72][1]cuda:0", arg51_1: "bf16[72, 1, 5, 5][25, 25, 5, 1]cuda:0", arg52_1: "bf16[72][1]cuda:0", arg53_1: "bf16[72][1]cuda:0", arg54_1: "bf16[72][1]cuda:0", arg55_1: "bf16[72][1]cuda:0", arg56_1: "bf16[24, 72, 1, 1][72, 1, 1, 1]cuda:0", arg57_1: "bf16[24][1]cuda:0", arg58_1: "bf16[72, 24, 1, 1][24, 1, 1, 1]cuda:0", arg59_1: "bf16[72][1]cuda:0", arg60_1: "bf16[40, 72, 1, 1][72, 1, 1, 1]cuda:0", arg61_1: "bf16[40][1]cuda:0", arg62_1: "bf16[40][1]cuda:0", arg63_1: "bf16[40][1]cuda:0", arg64_1: "bf16[40][1]cuda:0", arg65_1: "bf16[120, 40, 1, 1][40, 1, 1, 1]cuda:0", arg66_1: "bf16[120][1]cuda:0", arg67_1: "bf16[120][1]cuda:0", arg68_1: "bf16[120][1]cuda:0", arg69_1: "bf16[120][1]cuda:0", arg70_1: "bf16[120, 1, 5, 5][25, 25, 5, 1]cuda:0", arg71_1: "bf16[120][1]cuda:0", arg72_1: "bf16[120][1]cuda:0", arg73_1: "bf16[120][1]cuda:0", arg74_1: "bf16[120][1]cuda:0", arg75_1: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0", arg76_1: "bf16[32][1]cuda:0", arg77_1: "bf16[120, 32, 1, 1][32, 1, 1, 1]cuda:0", arg78_1: "bf16[120][1]cuda:0", arg79_1: "bf16[40, 120, 1, 1][120, 1, 1, 1]cuda:0", arg80_1: "bf16[40][1]cuda:0", arg81_1: "bf16[40][1]cuda:0", arg82_1: "bf16[40][1]cuda:0", arg83_1: "bf16[40][1]cuda:0", arg84_1: "bf16[120, 40, 1, 1][40, 1, 1, 1]cuda:0", arg85_1: "bf16[120][1]cuda:0", arg86_1: "bf16[120][1]cuda:0", arg87_1: "bf16[120][1]cuda:0", arg88_1: "bf16[120][1]cuda:0", arg89_1: "bf16[120, 1, 5, 5][25, 25, 5, 1]cuda:0", arg90_1: "bf16[120][1]cuda:0", arg91_1: "bf16[120][1]cuda:0", arg92_1: "bf16[120][1]cuda:0", arg93_1: "bf16[120][1]cuda:0", arg94_1: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0", arg95_1: "bf16[32][1]cuda:0", arg96_1: "bf16[120, 32, 1, 1][32, 1, 1, 1]cuda:0", arg97_1: "bf16[120][1]cuda:0", arg98_1: "bf16[40, 120, 1, 1][120, 1, 1, 1]cuda:0", arg99_1: "bf16[40][1]cuda:0", arg100_1: "bf16[40][1]cuda:0", arg101_1: "bf16[40][1]cuda:0", arg102_1: "bf16[40][1]cuda:0", arg103_1: "bf16[240, 40, 1, 1][40, 1, 1, 1]cuda:0", arg104_1: "bf16[240][1]cuda:0", arg105_1: "bf16[240][1]cuda:0", arg106_1: "bf16[240][1]cuda:0", arg107_1: "bf16[240][1]cuda:0", arg108_1: "bf16[240, 1, 3, 3][9, 9, 3, 1]cuda:0", arg109_1: "bf16[240][1]cuda:0", arg110_1: "bf16[240][1]cuda:0", arg111_1: "bf16[240][1]cuda:0", arg112_1: "bf16[240][1]cuda:0", arg113_1: "bf16[80, 240, 1, 1][240, 1, 1, 1]cuda:0", arg114_1: "bf16[80][1]cuda:0", arg115_1: "bf16[80][1]cuda:0", arg116_1: "bf16[80][1]cuda:0", arg117_1: "bf16[80][1]cuda:0", arg118_1: "bf16[200, 80, 1, 1][80, 1, 1, 1]cuda:0", arg119_1: "bf16[200][1]cuda:0", arg120_1: "bf16[200][1]cuda:0", arg121_1: "bf16[200][1]cuda:0", arg122_1: "bf16[200][1]cuda:0", arg123_1: "bf16[200, 1, 3, 3][9, 9, 3, 1]cuda:0", arg124_1: "bf16[200][1]cuda:0", arg125_1: "bf16[200][1]cuda:0", arg126_1: "bf16[200][1]cuda:0", arg127_1: "bf16[200][1]cuda:0", arg128_1: "bf16[80, 200, 1, 1][200, 1, 1, 1]cuda:0", arg129_1: "bf16[80][1]cuda:0", arg130_1: "bf16[80][1]cuda:0", arg131_1: "bf16[80][1]cuda:0", arg132_1: "bf16[80][1]cuda:0", arg133_1: "bf16[184, 80, 1, 1][80, 1, 1, 1]cuda:0", arg134_1: "bf16[184][1]cuda:0", arg135_1: "bf16[184][1]cuda:0", arg136_1: "bf16[184][1]cuda:0", arg137_1: "bf16[184][1]cuda:0", arg138_1: "bf16[184, 1, 3, 3][9, 9, 3, 1]cuda:0", arg139_1: "bf16[184][1]cuda:0", arg140_1: "bf16[184][1]cuda:0", arg141_1: "bf16[184][1]cuda:0", arg142_1: "bf16[184][1]cuda:0", arg143_1: "bf16[80, 184, 1, 1][184, 1, 1, 1]cuda:0", arg144_1: "bf16[80][1]cuda:0", arg145_1: "bf16[80][1]cuda:0", arg146_1: "bf16[80][1]cuda:0", arg147_1: "bf16[80][1]cuda:0", arg148_1: "bf16[184, 80, 1, 1][80, 1, 1, 1]cuda:0", arg149_1: "bf16[184][1]cuda:0", arg150_1: "bf16[184][1]cuda:0", arg151_1: "bf16[184][1]cuda:0", arg152_1: "bf16[184][1]cuda:0", arg153_1: "bf16[184, 1, 3, 3][9, 9, 3, 1]cuda:0", arg154_1: "bf16[184][1]cuda:0", arg155_1: "bf16[184][1]cuda:0", arg156_1: "bf16[184][1]cuda:0", arg157_1: "bf16[184][1]cuda:0", arg158_1: "bf16[80, 184, 1, 1][184, 1, 1, 1]cuda:0", arg159_1: "bf16[80][1]cuda:0", arg160_1: "bf16[80][1]cuda:0", arg161_1: "bf16[80][1]cuda:0", arg162_1: "bf16[80][1]cuda:0", arg163_1: "bf16[480, 80, 1, 1][80, 1, 1, 1]cuda:0", arg164_1: "bf16[480][1]cuda:0", arg165_1: "bf16[480][1]cuda:0", arg166_1: "bf16[480][1]cuda:0", arg167_1: "bf16[480][1]cuda:0", arg168_1: "bf16[480, 1, 3, 3][9, 9, 3, 1]cuda:0", arg169_1: "bf16[480][1]cuda:0", arg170_1: "bf16[480][1]cuda:0", arg171_1: "bf16[480][1]cuda:0", arg172_1: "bf16[480][1]cuda:0", arg173_1: "bf16[120, 480, 1, 1][480, 1, 1, 1]cuda:0", arg174_1: "bf16[120][1]cuda:0", arg175_1: "bf16[480, 120, 1, 1][120, 1, 1, 1]cuda:0", arg176_1: "bf16[480][1]cuda:0", arg177_1: "bf16[112, 480, 1, 1][480, 1, 1, 1]cuda:0", arg178_1: "bf16[112][1]cuda:0", arg179_1: "bf16[112][1]cuda:0", arg180_1: "bf16[112][1]cuda:0", arg181_1: "bf16[112][1]cuda:0", arg182_1: "bf16[672, 112, 1, 1][112, 1, 1, 1]cuda:0", arg183_1: "bf16[672][1]cuda:0", arg184_1: "bf16[672][1]cuda:0", arg185_1: "bf16[672][1]cuda:0", arg186_1: "bf16[672][1]cuda:0", arg187_1: "bf16[672, 1, 3, 3][9, 9, 3, 1]cuda:0", arg188_1: "bf16[672][1]cuda:0", arg189_1: "bf16[672][1]cuda:0", arg190_1: "bf16[672][1]cuda:0", arg191_1: "bf16[672][1]cuda:0", arg192_1: "bf16[168, 672, 1, 1][672, 1, 1, 1]cuda:0", arg193_1: "bf16[168][1]cuda:0", arg194_1: "bf16[672, 168, 1, 1][168, 1, 1, 1]cuda:0", arg195_1: "bf16[672][1]cuda:0", arg196_1: "bf16[112, 672, 1, 1][672, 1, 1, 1]cuda:0", arg197_1: "bf16[112][1]cuda:0", arg198_1: "bf16[112][1]cuda:0", arg199_1: "bf16[112][1]cuda:0", arg200_1: "bf16[112][1]cuda:0", arg201_1: "bf16[672, 112, 1, 1][112, 1, 1, 1]cuda:0", arg202_1: "bf16[672][1]cuda:0", arg203_1: "bf16[672][1]cuda:0", arg204_1: "bf16[672][1]cuda:0", arg205_1: "bf16[672][1]cuda:0", arg206_1: "bf16[672, 1, 5, 5][25, 25, 5, 1]cuda:0", arg207_1: "bf16[672][1]cuda:0", arg208_1: "bf16[672][1]cuda:0", arg209_1: "bf16[672][1]cuda:0", arg210_1: "bf16[672][1]cuda:0", arg211_1: "bf16[168, 672, 1, 1][672, 1, 1, 1]cuda:0", arg212_1: "bf16[168][1]cuda:0", arg213_1: "bf16[672, 168, 1, 1][168, 1, 1, 1]cuda:0", arg214_1: "bf16[672][1]cuda:0", arg215_1: "bf16[160, 672, 1, 1][672, 1, 1, 1]cuda:0", arg216_1: "bf16[160][1]cuda:0", arg217_1: "bf16[160][1]cuda:0", arg218_1: "bf16[160][1]cuda:0", arg219_1: "bf16[160][1]cuda:0", arg220_1: "bf16[960, 160, 1, 1][160, 1, 1, 1]cuda:0", arg221_1: "bf16[960][1]cuda:0", arg222_1: "bf16[960][1]cuda:0", arg223_1: "bf16[960][1]cuda:0", arg224_1: "bf16[960][1]cuda:0", arg225_1: "bf16[960, 1, 5, 5][25, 25, 5, 1]cuda:0", arg226_1: "bf16[960][1]cuda:0", arg227_1: "bf16[960][1]cuda:0", arg228_1: "bf16[960][1]cuda:0", arg229_1: "bf16[960][1]cuda:0", arg230_1: "bf16[240, 960, 1, 1][960, 1, 1, 1]cuda:0", arg231_1: "bf16[240][1]cuda:0", arg232_1: "bf16[960, 240, 1, 1][240, 1, 1, 1]cuda:0", arg233_1: "bf16[960][1]cuda:0", arg234_1: "bf16[160, 960, 1, 1][960, 1, 1, 1]cuda:0", arg235_1: "bf16[160][1]cuda:0", arg236_1: "bf16[160][1]cuda:0", arg237_1: "bf16[160][1]cuda:0", arg238_1: "bf16[160][1]cuda:0", arg239_1: "bf16[960, 160, 1, 1][160, 1, 1, 1]cuda:0", arg240_1: "bf16[960][1]cuda:0", arg241_1: "bf16[960][1]cuda:0", arg242_1: "bf16[960][1]cuda:0", arg243_1: "bf16[960][1]cuda:0", arg244_1: "bf16[960, 1, 5, 5][25, 25, 5, 1]cuda:0", arg245_1: "bf16[960][1]cuda:0", arg246_1: "bf16[960][1]cuda:0", arg247_1: "bf16[960][1]cuda:0", arg248_1: "bf16[960][1]cuda:0", arg249_1: "bf16[240, 960, 1, 1][960, 1, 1, 1]cuda:0", arg250_1: "bf16[240][1]cuda:0", arg251_1: "bf16[960, 240, 1, 1][240, 1, 1, 1]cuda:0", arg252_1: "bf16[960][1]cuda:0", arg253_1: "bf16[160, 960, 1, 1][960, 1, 1, 1]cuda:0", arg254_1: "bf16[160][1]cuda:0", arg255_1: "bf16[160][1]cuda:0", arg256_1: "bf16[160][1]cuda:0", arg257_1: "bf16[160][1]cuda:0", arg258_1: "bf16[960, 160, 1, 1][160, 1, 1, 1]cuda:0", arg259_1: "bf16[960][1]cuda:0", arg260_1: "bf16[960][1]cuda:0", arg261_1: "bf16[960][1]cuda:0", arg262_1: "bf16[960][1]cuda:0", arg263_1: "bf16[1280, 960][960, 1]cuda:0", arg264_1: "bf16[1280][1]cuda:0", arg265_1: "bf16[1000, 1280][1280, 1]cuda:0", arg266_1: "bf16[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:211 in _forward_impl, code: x = self.features(x)
        convolution: "bf16[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        convert_element_type: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        unsqueeze: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        convert_element_type_1: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1, 0.001);  convert_element_type_1 = None
        sqrt: "f32[16][1]cuda:0" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[16][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "bf16[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_3: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        add_2: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_3, 3)
        clamp_min: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_2, 0);  add_2 = None
        clamp_max: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul_3: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_3, clamp_max);  convert_element_type_3 = clamp_max = None
        div: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_3, 6);  mul_3 = None
        convert_element_type_4: "bf16[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_1: "bf16[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_4, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  arg6_1 = None
        convert_element_type_5: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        unsqueeze_8: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, -1);  convert_element_type_5 = None
        unsqueeze_9: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        convert_element_type_6: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        add_3: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_6, 0.001);  convert_element_type_6 = None
        sqrt_1: "f32[16][1]cuda:0" = torch.ops.aten.sqrt.default(add_3);  add_3 = None
        reciprocal_1: "f32[16][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_4: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4, -1);  mul_4 = None
        unsqueeze_11: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_5: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_13: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_6: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_13);  mul_5 = unsqueeze_13 = None
        unsqueeze_14: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_15: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_4: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_15);  mul_6 = unsqueeze_15 = None
        convert_element_type_7: "bf16[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        relu: "bf16[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_7);  convert_element_type_7 = None
        convolution_2: "bf16[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(relu, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu = arg11_1 = None
        convert_element_type_8: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg12_1, torch.float32);  arg12_1 = None
        unsqueeze_16: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
        unsqueeze_17: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_2: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        convert_element_type_9: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        add_5: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_9, 0.001);  convert_element_type_9 = None
        sqrt_2: "f32[16][1]cuda:0" = torch.ops.aten.sqrt.default(add_5);  add_5 = None
        reciprocal_2: "f32[16][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_7: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_7, -1);  mul_7 = None
        unsqueeze_19: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_8: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_21: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_9: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_21);  mul_8 = unsqueeze_21 = None
        unsqueeze_22: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_23: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_6: "f32[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, unsqueeze_23);  mul_9 = unsqueeze_23 = None
        convert_element_type_10: "bf16[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_7: "bf16[32, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_10, convert_element_type_4);  convert_element_type_10 = convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_3: "bf16[32, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(add_7, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_7 = arg16_1 = None
        convert_element_type_11: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        unsqueeze_24: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_11, -1);  convert_element_type_11 = None
        unsqueeze_25: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_3: "f32[32, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        convert_element_type_12: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg18_1, torch.float32);  arg18_1 = None
        add_8: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_12, 0.001);  convert_element_type_12 = None
        sqrt_3: "f32[64][1]cuda:0" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
        reciprocal_3: "f32[64][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_10: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_10, -1);  mul_10 = None
        unsqueeze_27: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_11: "f32[32, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_29: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_12: "f32[32, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_29);  mul_11 = unsqueeze_29 = None
        unsqueeze_30: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_31: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_9: "f32[32, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_12, unsqueeze_31);  mul_12 = unsqueeze_31 = None
        convert_element_type_13: "bf16[32, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        relu_1: "bf16[32, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_13);  convert_element_type_13 = None
        convolution_4: "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, arg21_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  relu_1 = arg21_1 = None
        convert_element_type_14: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg22_1, torch.float32);  arg22_1 = None
        unsqueeze_32: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
        unsqueeze_33: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        sub_4: "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        convert_element_type_15: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg23_1, torch.float32);  arg23_1 = None
        add_10: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_15, 0.001);  convert_element_type_15 = None
        sqrt_4: "f32[64][1]cuda:0" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
        reciprocal_4: "f32[64][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_13: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_34: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_13, -1);  mul_13 = None
        unsqueeze_35: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_14: "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
        unsqueeze_36: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_37: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_15: "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_37);  mul_14 = unsqueeze_37 = None
        unsqueeze_38: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_39: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_11: "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, unsqueeze_39);  mul_15 = unsqueeze_39 = None
        convert_element_type_16: "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        relu_2: "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_16);  convert_element_type_16 = None
        convolution_5: "bf16[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg26_1 = None
        convert_element_type_17: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg27_1, torch.float32);  arg27_1 = None
        unsqueeze_40: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_17, -1);  convert_element_type_17 = None
        unsqueeze_41: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        sub_5: "f32[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        convert_element_type_18: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg28_1, torch.float32);  arg28_1 = None
        add_12: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_18, 0.001);  convert_element_type_18 = None
        sqrt_5: "f32[24][1]cuda:0" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
        reciprocal_5: "f32[24][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_16: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_42: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_16, -1);  mul_16 = None
        unsqueeze_43: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_17: "f32[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
        unsqueeze_44: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_45: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_18: "f32[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_45);  mul_17 = unsqueeze_45 = None
        unsqueeze_46: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_47: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_13: "f32[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, unsqueeze_47);  mul_18 = unsqueeze_47 = None
        convert_element_type_19: "bf16[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        convolution_6: "bf16[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_19, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg31_1 = None
        convert_element_type_20: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg32_1, torch.float32);  arg32_1 = None
        unsqueeze_48: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
        unsqueeze_49: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        sub_6: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        convert_element_type_21: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg33_1, torch.float32);  arg33_1 = None
        add_14: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_21, 0.001);  convert_element_type_21 = None
        sqrt_6: "f32[72][1]cuda:0" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
        reciprocal_6: "f32[72][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_19: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_50: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_19, -1);  mul_19 = None
        unsqueeze_51: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_20: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
        unsqueeze_52: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_53: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_21: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, unsqueeze_53);  mul_20 = unsqueeze_53 = None
        unsqueeze_54: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_55: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_15: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_55);  mul_21 = unsqueeze_55 = None
        convert_element_type_22: "bf16[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        relu_3: "bf16[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_22);  convert_element_type_22 = None
        convolution_7: "bf16[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72);  relu_3 = arg36_1 = None
        convert_element_type_23: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg37_1, torch.float32);  arg37_1 = None
        unsqueeze_56: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_23, -1);  convert_element_type_23 = None
        unsqueeze_57: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        sub_7: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        convert_element_type_24: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg38_1, torch.float32);  arg38_1 = None
        add_16: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_24, 0.001);  convert_element_type_24 = None
        sqrt_7: "f32[72][1]cuda:0" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
        reciprocal_7: "f32[72][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_22: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_58: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_22, -1);  mul_22 = None
        unsqueeze_59: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_23: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
        unsqueeze_60: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_61: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_24: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, unsqueeze_61);  mul_23 = unsqueeze_61 = None
        unsqueeze_62: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_63: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_17: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_24, unsqueeze_63);  mul_24 = unsqueeze_63 = None
        convert_element_type_25: "bf16[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None
        relu_4: "bf16[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_25);  convert_element_type_25 = None
        convolution_8: "bf16[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_4 = arg41_1 = None
        convert_element_type_26: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg42_1, torch.float32);  arg42_1 = None
        unsqueeze_64: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
        unsqueeze_65: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        sub_8: "f32[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
        convert_element_type_27: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg43_1, torch.float32);  arg43_1 = None
        add_18: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_27, 0.001);  convert_element_type_27 = None
        sqrt_8: "f32[24][1]cuda:0" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
        reciprocal_8: "f32[24][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_25: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_66: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_25, -1);  mul_25 = None
        unsqueeze_67: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_26: "f32[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
        unsqueeze_68: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_69: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_27: "f32[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, unsqueeze_69);  mul_26 = unsqueeze_69 = None
        unsqueeze_70: "bf16[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_71: "bf16[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_19: "f32[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_71);  mul_27 = unsqueeze_71 = None
        convert_element_type_28: "bf16[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_20: "bf16[32, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_28, convert_element_type_19);  convert_element_type_28 = convert_element_type_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_9: "bf16[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(add_20, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_20 = arg46_1 = None
        convert_element_type_29: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg47_1, torch.float32);  arg47_1 = None
        unsqueeze_72: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_29, -1);  convert_element_type_29 = None
        unsqueeze_73: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        sub_9: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
        convert_element_type_30: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg48_1, torch.float32);  arg48_1 = None
        add_21: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_30, 0.001);  convert_element_type_30 = None
        sqrt_9: "f32[72][1]cuda:0" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
        reciprocal_9: "f32[72][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_28: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_74: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_28, -1);  mul_28 = None
        unsqueeze_75: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_29: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
        unsqueeze_76: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_77: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_30: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_77);  mul_29 = unsqueeze_77 = None
        unsqueeze_78: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_79: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_22: "f32[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_79);  mul_30 = unsqueeze_79 = None
        convert_element_type_31: "bf16[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None
        relu_5: "bf16[32, 72, 56, 56][225792, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_31);  convert_element_type_31 = None
        convolution_10: "bf16[32, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, arg51_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72);  relu_5 = arg51_1 = None
        convert_element_type_32: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg52_1, torch.float32);  arg52_1 = None
        unsqueeze_80: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
        unsqueeze_81: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        sub_10: "f32[32, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
        convert_element_type_33: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg53_1, torch.float32);  arg53_1 = None
        add_23: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_33, 0.001);  convert_element_type_33 = None
        sqrt_10: "f32[72][1]cuda:0" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
        reciprocal_10: "f32[72][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_31: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_82: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_31, -1);  mul_31 = None
        unsqueeze_83: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_32: "f32[32, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
        unsqueeze_84: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_85: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_33: "f32[32, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, unsqueeze_85);  mul_32 = unsqueeze_85 = None
        unsqueeze_86: "bf16[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_87: "bf16[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_24: "f32[32, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_87);  mul_33 = unsqueeze_87 = None
        convert_element_type_34: "bf16[32, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None
        relu_6: "bf16[32, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_34);  convert_element_type_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean: "bf16[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_6, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_11: "bf16[32, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean, arg56_1, arg57_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg56_1 = arg57_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_7: "bf16[32, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_12: "bf16[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, arg58_1, arg59_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_7 = arg58_1 = arg59_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_35: "f32[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        add_25: "f32[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_35, 3);  convert_element_type_35 = None
        clamp_min_1: "f32[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_25, 0);  add_25 = None
        clamp_max_1: "f32[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
        div_1: "f32[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
        convert_element_type_36: "bf16[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_34: "bf16[32, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_36, relu_6);  convert_element_type_36 = relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_13: "bf16[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(mul_34, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_34 = arg60_1 = None
        convert_element_type_37: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg61_1, torch.float32);  arg61_1 = None
        unsqueeze_88: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_37, -1);  convert_element_type_37 = None
        unsqueeze_89: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        sub_11: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_89);  convolution_13 = unsqueeze_89 = None
        convert_element_type_38: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg62_1, torch.float32);  arg62_1 = None
        add_26: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_38, 0.001);  convert_element_type_38 = None
        sqrt_11: "f32[40][1]cuda:0" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
        reciprocal_11: "f32[40][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_35: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_90: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_35, -1);  mul_35 = None
        unsqueeze_91: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_36: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
        unsqueeze_92: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_93: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_37: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_93);  mul_36 = unsqueeze_93 = None
        unsqueeze_94: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_95: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_27: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_95);  mul_37 = unsqueeze_95 = None
        convert_element_type_39: "bf16[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None
        convolution_14: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_39, arg65_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg65_1 = None
        convert_element_type_40: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg66_1, torch.float32);  arg66_1 = None
        unsqueeze_96: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
        unsqueeze_97: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        sub_12: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_97);  convolution_14 = unsqueeze_97 = None
        convert_element_type_41: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg67_1, torch.float32);  arg67_1 = None
        add_28: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_41, 0.001);  convert_element_type_41 = None
        sqrt_12: "f32[120][1]cuda:0" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        reciprocal_12: "f32[120][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_38: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_98: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_38, -1);  mul_38 = None
        unsqueeze_99: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_39: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
        unsqueeze_100: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_101: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_40: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, unsqueeze_101);  mul_39 = unsqueeze_101 = None
        unsqueeze_102: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_103: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_29: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_40, unsqueeze_103);  mul_40 = unsqueeze_103 = None
        convert_element_type_42: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None
        relu_8: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_42);  convert_element_type_42 = None
        convolution_15: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, arg70_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  relu_8 = arg70_1 = None
        convert_element_type_43: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg71_1, torch.float32);  arg71_1 = None
        unsqueeze_104: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_43, -1);  convert_element_type_43 = None
        unsqueeze_105: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        sub_13: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_105);  convolution_15 = unsqueeze_105 = None
        convert_element_type_44: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg72_1, torch.float32);  arg72_1 = None
        add_30: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_44, 0.001);  convert_element_type_44 = None
        sqrt_13: "f32[120][1]cuda:0" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
        reciprocal_13: "f32[120][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_41: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_106: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_41, -1);  mul_41 = None
        unsqueeze_107: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_42: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
        unsqueeze_108: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_109: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_43: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_109);  mul_42 = unsqueeze_109 = None
        unsqueeze_110: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_111: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_31: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_111);  mul_43 = unsqueeze_111 = None
        convert_element_type_45: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        relu_9: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_45);  convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_1: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_9, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_16: "bf16[32, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_1, arg75_1, arg76_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg75_1 = arg76_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_10: "bf16[32, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_17: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, arg77_1, arg78_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_10 = arg77_1 = arg78_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_46: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        add_32: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_46, 3);  convert_element_type_46 = None
        clamp_min_2: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_32, 0);  add_32 = None
        clamp_max_2: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
        div_2: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
        convert_element_type_47: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_44: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_47, relu_9);  convert_element_type_47 = relu_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_18: "bf16[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(mul_44, arg79_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_44 = arg79_1 = None
        convert_element_type_48: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg80_1, torch.float32);  arg80_1 = None
        unsqueeze_112: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
        unsqueeze_113: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        sub_14: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_113);  convolution_18 = unsqueeze_113 = None
        convert_element_type_49: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg81_1, torch.float32);  arg81_1 = None
        add_33: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_49, 0.001);  convert_element_type_49 = None
        sqrt_14: "f32[40][1]cuda:0" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
        reciprocal_14: "f32[40][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_45: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_114: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
        unsqueeze_115: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_46: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
        unsqueeze_116: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_117: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_47: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_117);  mul_46 = unsqueeze_117 = None
        unsqueeze_118: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_119: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_34: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_119);  mul_47 = unsqueeze_119 = None
        convert_element_type_50: "bf16[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_35: "bf16[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_50, convert_element_type_39);  convert_element_type_50 = convert_element_type_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_19: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(add_35, arg84_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg84_1 = None
        convert_element_type_51: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg85_1, torch.float32);  arg85_1 = None
        unsqueeze_120: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_51, -1);  convert_element_type_51 = None
        unsqueeze_121: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        sub_15: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_121);  convolution_19 = unsqueeze_121 = None
        convert_element_type_52: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg86_1, torch.float32);  arg86_1 = None
        add_36: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_52, 0.001);  convert_element_type_52 = None
        sqrt_15: "f32[120][1]cuda:0" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
        reciprocal_15: "f32[120][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_48: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_122: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
        unsqueeze_123: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_49: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
        unsqueeze_124: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_125: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_50: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_125);  mul_49 = unsqueeze_125 = None
        unsqueeze_126: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_127: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_37: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_127);  mul_50 = unsqueeze_127 = None
        convert_element_type_53: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None
        relu_11: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_53);  convert_element_type_53 = None
        convolution_20: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_11, arg89_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  relu_11 = arg89_1 = None
        convert_element_type_54: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg90_1, torch.float32);  arg90_1 = None
        unsqueeze_128: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
        unsqueeze_129: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        sub_16: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_129);  convolution_20 = unsqueeze_129 = None
        convert_element_type_55: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg91_1, torch.float32);  arg91_1 = None
        add_38: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_55, 0.001);  convert_element_type_55 = None
        sqrt_16: "f32[120][1]cuda:0" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
        reciprocal_16: "f32[120][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_51: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_130: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
        unsqueeze_131: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_52: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
        unsqueeze_132: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_133: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_53: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_133);  mul_52 = unsqueeze_133 = None
        unsqueeze_134: "bf16[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_135: "bf16[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_39: "f32[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_135);  mul_53 = unsqueeze_135 = None
        convert_element_type_56: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        relu_12: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_56);  convert_element_type_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_2: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_12, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_21: "bf16[32, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_2, arg94_1, arg95_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg94_1 = arg95_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_13: "bf16[32, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_22: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, arg96_1, arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_13 = arg96_1 = arg97_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_57: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        add_40: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_57, 3);  convert_element_type_57 = None
        clamp_min_3: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_40, 0);  add_40 = None
        clamp_max_3: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
        div_3: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
        convert_element_type_58: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_54: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, relu_12);  convert_element_type_58 = relu_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_23: "bf16[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(mul_54, arg98_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_54 = arg98_1 = None
        convert_element_type_59: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg99_1, torch.float32);  arg99_1 = None
        unsqueeze_136: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_59, -1);  convert_element_type_59 = None
        unsqueeze_137: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        sub_17: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_137);  convolution_23 = unsqueeze_137 = None
        convert_element_type_60: "f32[40][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg100_1, torch.float32);  arg100_1 = None
        add_41: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_60, 0.001);  convert_element_type_60 = None
        sqrt_17: "f32[40][1]cuda:0" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
        reciprocal_17: "f32[40][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_55: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        unsqueeze_138: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_55, -1);  mul_55 = None
        unsqueeze_139: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        mul_56: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
        unsqueeze_140: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_141: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_57: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_141);  mul_56 = unsqueeze_141 = None
        unsqueeze_142: "bf16[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_143: "bf16[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_42: "f32[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_143);  mul_57 = unsqueeze_143 = None
        convert_element_type_61: "bf16[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_42, torch.bfloat16);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_43: "bf16[32, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_61, add_35);  convert_element_type_61 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_24: "bf16[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(add_43, arg103_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_43 = arg103_1 = None
        convert_element_type_62: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg104_1, torch.float32);  arg104_1 = None
        unsqueeze_144: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
        unsqueeze_145: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        sub_18: "f32[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_145);  convolution_24 = unsqueeze_145 = None
        convert_element_type_63: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg105_1, torch.float32);  arg105_1 = None
        add_44: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_63, 0.001);  convert_element_type_63 = None
        sqrt_18: "f32[240][1]cuda:0" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
        reciprocal_18: "f32[240][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_58: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        unsqueeze_146: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_58, -1);  mul_58 = None
        unsqueeze_147: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        mul_59: "f32[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
        unsqueeze_148: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_149: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_60: "f32[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, unsqueeze_149);  mul_59 = unsqueeze_149 = None
        unsqueeze_150: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_151: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_45: "f32[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, unsqueeze_151);  mul_60 = unsqueeze_151 = None
        convert_element_type_64: "bf16[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None
        convert_element_type_65: "f32[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_64, torch.float32);  convert_element_type_64 = None
        add_46: "f32[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_65, 3)
        clamp_min_4: "f32[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_46, 0);  add_46 = None
        clamp_max_4: "f32[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
        mul_61: "f32[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_65, clamp_max_4);  convert_element_type_65 = clamp_max_4 = None
        div_4: "f32[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_61, 6);  mul_61 = None
        convert_element_type_66: "bf16[32, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        convolution_25: "bf16[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_66, arg108_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240);  convert_element_type_66 = arg108_1 = None
        convert_element_type_67: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg109_1, torch.float32);  arg109_1 = None
        unsqueeze_152: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_67, -1);  convert_element_type_67 = None
        unsqueeze_153: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        sub_19: "f32[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_153);  convolution_25 = unsqueeze_153 = None
        convert_element_type_68: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg110_1, torch.float32);  arg110_1 = None
        add_47: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_68, 0.001);  convert_element_type_68 = None
        sqrt_19: "f32[240][1]cuda:0" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
        reciprocal_19: "f32[240][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_62: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        unsqueeze_154: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_62, -1);  mul_62 = None
        unsqueeze_155: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        mul_63: "f32[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
        unsqueeze_156: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_157: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_64: "f32[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_157);  mul_63 = unsqueeze_157 = None
        unsqueeze_158: "bf16[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_159: "bf16[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_48: "f32[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_159);  mul_64 = unsqueeze_159 = None
        convert_element_type_69: "bf16[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None
        convert_element_type_70: "f32[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_69, torch.float32);  convert_element_type_69 = None
        add_49: "f32[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_70, 3)
        clamp_min_5: "f32[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_49, 0);  add_49 = None
        clamp_max_5: "f32[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
        mul_65: "f32[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_70, clamp_max_5);  convert_element_type_70 = clamp_max_5 = None
        div_5: "f32[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_65, 6);  mul_65 = None
        convert_element_type_71: "bf16[32, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        convolution_26: "bf16[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_71, arg113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_71 = arg113_1 = None
        convert_element_type_72: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg114_1, torch.float32);  arg114_1 = None
        unsqueeze_160: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
        unsqueeze_161: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        sub_20: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_161);  convolution_26 = unsqueeze_161 = None
        convert_element_type_73: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg115_1, torch.float32);  arg115_1 = None
        add_50: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_73, 0.001);  convert_element_type_73 = None
        sqrt_20: "f32[80][1]cuda:0" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
        reciprocal_20: "f32[80][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        mul_66: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        unsqueeze_162: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
        unsqueeze_163: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        mul_67: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
        unsqueeze_164: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_165: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_68: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_165);  mul_67 = unsqueeze_165 = None
        unsqueeze_166: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_167: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_51: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_167);  mul_68 = unsqueeze_167 = None
        convert_element_type_74: "bf16[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None
        convolution_27: "bf16[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_74, arg118_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg118_1 = None
        convert_element_type_75: "f32[200][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg119_1, torch.float32);  arg119_1 = None
        unsqueeze_168: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_75, -1);  convert_element_type_75 = None
        unsqueeze_169: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        sub_21: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_169);  convolution_27 = unsqueeze_169 = None
        convert_element_type_76: "f32[200][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg120_1, torch.float32);  arg120_1 = None
        add_52: "f32[200][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_76, 0.001);  convert_element_type_76 = None
        sqrt_21: "f32[200][1]cuda:0" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
        reciprocal_21: "f32[200][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        mul_69: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        unsqueeze_170: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
        unsqueeze_171: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        mul_70: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
        unsqueeze_172: "bf16[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_173: "bf16[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_71: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_173);  mul_70 = unsqueeze_173 = None
        unsqueeze_174: "bf16[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_175: "bf16[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_53: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_175);  mul_71 = unsqueeze_175 = None
        convert_element_type_77: "bf16[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None
        convert_element_type_78: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_77, torch.float32);  convert_element_type_77 = None
        add_54: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_78, 3)
        clamp_min_6: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_54, 0);  add_54 = None
        clamp_max_6: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
        mul_72: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_78, clamp_max_6);  convert_element_type_78 = clamp_max_6 = None
        div_6: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_72, 6);  mul_72 = None
        convert_element_type_79: "bf16[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        convolution_28: "bf16[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_79, arg123_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 200);  convert_element_type_79 = arg123_1 = None
        convert_element_type_80: "f32[200][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg124_1, torch.float32);  arg124_1 = None
        unsqueeze_176: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
        unsqueeze_177: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        sub_22: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_177);  convolution_28 = unsqueeze_177 = None
        convert_element_type_81: "f32[200][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg125_1, torch.float32);  arg125_1 = None
        add_55: "f32[200][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_81, 0.001);  convert_element_type_81 = None
        sqrt_22: "f32[200][1]cuda:0" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
        reciprocal_22: "f32[200][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        mul_73: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        unsqueeze_178: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_73, -1);  mul_73 = None
        unsqueeze_179: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        mul_74: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
        unsqueeze_180: "bf16[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_181: "bf16[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_75: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, unsqueeze_181);  mul_74 = unsqueeze_181 = None
        unsqueeze_182: "bf16[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_183: "bf16[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_56: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_75, unsqueeze_183);  mul_75 = unsqueeze_183 = None
        convert_element_type_82: "bf16[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None
        convert_element_type_83: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_82, torch.float32);  convert_element_type_82 = None
        add_57: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_83, 3)
        clamp_min_7: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_57, 0);  add_57 = None
        clamp_max_7: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
        mul_76: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_83, clamp_max_7);  convert_element_type_83 = clamp_max_7 = None
        div_7: "f32[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_76, 6);  mul_76 = None
        convert_element_type_84: "bf16[32, 200, 14, 14][39200, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        convolution_29: "bf16[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_84, arg128_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_84 = arg128_1 = None
        convert_element_type_85: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg129_1, torch.float32);  arg129_1 = None
        unsqueeze_184: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_85, -1);  convert_element_type_85 = None
        unsqueeze_185: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        sub_23: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_185);  convolution_29 = unsqueeze_185 = None
        convert_element_type_86: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg130_1, torch.float32);  arg130_1 = None
        add_58: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_86, 0.001);  convert_element_type_86 = None
        sqrt_23: "f32[80][1]cuda:0" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
        reciprocal_23: "f32[80][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_77: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        unsqueeze_186: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_77, -1);  mul_77 = None
        unsqueeze_187: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        mul_78: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
        unsqueeze_188: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_189: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_79: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, unsqueeze_189);  mul_78 = unsqueeze_189 = None
        unsqueeze_190: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_191: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_59: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_191);  mul_79 = unsqueeze_191 = None
        convert_element_type_87: "bf16[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_60: "bf16[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_87, convert_element_type_74);  convert_element_type_87 = convert_element_type_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_30: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(add_60, arg133_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg133_1 = None
        convert_element_type_88: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg134_1, torch.float32);  arg134_1 = None
        unsqueeze_192: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
        unsqueeze_193: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        sub_24: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_193);  convolution_30 = unsqueeze_193 = None
        convert_element_type_89: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg135_1, torch.float32);  arg135_1 = None
        add_61: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_89, 0.001);  convert_element_type_89 = None
        sqrt_24: "f32[184][1]cuda:0" = torch.ops.aten.sqrt.default(add_61);  add_61 = None
        reciprocal_24: "f32[184][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_80: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        unsqueeze_194: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_80, -1);  mul_80 = None
        unsqueeze_195: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        mul_81: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
        unsqueeze_196: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_197: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_82: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_197);  mul_81 = unsqueeze_197 = None
        unsqueeze_198: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_199: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_62: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, unsqueeze_199);  mul_82 = unsqueeze_199 = None
        convert_element_type_90: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None
        convert_element_type_91: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_90, torch.float32);  convert_element_type_90 = None
        add_63: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_91, 3)
        clamp_min_8: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_63, 0);  add_63 = None
        clamp_max_8: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
        mul_83: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, clamp_max_8);  convert_element_type_91 = clamp_max_8 = None
        div_8: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_83, 6);  mul_83 = None
        convert_element_type_92: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        convolution_31: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_92, arg138_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184);  convert_element_type_92 = arg138_1 = None
        convert_element_type_93: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg139_1, torch.float32);  arg139_1 = None
        unsqueeze_200: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_93, -1);  convert_element_type_93 = None
        unsqueeze_201: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        sub_25: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_201);  convolution_31 = unsqueeze_201 = None
        convert_element_type_94: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float32);  arg140_1 = None
        add_64: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_94, 0.001);  convert_element_type_94 = None
        sqrt_25: "f32[184][1]cuda:0" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
        reciprocal_25: "f32[184][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_84: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        unsqueeze_202: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
        unsqueeze_203: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        mul_85: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
        unsqueeze_204: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_205: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_86: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_205);  mul_85 = unsqueeze_205 = None
        unsqueeze_206: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_207: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_65: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_207);  mul_86 = unsqueeze_207 = None
        convert_element_type_95: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None
        convert_element_type_96: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_95, torch.float32);  convert_element_type_95 = None
        add_66: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_96, 3)
        clamp_min_9: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_66, 0);  add_66 = None
        clamp_max_9: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
        mul_87: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_96, clamp_max_9);  convert_element_type_96 = clamp_max_9 = None
        div_9: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_87, 6);  mul_87 = None
        convert_element_type_97: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        convolution_32: "bf16[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_97, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_97 = arg143_1 = None
        convert_element_type_98: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg144_1, torch.float32);  arg144_1 = None
        unsqueeze_208: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
        unsqueeze_209: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        sub_26: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_209);  convolution_32 = unsqueeze_209 = None
        convert_element_type_99: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg145_1, torch.float32);  arg145_1 = None
        add_67: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_99, 0.001);  convert_element_type_99 = None
        sqrt_26: "f32[80][1]cuda:0" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
        reciprocal_26: "f32[80][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_88: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        unsqueeze_210: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_88, -1);  mul_88 = None
        unsqueeze_211: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        mul_89: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
        unsqueeze_212: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_213: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_90: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_213);  mul_89 = unsqueeze_213 = None
        unsqueeze_214: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_215: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_68: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_215);  mul_90 = unsqueeze_215 = None
        convert_element_type_100: "bf16[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_69: "bf16[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_100, add_60);  convert_element_type_100 = add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_33: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(add_69, arg148_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg148_1 = None
        convert_element_type_101: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg149_1, torch.float32);  arg149_1 = None
        unsqueeze_216: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_101, -1);  convert_element_type_101 = None
        unsqueeze_217: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        sub_27: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_217);  convolution_33 = unsqueeze_217 = None
        convert_element_type_102: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg150_1, torch.float32);  arg150_1 = None
        add_70: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_102, 0.001);  convert_element_type_102 = None
        sqrt_27: "f32[184][1]cuda:0" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
        reciprocal_27: "f32[184][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_91: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_218: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_91, -1);  mul_91 = None
        unsqueeze_219: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        mul_92: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
        unsqueeze_220: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_221: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_93: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_221);  mul_92 = unsqueeze_221 = None
        unsqueeze_222: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_223: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_71: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_223);  mul_93 = unsqueeze_223 = None
        convert_element_type_103: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None
        convert_element_type_104: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_103, torch.float32);  convert_element_type_103 = None
        add_72: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_104, 3)
        clamp_min_10: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_72, 0);  add_72 = None
        clamp_max_10: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
        mul_94: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_104, clamp_max_10);  convert_element_type_104 = clamp_max_10 = None
        div_10: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_94, 6);  mul_94 = None
        convert_element_type_105: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        convolution_34: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_105, arg153_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184);  convert_element_type_105 = arg153_1 = None
        convert_element_type_106: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg154_1, torch.float32);  arg154_1 = None
        unsqueeze_224: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_106, -1);  convert_element_type_106 = None
        unsqueeze_225: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        sub_28: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_225);  convolution_34 = unsqueeze_225 = None
        convert_element_type_107: "f32[184][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg155_1, torch.float32);  arg155_1 = None
        add_73: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_107, 0.001);  convert_element_type_107 = None
        sqrt_28: "f32[184][1]cuda:0" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
        reciprocal_28: "f32[184][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_95: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_226: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_95, -1);  mul_95 = None
        unsqueeze_227: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        mul_96: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
        unsqueeze_228: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_229: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_97: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_229);  mul_96 = unsqueeze_229 = None
        unsqueeze_230: "bf16[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_231: "bf16[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_74: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_231);  mul_97 = unsqueeze_231 = None
        convert_element_type_108: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None
        convert_element_type_109: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_108, torch.float32);  convert_element_type_108 = None
        add_75: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_109, 3)
        clamp_min_11: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_75, 0);  add_75 = None
        clamp_max_11: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
        mul_98: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_109, clamp_max_11);  convert_element_type_109 = clamp_max_11 = None
        div_11: "f32[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_98, 6);  mul_98 = None
        convert_element_type_110: "bf16[32, 184, 14, 14][36064, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        convolution_35: "bf16[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_110, arg158_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_110 = arg158_1 = None
        convert_element_type_111: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg159_1, torch.float32);  arg159_1 = None
        unsqueeze_232: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_111, -1);  convert_element_type_111 = None
        unsqueeze_233: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        sub_29: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_233);  convolution_35 = unsqueeze_233 = None
        convert_element_type_112: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg160_1, torch.float32);  arg160_1 = None
        add_76: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_112, 0.001);  convert_element_type_112 = None
        sqrt_29: "f32[80][1]cuda:0" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
        reciprocal_29: "f32[80][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_99: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_234: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
        unsqueeze_235: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        mul_100: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
        unsqueeze_236: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_237: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_101: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_237);  mul_100 = unsqueeze_237 = None
        unsqueeze_238: "bf16[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_239: "bf16[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_77: "f32[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_239);  mul_101 = unsqueeze_239 = None
        convert_element_type_113: "bf16[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_78: "bf16[32, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_113, add_69);  convert_element_type_113 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_36: "bf16[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(add_78, arg163_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_78 = arg163_1 = None
        convert_element_type_114: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg164_1, torch.float32);  arg164_1 = None
        unsqueeze_240: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_114, -1);  convert_element_type_114 = None
        unsqueeze_241: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        sub_30: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_241);  convolution_36 = unsqueeze_241 = None
        convert_element_type_115: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg165_1, torch.float32);  arg165_1 = None
        add_79: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_115, 0.001);  convert_element_type_115 = None
        sqrt_30: "f32[480][1]cuda:0" = torch.ops.aten.sqrt.default(add_79);  add_79 = None
        reciprocal_30: "f32[480][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_102: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_242: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
        unsqueeze_243: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        mul_103: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
        unsqueeze_244: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_245: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_104: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_245);  mul_103 = unsqueeze_245 = None
        unsqueeze_246: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_247: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_80: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_247);  mul_104 = unsqueeze_247 = None
        convert_element_type_116: "bf16[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None
        convert_element_type_117: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_116, torch.float32);  convert_element_type_116 = None
        add_81: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_117, 3)
        clamp_min_12: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_81, 0);  add_81 = None
        clamp_max_12: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
        mul_105: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_117, clamp_max_12);  convert_element_type_117 = clamp_max_12 = None
        div_12: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_105, 6);  mul_105 = None
        convert_element_type_118: "bf16[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        convolution_37: "bf16[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_118, arg168_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  convert_element_type_118 = arg168_1 = None
        convert_element_type_119: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg169_1, torch.float32);  arg169_1 = None
        unsqueeze_248: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_119, -1);  convert_element_type_119 = None
        unsqueeze_249: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        sub_31: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_249);  convolution_37 = unsqueeze_249 = None
        convert_element_type_120: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg170_1, torch.float32);  arg170_1 = None
        add_82: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_120, 0.001);  convert_element_type_120 = None
        sqrt_31: "f32[480][1]cuda:0" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
        reciprocal_31: "f32[480][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_106: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_250: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_106, -1);  mul_106 = None
        unsqueeze_251: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        mul_107: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
        unsqueeze_252: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_253: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_108: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_253);  mul_107 = unsqueeze_253 = None
        unsqueeze_254: "bf16[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_255: "bf16[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_83: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_108, unsqueeze_255);  mul_108 = unsqueeze_255 = None
        convert_element_type_121: "bf16[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None
        convert_element_type_122: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_121, torch.float32);  convert_element_type_121 = None
        add_84: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_122, 3)
        clamp_min_13: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_84, 0);  add_84 = None
        clamp_max_13: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
        mul_109: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, clamp_max_13);  convert_element_type_122 = clamp_max_13 = None
        div_13: "f32[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_109, 6);  mul_109 = None
        convert_element_type_123: "bf16[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_3: "bf16[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_123, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_38: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_3, arg173_1, arg174_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg173_1 = arg174_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_14: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_38);  convolution_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_39: "bf16[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_14, arg175_1, arg176_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = arg175_1 = arg176_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_124: "f32[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        add_85: "f32[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_124, 3);  convert_element_type_124 = None
        clamp_min_14: "f32[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_85, 0);  add_85 = None
        clamp_max_14: "f32[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
        div_14: "f32[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_14, 6);  clamp_max_14 = None
        convert_element_type_125: "bf16[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_110: "bf16[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_125, convert_element_type_123);  convert_element_type_125 = convert_element_type_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_40: "bf16[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(mul_110, arg177_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_110 = arg177_1 = None
        convert_element_type_126: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg178_1, torch.float32);  arg178_1 = None
        unsqueeze_256: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_126, -1);  convert_element_type_126 = None
        unsqueeze_257: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        sub_32: "f32[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_257);  convolution_40 = unsqueeze_257 = None
        convert_element_type_127: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg179_1, torch.float32);  arg179_1 = None
        add_86: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_127, 0.001);  convert_element_type_127 = None
        sqrt_32: "f32[112][1]cuda:0" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
        reciprocal_32: "f32[112][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_111: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_258: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
        unsqueeze_259: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        mul_112: "f32[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
        unsqueeze_260: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_261: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_113: "f32[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_261);  mul_112 = unsqueeze_261 = None
        unsqueeze_262: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_263: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_87: "f32[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_263);  mul_113 = unsqueeze_263 = None
        convert_element_type_128: "bf16[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None
        convolution_41: "bf16[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_128, arg182_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg182_1 = None
        convert_element_type_129: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg183_1, torch.float32);  arg183_1 = None
        unsqueeze_264: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_129, -1);  convert_element_type_129 = None
        unsqueeze_265: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        sub_33: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_265);  convolution_41 = unsqueeze_265 = None
        convert_element_type_130: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg184_1, torch.float32);  arg184_1 = None
        add_88: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_130, 0.001);  convert_element_type_130 = None
        sqrt_33: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
        reciprocal_33: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_114: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_266: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
        unsqueeze_267: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        mul_115: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
        unsqueeze_268: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_269: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_116: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_269);  mul_115 = unsqueeze_269 = None
        unsqueeze_270: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_271: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_89: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_271);  mul_116 = unsqueeze_271 = None
        convert_element_type_131: "bf16[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None
        convert_element_type_132: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_131, torch.float32);  convert_element_type_131 = None
        add_90: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_132, 3)
        clamp_min_15: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_90, 0);  add_90 = None
        clamp_max_15: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
        mul_117: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_132, clamp_max_15);  convert_element_type_132 = clamp_max_15 = None
        div_15: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_117, 6);  mul_117 = None
        convert_element_type_133: "bf16[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        convolution_42: "bf16[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_133, arg187_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 672);  convert_element_type_133 = arg187_1 = None
        convert_element_type_134: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg188_1, torch.float32);  arg188_1 = None
        unsqueeze_272: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_134, -1);  convert_element_type_134 = None
        unsqueeze_273: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        sub_34: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_273);  convolution_42 = unsqueeze_273 = None
        convert_element_type_135: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg189_1, torch.float32);  arg189_1 = None
        add_91: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_135, 0.001);  convert_element_type_135 = None
        sqrt_34: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
        reciprocal_34: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_118: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_274: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_118, -1);  mul_118 = None
        unsqueeze_275: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        mul_119: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
        unsqueeze_276: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_277: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_120: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_277);  mul_119 = unsqueeze_277 = None
        unsqueeze_278: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
        unsqueeze_279: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_92: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, unsqueeze_279);  mul_120 = unsqueeze_279 = None
        convert_element_type_136: "bf16[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_92, torch.bfloat16);  add_92 = None
        convert_element_type_137: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_136, torch.float32);  convert_element_type_136 = None
        add_93: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_137, 3)
        clamp_min_16: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_93, 0);  add_93 = None
        clamp_max_16: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
        mul_121: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_137, clamp_max_16);  convert_element_type_137 = clamp_max_16 = None
        div_16: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_121, 6);  mul_121 = None
        convert_element_type_138: "bf16[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_4: "bf16[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_138, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_43: "bf16[32, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_4, arg192_1, arg193_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_4 = arg192_1 = arg193_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_15: "bf16[32, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_43);  convolution_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_44: "bf16[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_15, arg194_1, arg195_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_15 = arg194_1 = arg195_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_139: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        add_94: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_139, 3);  convert_element_type_139 = None
        clamp_min_17: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_94, 0);  add_94 = None
        clamp_max_17: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
        div_17: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_17, 6);  clamp_max_17 = None
        convert_element_type_140: "bf16[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_122: "bf16[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, convert_element_type_138);  convert_element_type_140 = convert_element_type_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_45: "bf16[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(mul_122, arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_122 = arg196_1 = None
        convert_element_type_141: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg197_1, torch.float32);  arg197_1 = None
        unsqueeze_280: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_141, -1);  convert_element_type_141 = None
        unsqueeze_281: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        sub_35: "f32[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_281);  convolution_45 = unsqueeze_281 = None
        convert_element_type_142: "f32[112][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg198_1, torch.float32);  arg198_1 = None
        add_95: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_142, 0.001);  convert_element_type_142 = None
        sqrt_35: "f32[112][1]cuda:0" = torch.ops.aten.sqrt.default(add_95);  add_95 = None
        reciprocal_35: "f32[112][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_123: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_282: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_283: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        mul_124: "f32[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
        unsqueeze_284: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_285: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_125: "f32[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_285);  mul_124 = unsqueeze_285 = None
        unsqueeze_286: "bf16[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_287: "bf16[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_96: "f32[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_287);  mul_125 = unsqueeze_287 = None
        convert_element_type_143: "bf16[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16);  add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_97: "bf16[32, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_143, convert_element_type_128);  convert_element_type_143 = convert_element_type_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_46: "bf16[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(add_97, arg201_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_97 = arg201_1 = None
        convert_element_type_144: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg202_1, torch.float32);  arg202_1 = None
        unsqueeze_288: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_144, -1);  convert_element_type_144 = None
        unsqueeze_289: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        sub_36: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_289);  convolution_46 = unsqueeze_289 = None
        convert_element_type_145: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg203_1, torch.float32);  arg203_1 = None
        add_98: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_145, 0.001);  convert_element_type_145 = None
        sqrt_36: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
        reciprocal_36: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_126: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_290: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
        unsqueeze_291: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        mul_127: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
        unsqueeze_292: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_293: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_128: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_293);  mul_127 = unsqueeze_293 = None
        unsqueeze_294: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_295: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_99: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_295);  mul_128 = unsqueeze_295 = None
        convert_element_type_146: "bf16[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None
        convert_element_type_147: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_146, torch.float32);  convert_element_type_146 = None
        add_100: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_147, 3)
        clamp_min_18: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_100, 0);  add_100 = None
        clamp_max_18: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
        mul_129: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_147, clamp_max_18);  convert_element_type_147 = clamp_max_18 = None
        div_18: "f32[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_129, 6);  mul_129 = None
        convert_element_type_148: "bf16[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        convolution_47: "bf16[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_148, arg206_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672);  convert_element_type_148 = arg206_1 = None
        convert_element_type_149: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg207_1, torch.float32);  arg207_1 = None
        unsqueeze_296: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_149, -1);  convert_element_type_149 = None
        unsqueeze_297: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        sub_37: "f32[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_297);  convolution_47 = unsqueeze_297 = None
        convert_element_type_150: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg208_1, torch.float32);  arg208_1 = None
        add_101: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_150, 0.001);  convert_element_type_150 = None
        sqrt_37: "f32[672][1]cuda:0" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
        reciprocal_37: "f32[672][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_130: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_298: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_130, -1);  mul_130 = None
        unsqueeze_299: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        mul_131: "f32[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
        unsqueeze_300: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_301: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_132: "f32[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_301);  mul_131 = unsqueeze_301 = None
        unsqueeze_302: "bf16[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_303: "bf16[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_102: "f32[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_303);  mul_132 = unsqueeze_303 = None
        convert_element_type_151: "bf16[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16);  add_102 = None
        convert_element_type_152: "f32[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_151, torch.float32);  convert_element_type_151 = None
        add_103: "f32[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_152, 3)
        clamp_min_19: "f32[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_103, 0);  add_103 = None
        clamp_max_19: "f32[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
        mul_133: "f32[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_152, clamp_max_19);  convert_element_type_152 = clamp_max_19 = None
        div_19: "f32[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_133, 6);  mul_133 = None
        convert_element_type_153: "bf16[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_5: "bf16[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_153, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_48: "bf16[32, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_5, arg211_1, arg212_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg211_1 = arg212_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_16: "bf16[32, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_48);  convolution_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_49: "bf16[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_16, arg213_1, arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_16 = arg213_1 = arg214_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_154: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        add_104: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_154, 3);  convert_element_type_154 = None
        clamp_min_20: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_104, 0);  add_104 = None
        clamp_max_20: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
        div_20: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_20, 6);  clamp_max_20 = None
        convert_element_type_155: "bf16[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_134: "bf16[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_155, convert_element_type_153);  convert_element_type_155 = convert_element_type_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_50: "bf16[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(mul_134, arg215_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_134 = arg215_1 = None
        convert_element_type_156: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg216_1, torch.float32);  arg216_1 = None
        unsqueeze_304: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_156, -1);  convert_element_type_156 = None
        unsqueeze_305: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        sub_38: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_305);  convolution_50 = unsqueeze_305 = None
        convert_element_type_157: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg217_1, torch.float32);  arg217_1 = None
        add_105: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_157, 0.001);  convert_element_type_157 = None
        sqrt_38: "f32[160][1]cuda:0" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
        reciprocal_38: "f32[160][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_135: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_306: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
        unsqueeze_307: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        mul_136: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
        unsqueeze_308: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_309: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_137: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_309);  mul_136 = unsqueeze_309 = None
        unsqueeze_310: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_311: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_106: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_311);  mul_137 = unsqueeze_311 = None
        convert_element_type_158: "bf16[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16);  add_106 = None
        convolution_51: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_158, arg220_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg220_1 = None
        convert_element_type_159: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg221_1, torch.float32);  arg221_1 = None
        unsqueeze_312: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_159, -1);  convert_element_type_159 = None
        unsqueeze_313: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        sub_39: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_313);  convolution_51 = unsqueeze_313 = None
        convert_element_type_160: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg222_1, torch.float32);  arg222_1 = None
        add_107: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_160, 0.001);  convert_element_type_160 = None
        sqrt_39: "f32[960][1]cuda:0" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
        reciprocal_39: "f32[960][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_138: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_314: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
        unsqueeze_315: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        mul_139: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
        unsqueeze_316: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_317: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_140: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_317);  mul_139 = unsqueeze_317 = None
        unsqueeze_318: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_319: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_108: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_319);  mul_140 = unsqueeze_319 = None
        convert_element_type_161: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.bfloat16);  add_108 = None
        convert_element_type_162: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_161, torch.float32);  convert_element_type_161 = None
        add_109: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_162, 3)
        clamp_min_21: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_109, 0);  add_109 = None
        clamp_max_21: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
        mul_141: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_162, clamp_max_21);  convert_element_type_162 = clamp_max_21 = None
        div_21: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_141, 6);  mul_141 = None
        convert_element_type_163: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        convolution_52: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_163, arg225_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960);  convert_element_type_163 = arg225_1 = None
        convert_element_type_164: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg226_1, torch.float32);  arg226_1 = None
        unsqueeze_320: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_164, -1);  convert_element_type_164 = None
        unsqueeze_321: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        sub_40: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_321);  convolution_52 = unsqueeze_321 = None
        convert_element_type_165: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg227_1, torch.float32);  arg227_1 = None
        add_110: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_165, 0.001);  convert_element_type_165 = None
        sqrt_40: "f32[960][1]cuda:0" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
        reciprocal_40: "f32[960][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_142: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_322: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_142, -1);  mul_142 = None
        unsqueeze_323: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        mul_143: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
        unsqueeze_324: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_325: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_144: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_325);  mul_143 = unsqueeze_325 = None
        unsqueeze_326: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_327: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_111: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_327);  mul_144 = unsqueeze_327 = None
        convert_element_type_166: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None
        convert_element_type_167: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_166, torch.float32);  convert_element_type_166 = None
        add_112: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_167, 3)
        clamp_min_22: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_112, 0);  add_112 = None
        clamp_max_22: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
        mul_145: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_167, clamp_max_22);  convert_element_type_167 = clamp_max_22 = None
        div_22: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_145, 6);  mul_145 = None
        convert_element_type_168: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_6: "bf16[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_168, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_53: "bf16[32, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_6, arg230_1, arg231_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_6 = arg230_1 = arg231_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_17: "bf16[32, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_54: "bf16[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_17, arg232_1, arg233_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg232_1 = arg233_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_169: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_54, torch.float32);  convolution_54 = None
        add_113: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_169, 3);  convert_element_type_169 = None
        clamp_min_23: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_113, 0);  add_113 = None
        clamp_max_23: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
        div_23: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_23, 6);  clamp_max_23 = None
        convert_element_type_170: "bf16[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_146: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_170, convert_element_type_168);  convert_element_type_170 = convert_element_type_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_55: "bf16[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(mul_146, arg234_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_146 = arg234_1 = None
        convert_element_type_171: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg235_1, torch.float32);  arg235_1 = None
        unsqueeze_328: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_171, -1);  convert_element_type_171 = None
        unsqueeze_329: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        sub_41: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_329);  convolution_55 = unsqueeze_329 = None
        convert_element_type_172: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg236_1, torch.float32);  arg236_1 = None
        add_114: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_172, 0.001);  convert_element_type_172 = None
        sqrt_41: "f32[160][1]cuda:0" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_41: "f32[160][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_147: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_330: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
        unsqueeze_331: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        mul_148: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
        unsqueeze_332: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_333: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_149: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_333);  mul_148 = unsqueeze_333 = None
        unsqueeze_334: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_335: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_115: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_335);  mul_149 = unsqueeze_335 = None
        convert_element_type_173: "bf16[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16);  add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_116: "bf16[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_173, convert_element_type_158);  convert_element_type_173 = convert_element_type_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_56: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(add_116, arg239_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg239_1 = None
        convert_element_type_174: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg240_1, torch.float32);  arg240_1 = None
        unsqueeze_336: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_174, -1);  convert_element_type_174 = None
        unsqueeze_337: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        sub_42: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_337);  convolution_56 = unsqueeze_337 = None
        convert_element_type_175: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg241_1, torch.float32);  arg241_1 = None
        add_117: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_175, 0.001);  convert_element_type_175 = None
        sqrt_42: "f32[960][1]cuda:0" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
        reciprocal_42: "f32[960][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_150: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_338: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
        unsqueeze_339: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        mul_151: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
        unsqueeze_340: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_341: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_152: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_341);  mul_151 = unsqueeze_341 = None
        unsqueeze_342: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
        unsqueeze_343: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_118: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_343);  mul_152 = unsqueeze_343 = None
        convert_element_type_176: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_118, torch.bfloat16);  add_118 = None
        convert_element_type_177: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_176, torch.float32);  convert_element_type_176 = None
        add_119: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_177, 3)
        clamp_min_24: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_119, 0);  add_119 = None
        clamp_max_24: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
        mul_153: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_177, clamp_max_24);  convert_element_type_177 = clamp_max_24 = None
        div_24: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_153, 6);  mul_153 = None
        convert_element_type_178: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None
        convolution_57: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_178, arg244_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960);  convert_element_type_178 = arg244_1 = None
        convert_element_type_179: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg245_1, torch.float32);  arg245_1 = None
        unsqueeze_344: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_179, -1);  convert_element_type_179 = None
        unsqueeze_345: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        sub_43: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_345);  convolution_57 = unsqueeze_345 = None
        convert_element_type_180: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg246_1, torch.float32);  arg246_1 = None
        add_120: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_180, 0.001);  convert_element_type_180 = None
        sqrt_43: "f32[960][1]cuda:0" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_43: "f32[960][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_154: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_346: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_154, -1);  mul_154 = None
        unsqueeze_347: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        mul_155: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
        unsqueeze_348: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_349: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_156: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_349);  mul_155 = unsqueeze_349 = None
        unsqueeze_350: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_351: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_121: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_351);  mul_156 = unsqueeze_351 = None
        convert_element_type_181: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None
        convert_element_type_182: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_181, torch.float32);  convert_element_type_181 = None
        add_122: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_182, 3)
        clamp_min_25: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_122, 0);  add_122 = None
        clamp_max_25: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
        mul_157: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_182, clamp_max_25);  convert_element_type_182 = clamp_max_25 = None
        div_25: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_157, 6);  mul_157 = None
        convert_element_type_183: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        mean_7: "bf16[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_183, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        convolution_58: "bf16[32, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(mean_7, arg249_1, arg250_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg249_1 = arg250_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        relu_18: "bf16[32, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_58);  convolution_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        convolution_59: "bf16[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_18, arg251_1, arg252_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_18 = arg251_1 = arg252_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_184: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32);  convolution_59 = None
        add_123: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_184, 3);  convert_element_type_184 = None
        clamp_min_26: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_123, 0);  add_123 = None
        clamp_max_26: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
        div_26: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_26, 6);  clamp_max_26 = None
        convert_element_type_185: "bf16[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_158: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_185, convert_element_type_183);  convert_element_type_185 = convert_element_type_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convolution_60: "bf16[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(mul_158, arg253_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_158 = arg253_1 = None
        convert_element_type_186: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg254_1, torch.float32);  arg254_1 = None
        unsqueeze_352: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_186, -1);  convert_element_type_186 = None
        unsqueeze_353: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        sub_44: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_353);  convolution_60 = unsqueeze_353 = None
        convert_element_type_187: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg255_1, torch.float32);  arg255_1 = None
        add_124: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_187, 0.001);  convert_element_type_187 = None
        sqrt_44: "f32[160][1]cuda:0" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
        reciprocal_44: "f32[160][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_159: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_354: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
        unsqueeze_355: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        mul_160: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
        unsqueeze_356: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_357: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_161: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_357);  mul_160 = unsqueeze_357 = None
        unsqueeze_358: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_359: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_125: "f32[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_359);  mul_161 = unsqueeze_359 = None
        convert_element_type_188: "bf16[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:114 in forward, code: result += input
        add_126: "bf16[32, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_188, add_116);  convert_element_type_188 = add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:211 in _forward_impl, code: x = self.features(x)
        convolution_61: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(add_126, arg258_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_126 = arg258_1 = None
        convert_element_type_189: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg259_1, torch.float32);  arg259_1 = None
        unsqueeze_360: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_189, -1);  convert_element_type_189 = None
        unsqueeze_361: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        sub_45: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_361);  convolution_61 = unsqueeze_361 = None
        convert_element_type_190: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg260_1, torch.float32);  arg260_1 = None
        add_127: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_190, 0.001);  convert_element_type_190 = None
        sqrt_45: "f32[960][1]cuda:0" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
        reciprocal_45: "f32[960][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_162: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_362: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
        unsqueeze_363: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        mul_163: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
        unsqueeze_364: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_365: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_164: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_365);  mul_163 = unsqueeze_365 = None
        unsqueeze_366: "bf16[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_367: "bf16[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_128: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_367);  mul_164 = unsqueeze_367 = None
        convert_element_type_191: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_128, torch.bfloat16);  add_128 = None
        convert_element_type_192: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_191, torch.float32);  convert_element_type_191 = None
        add_129: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_192, 3)
        clamp_min_27: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_129, 0);  add_129 = None
        clamp_max_27: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
        mul_165: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_192, clamp_max_27);  convert_element_type_192 = clamp_max_27 = None
        div_27: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_165, 6);  mul_165 = None
        convert_element_type_193: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:213 in _forward_impl, code: x = self.avgpool(x)
        mean_8: "bf16[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_193, [-1, -2], True);  convert_element_type_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:214 in _forward_impl, code: x = torch.flatten(x, 1)
        view: "bf16[32, 960][960, 1]cuda:0" = torch.ops.aten.reshape.default(mean_8, [32, 960]);  mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:216 in _forward_impl, code: x = self.classifier(x)
        permute: "bf16[960, 1280][1, 960]cuda:0" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm: "bf16[32, 1280][1280, 1]cuda:0" = torch.ops.aten.addmm.default(arg264_1, view, permute);  arg264_1 = view = permute = None
        convert_element_type_197: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        add_130: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_197, 3)
        clamp_min_28: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.aten.clamp_min.default(add_130, 0);  add_130 = None
        clamp_max_28: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
        mul_166: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_197, clamp_max_28);  convert_element_type_197 = clamp_max_28 = None
        div_28: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_166, 6);  mul_166 = None
        convert_element_type_198: "bf16[32, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None
        permute_1: "bf16[1280, 1000][1, 1280]cuda:0" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_1: "bf16[32, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg266_1, convert_element_type_198, permute_1);  arg266_1 = convert_element_type_198 = permute_1 = None
        return (addmm_1,)
