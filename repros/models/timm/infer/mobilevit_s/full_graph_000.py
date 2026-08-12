class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[16, 3, 3, 3][27, 1, 9, 3]cuda:0", arg1_1: "bf16[128, 3, 256, 256][196608, 1, 768, 3]cuda:0", arg2_1: "bf16[16][1]cuda:0", arg3_1: "bf16[16][1]cuda:0", arg4_1: "bf16[16][1]cuda:0", arg5_1: "bf16[16][1]cuda:0", arg6_1: "bf16[64, 16, 1, 1][16, 1, 16, 16]cuda:0", arg7_1: "bf16[64][1]cuda:0", arg8_1: "bf16[64][1]cuda:0", arg9_1: "bf16[64][1]cuda:0", arg10_1: "bf16[64][1]cuda:0", arg11_1: "bf16[64, 1, 3, 3][9, 1, 3, 1]cuda:0", arg12_1: "bf16[64][1]cuda:0", arg13_1: "bf16[64][1]cuda:0", arg14_1: "bf16[64][1]cuda:0", arg15_1: "bf16[64][1]cuda:0", arg16_1: "bf16[32, 64, 1, 1][64, 1, 64, 64]cuda:0", arg17_1: "bf16[32][1]cuda:0", arg18_1: "bf16[32][1]cuda:0", arg19_1: "bf16[32][1]cuda:0", arg20_1: "bf16[32][1]cuda:0", arg21_1: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0", arg22_1: "bf16[128][1]cuda:0", arg23_1: "bf16[128][1]cuda:0", arg24_1: "bf16[128][1]cuda:0", arg25_1: "bf16[128][1]cuda:0", arg26_1: "bf16[128, 1, 3, 3][9, 1, 3, 1]cuda:0", arg27_1: "bf16[128][1]cuda:0", arg28_1: "bf16[128][1]cuda:0", arg29_1: "bf16[128][1]cuda:0", arg30_1: "bf16[128][1]cuda:0", arg31_1: "bf16[64, 128, 1, 1][128, 1, 128, 128]cuda:0", arg32_1: "bf16[64][1]cuda:0", arg33_1: "bf16[64][1]cuda:0", arg34_1: "bf16[64][1]cuda:0", arg35_1: "bf16[64][1]cuda:0", arg36_1: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", arg37_1: "bf16[256][1]cuda:0", arg38_1: "bf16[256][1]cuda:0", arg39_1: "bf16[256][1]cuda:0", arg40_1: "bf16[256][1]cuda:0", arg41_1: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0", arg42_1: "bf16[256][1]cuda:0", arg43_1: "bf16[256][1]cuda:0", arg44_1: "bf16[256][1]cuda:0", arg45_1: "bf16[256][1]cuda:0", arg46_1: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0", arg47_1: "bf16[64][1]cuda:0", arg48_1: "bf16[64][1]cuda:0", arg49_1: "bf16[64][1]cuda:0", arg50_1: "bf16[64][1]cuda:0", arg51_1: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", arg52_1: "bf16[256][1]cuda:0", arg53_1: "bf16[256][1]cuda:0", arg54_1: "bf16[256][1]cuda:0", arg55_1: "bf16[256][1]cuda:0", arg56_1: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0", arg57_1: "bf16[256][1]cuda:0", arg58_1: "bf16[256][1]cuda:0", arg59_1: "bf16[256][1]cuda:0", arg60_1: "bf16[256][1]cuda:0", arg61_1: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0", arg62_1: "bf16[64][1]cuda:0", arg63_1: "bf16[64][1]cuda:0", arg64_1: "bf16[64][1]cuda:0", arg65_1: "bf16[64][1]cuda:0", arg66_1: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", arg67_1: "bf16[256][1]cuda:0", arg68_1: "bf16[256][1]cuda:0", arg69_1: "bf16[256][1]cuda:0", arg70_1: "bf16[256][1]cuda:0", arg71_1: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0", arg72_1: "bf16[256][1]cuda:0", arg73_1: "bf16[256][1]cuda:0", arg74_1: "bf16[256][1]cuda:0", arg75_1: "bf16[256][1]cuda:0", arg76_1: "bf16[96, 256, 1, 1][256, 1, 256, 256]cuda:0", arg77_1: "bf16[96][1]cuda:0", arg78_1: "bf16[96][1]cuda:0", arg79_1: "bf16[96][1]cuda:0", arg80_1: "bf16[96][1]cuda:0", arg81_1: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0", arg82_1: "bf16[96][1]cuda:0", arg83_1: "bf16[96][1]cuda:0", arg84_1: "bf16[96][1]cuda:0", arg85_1: "bf16[96][1]cuda:0", arg86_1: "bf16[144, 96, 1, 1][96, 1, 96, 96]cuda:0", arg87_1: "bf16[144][1]cuda:0", arg88_1: "bf16[144][1]cuda:0", arg89_1: "bf16[432, 144][144, 1]cuda:0", arg90_1: "bf16[432][1]cuda:0", arg91_1: "bf16[144, 144][144, 1]cuda:0", arg92_1: "bf16[144][1]cuda:0", arg93_1: "bf16[144][1]cuda:0", arg94_1: "bf16[144][1]cuda:0", arg95_1: "bf16[288, 144][144, 1]cuda:0", arg96_1: "bf16[288][1]cuda:0", arg97_1: "bf16[144, 288][288, 1]cuda:0", arg98_1: "bf16[144][1]cuda:0", arg99_1: "bf16[144][1]cuda:0", arg100_1: "bf16[144][1]cuda:0", arg101_1: "bf16[432, 144][144, 1]cuda:0", arg102_1: "bf16[432][1]cuda:0", arg103_1: "bf16[144, 144][144, 1]cuda:0", arg104_1: "bf16[144][1]cuda:0", arg105_1: "bf16[144][1]cuda:0", arg106_1: "bf16[144][1]cuda:0", arg107_1: "bf16[288, 144][144, 1]cuda:0", arg108_1: "bf16[288][1]cuda:0", arg109_1: "bf16[144, 288][288, 1]cuda:0", arg110_1: "bf16[144][1]cuda:0", arg111_1: "bf16[144][1]cuda:0", arg112_1: "bf16[144][1]cuda:0", arg113_1: "bf16[96, 144, 1, 1][144, 1, 144, 144]cuda:0", arg114_1: "bf16[96][1]cuda:0", arg115_1: "bf16[96][1]cuda:0", arg116_1: "bf16[96][1]cuda:0", arg117_1: "bf16[96][1]cuda:0", arg118_1: "bf16[96, 192, 3, 3][1728, 1, 576, 192]cuda:0", arg119_1: "bf16[96][1]cuda:0", arg120_1: "bf16[96][1]cuda:0", arg121_1: "bf16[96][1]cuda:0", arg122_1: "bf16[96][1]cuda:0", arg123_1: "bf16[384, 96, 1, 1][96, 1, 96, 96]cuda:0", arg124_1: "bf16[384][1]cuda:0", arg125_1: "bf16[384][1]cuda:0", arg126_1: "bf16[384][1]cuda:0", arg127_1: "bf16[384][1]cuda:0", arg128_1: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0", arg129_1: "bf16[384][1]cuda:0", arg130_1: "bf16[384][1]cuda:0", arg131_1: "bf16[384][1]cuda:0", arg132_1: "bf16[384][1]cuda:0", arg133_1: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0", arg134_1: "bf16[128][1]cuda:0", arg135_1: "bf16[128][1]cuda:0", arg136_1: "bf16[128][1]cuda:0", arg137_1: "bf16[128][1]cuda:0", arg138_1: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", arg139_1: "bf16[128][1]cuda:0", arg140_1: "bf16[128][1]cuda:0", arg141_1: "bf16[128][1]cuda:0", arg142_1: "bf16[128][1]cuda:0", arg143_1: "bf16[192, 128, 1, 1][128, 1, 128, 128]cuda:0", arg144_1: "bf16[192][1]cuda:0", arg145_1: "bf16[192][1]cuda:0", arg146_1: "bf16[576, 192][192, 1]cuda:0", arg147_1: "bf16[576][1]cuda:0", arg148_1: "bf16[192, 192][192, 1]cuda:0", arg149_1: "bf16[192][1]cuda:0", arg150_1: "bf16[192][1]cuda:0", arg151_1: "bf16[192][1]cuda:0", arg152_1: "bf16[384, 192][192, 1]cuda:0", arg153_1: "bf16[384][1]cuda:0", arg154_1: "bf16[192, 384][384, 1]cuda:0", arg155_1: "bf16[192][1]cuda:0", arg156_1: "bf16[192][1]cuda:0", arg157_1: "bf16[192][1]cuda:0", arg158_1: "bf16[576, 192][192, 1]cuda:0", arg159_1: "bf16[576][1]cuda:0", arg160_1: "bf16[192, 192][192, 1]cuda:0", arg161_1: "bf16[192][1]cuda:0", arg162_1: "bf16[192][1]cuda:0", arg163_1: "bf16[192][1]cuda:0", arg164_1: "bf16[384, 192][192, 1]cuda:0", arg165_1: "bf16[384][1]cuda:0", arg166_1: "bf16[192, 384][384, 1]cuda:0", arg167_1: "bf16[192][1]cuda:0", arg168_1: "bf16[192][1]cuda:0", arg169_1: "bf16[192][1]cuda:0", arg170_1: "bf16[576, 192][192, 1]cuda:0", arg171_1: "bf16[576][1]cuda:0", arg172_1: "bf16[192, 192][192, 1]cuda:0", arg173_1: "bf16[192][1]cuda:0", arg174_1: "bf16[192][1]cuda:0", arg175_1: "bf16[192][1]cuda:0", arg176_1: "bf16[384, 192][192, 1]cuda:0", arg177_1: "bf16[384][1]cuda:0", arg178_1: "bf16[192, 384][384, 1]cuda:0", arg179_1: "bf16[192][1]cuda:0", arg180_1: "bf16[192][1]cuda:0", arg181_1: "bf16[192][1]cuda:0", arg182_1: "bf16[576, 192][192, 1]cuda:0", arg183_1: "bf16[576][1]cuda:0", arg184_1: "bf16[192, 192][192, 1]cuda:0", arg185_1: "bf16[192][1]cuda:0", arg186_1: "bf16[192][1]cuda:0", arg187_1: "bf16[192][1]cuda:0", arg188_1: "bf16[384, 192][192, 1]cuda:0", arg189_1: "bf16[384][1]cuda:0", arg190_1: "bf16[192, 384][384, 1]cuda:0", arg191_1: "bf16[192][1]cuda:0", arg192_1: "bf16[192][1]cuda:0", arg193_1: "bf16[192][1]cuda:0", arg194_1: "bf16[128, 192, 1, 1][192, 1, 192, 192]cuda:0", arg195_1: "bf16[128][1]cuda:0", arg196_1: "bf16[128][1]cuda:0", arg197_1: "bf16[128][1]cuda:0", arg198_1: "bf16[128][1]cuda:0", arg199_1: "bf16[128, 256, 3, 3][2304, 1, 768, 256]cuda:0", arg200_1: "bf16[128][1]cuda:0", arg201_1: "bf16[128][1]cuda:0", arg202_1: "bf16[128][1]cuda:0", arg203_1: "bf16[128][1]cuda:0", arg204_1: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", arg205_1: "bf16[512][1]cuda:0", arg206_1: "bf16[512][1]cuda:0", arg207_1: "bf16[512][1]cuda:0", arg208_1: "bf16[512][1]cuda:0", arg209_1: "bf16[512, 1, 3, 3][9, 1, 3, 1]cuda:0", arg210_1: "bf16[512][1]cuda:0", arg211_1: "bf16[512][1]cuda:0", arg212_1: "bf16[512][1]cuda:0", arg213_1: "bf16[512][1]cuda:0", arg214_1: "bf16[160, 512, 1, 1][512, 1, 512, 512]cuda:0", arg215_1: "bf16[160][1]cuda:0", arg216_1: "bf16[160][1]cuda:0", arg217_1: "bf16[160][1]cuda:0", arg218_1: "bf16[160][1]cuda:0", arg219_1: "bf16[160, 160, 3, 3][1440, 1, 480, 160]cuda:0", arg220_1: "bf16[160][1]cuda:0", arg221_1: "bf16[160][1]cuda:0", arg222_1: "bf16[160][1]cuda:0", arg223_1: "bf16[160][1]cuda:0", arg224_1: "bf16[240, 160, 1, 1][160, 1, 160, 160]cuda:0", arg225_1: "bf16[240][1]cuda:0", arg226_1: "bf16[240][1]cuda:0", arg227_1: "bf16[720, 240][240, 1]cuda:0", arg228_1: "bf16[720][1]cuda:0", arg229_1: "bf16[240, 240][240, 1]cuda:0", arg230_1: "bf16[240][1]cuda:0", arg231_1: "bf16[240][1]cuda:0", arg232_1: "bf16[240][1]cuda:0", arg233_1: "bf16[480, 240][240, 1]cuda:0", arg234_1: "bf16[480][1]cuda:0", arg235_1: "bf16[240, 480][480, 1]cuda:0", arg236_1: "bf16[240][1]cuda:0", arg237_1: "bf16[240][1]cuda:0", arg238_1: "bf16[240][1]cuda:0", arg239_1: "bf16[720, 240][240, 1]cuda:0", arg240_1: "bf16[720][1]cuda:0", arg241_1: "bf16[240, 240][240, 1]cuda:0", arg242_1: "bf16[240][1]cuda:0", arg243_1: "bf16[240][1]cuda:0", arg244_1: "bf16[240][1]cuda:0", arg245_1: "bf16[480, 240][240, 1]cuda:0", arg246_1: "bf16[480][1]cuda:0", arg247_1: "bf16[240, 480][480, 1]cuda:0", arg248_1: "bf16[240][1]cuda:0", arg249_1: "bf16[240][1]cuda:0", arg250_1: "bf16[240][1]cuda:0", arg251_1: "bf16[720, 240][240, 1]cuda:0", arg252_1: "bf16[720][1]cuda:0", arg253_1: "bf16[240, 240][240, 1]cuda:0", arg254_1: "bf16[240][1]cuda:0", arg255_1: "bf16[240][1]cuda:0", arg256_1: "bf16[240][1]cuda:0", arg257_1: "bf16[480, 240][240, 1]cuda:0", arg258_1: "bf16[480][1]cuda:0", arg259_1: "bf16[240, 480][480, 1]cuda:0", arg260_1: "bf16[240][1]cuda:0", arg261_1: "bf16[240][1]cuda:0", arg262_1: "bf16[240][1]cuda:0", arg263_1: "bf16[160, 240, 1, 1][240, 1, 240, 240]cuda:0", arg264_1: "bf16[160][1]cuda:0", arg265_1: "bf16[160][1]cuda:0", arg266_1: "bf16[160][1]cuda:0", arg267_1: "bf16[160][1]cuda:0", arg268_1: "bf16[160, 320, 3, 3][2880, 1, 960, 320]cuda:0", arg269_1: "bf16[160][1]cuda:0", arg270_1: "bf16[160][1]cuda:0", arg271_1: "bf16[160][1]cuda:0", arg272_1: "bf16[160][1]cuda:0", arg273_1: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0", arg274_1: "bf16[640][1]cuda:0", arg275_1: "bf16[640][1]cuda:0", arg276_1: "bf16[640][1]cuda:0", arg277_1: "bf16[640][1]cuda:0", arg278_1: "bf16[1000, 640][640, 1]cuda:0", arg279_1: "bf16[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        unsqueeze: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        convert_element_type_1: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
        sqrt: "f32[16][1]cuda:0" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[16][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7: "bf16[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_3: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        neg: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.neg.default(convert_element_type_3)
        exp: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.exp.default(neg);  neg = None
        add_2: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_3, add_2);  convert_element_type_3 = add_2 = None
        convert_element_type_4: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_1: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_4, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_4 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_5: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        unsqueeze_8: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, -1);  convert_element_type_5 = None
        unsqueeze_9: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        convert_element_type_6: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        add_3: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_6, 1e-05);  convert_element_type_6 = None
        sqrt_1: "f32[64][1]cuda:0" = torch.ops.aten.sqrt.default(add_3);  add_3 = None
        reciprocal_1: "f32[64][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_4: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_13: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_15: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_4: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
        convert_element_type_7: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_8: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        neg_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.neg.default(convert_element_type_8)
        exp_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_5: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_8, add_5);  convert_element_type_8 = add_5 = None
        convert_element_type_9: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_2: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_9, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  convert_element_type_9 = arg11_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_10: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg12_1, torch.float32);  arg12_1 = None
        unsqueeze_16: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
        unsqueeze_17: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        convert_element_type_11: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        add_6: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
        sqrt_2: "f32[64][1]cuda:0" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
        reciprocal_2: "f32[64][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_6: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
        unsqueeze_19: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_7: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_21: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_8: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
        unsqueeze_22: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_23: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_7: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
        convert_element_type_12: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_13: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32);  convert_element_type_12 = None
        neg_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.neg.default(convert_element_type_13)
        exp_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_8: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_13, add_8);  convert_element_type_13 = add_8 = None
        convert_element_type_14: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_3: "bf16[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_14, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_14 = arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_15: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        unsqueeze_24: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_15, -1);  convert_element_type_15 = None
        unsqueeze_25: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_3: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        convert_element_type_16: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg18_1, torch.float32);  arg18_1 = None
        add_9: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_16, 1e-05);  convert_element_type_16 = None
        sqrt_3: "f32[32][1]cuda:0" = torch.ops.aten.sqrt.default(add_9);  add_9 = None
        reciprocal_3: "f32[32][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_9: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_27: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_10: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28: "bf16[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_29: "bf16[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_11: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
        unsqueeze_30: "bf16[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_31: "bf16[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_10: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
        convert_element_type_17: "bf16[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_4: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_17, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_17 = arg21_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_18: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg22_1, torch.float32);  arg22_1 = None
        unsqueeze_32: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
        unsqueeze_33: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        sub_4: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        convert_element_type_19: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg23_1, torch.float32);  arg23_1 = None
        add_11: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
        sqrt_4: "f32[128][1]cuda:0" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
        reciprocal_4: "f32[128][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_12: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_34: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
        unsqueeze_35: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_13: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
        unsqueeze_36: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_37: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_14: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
        unsqueeze_38: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_39: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_12: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
        convert_element_type_20: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_21: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        neg_3: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_21)
        exp_3: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_13: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_21, add_13);  convert_element_type_21 = add_13 = None
        convert_element_type_22: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_5: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_22, arg26_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 128);  convert_element_type_22 = arg26_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_23: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg27_1, torch.float32);  arg27_1 = None
        unsqueeze_40: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_23, -1);  convert_element_type_23 = None
        unsqueeze_41: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        sub_5: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        convert_element_type_24: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg28_1, torch.float32);  arg28_1 = None
        add_14: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_24, 1e-05);  convert_element_type_24 = None
        sqrt_5: "f32[128][1]cuda:0" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
        reciprocal_5: "f32[128][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_15: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_42: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
        unsqueeze_43: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_16: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
        unsqueeze_44: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_45: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_17: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
        unsqueeze_46: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_47: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_15: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
        convert_element_type_25: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_26: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_25, torch.float32);  convert_element_type_25 = None
        neg_4: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_26)
        exp_4: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_16: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_26, add_16);  convert_element_type_26 = add_16 = None
        convert_element_type_27: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_6: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_27, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_27 = arg31_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_28: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg32_1, torch.float32);  arg32_1 = None
        unsqueeze_48: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
        unsqueeze_49: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        sub_6: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        convert_element_type_29: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg33_1, torch.float32);  arg33_1 = None
        add_17: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
        sqrt_6: "f32[64][1]cuda:0" = torch.ops.aten.sqrt.default(add_17);  add_17 = None
        reciprocal_6: "f32[64][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_18: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_50: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
        unsqueeze_51: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_19: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
        unsqueeze_52: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_53: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_20: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
        unsqueeze_54: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_55: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_18: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
        convert_element_type_30: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_7: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_30, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_31: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg37_1, torch.float32);  arg37_1 = None
        unsqueeze_56: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_31, -1);  convert_element_type_31 = None
        unsqueeze_57: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        sub_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        convert_element_type_32: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg38_1, torch.float32);  arg38_1 = None
        add_19: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_32, 1e-05);  convert_element_type_32 = None
        sqrt_7: "f32[256][1]cuda:0" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
        reciprocal_7: "f32[256][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_21: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_58: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
        unsqueeze_59: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_22: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
        unsqueeze_60: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_61: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_23: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
        unsqueeze_62: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_63: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_20: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
        convert_element_type_33: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_34: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_33, torch.float32);  convert_element_type_33 = None
        neg_5: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_34)
        exp_5: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_21: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_34, add_21);  convert_element_type_34 = add_21 = None
        convert_element_type_35: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_8: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_35, arg41_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256);  convert_element_type_35 = arg41_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_36: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg42_1, torch.float32);  arg42_1 = None
        unsqueeze_64: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
        unsqueeze_65: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        sub_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
        convert_element_type_37: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg43_1, torch.float32);  arg43_1 = None
        add_22: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
        sqrt_8: "f32[256][1]cuda:0" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
        reciprocal_8: "f32[256][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_24: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_66: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
        unsqueeze_67: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_25: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
        unsqueeze_68: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_69: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_26: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
        unsqueeze_70: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_71: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_23: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
        convert_element_type_38: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_39: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_38, torch.float32);  convert_element_type_38 = None
        neg_6: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_39)
        exp_6: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_24: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_39, add_24);  convert_element_type_39 = add_24 = None
        convert_element_type_40: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_9: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_40, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_40 = arg46_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_41: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg47_1, torch.float32);  arg47_1 = None
        unsqueeze_72: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_41, -1);  convert_element_type_41 = None
        unsqueeze_73: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        sub_9: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
        convert_element_type_42: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg48_1, torch.float32);  arg48_1 = None
        add_25: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_42, 1e-05);  convert_element_type_42 = None
        sqrt_9: "f32[64][1]cuda:0" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
        reciprocal_9: "f32[64][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_27: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_74: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
        unsqueeze_75: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_28: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
        unsqueeze_76: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_77: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_29: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
        unsqueeze_78: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_79: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_26: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
        convert_element_type_43: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:501 in forward, code: x = x + self.shortcut(shortcut)
        add_27: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_43, convert_element_type_30);  convert_element_type_43 = convert_element_type_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_10: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.convolution.default(add_27, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg51_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_44: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg52_1, torch.float32);  arg52_1 = None
        unsqueeze_80: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
        unsqueeze_81: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        sub_10: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
        convert_element_type_45: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg53_1, torch.float32);  arg53_1 = None
        add_28: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
        sqrt_10: "f32[256][1]cuda:0" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        reciprocal_10: "f32[256][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_30: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_82: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
        unsqueeze_83: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_31: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
        unsqueeze_84: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_85: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_32: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
        unsqueeze_86: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_87: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_29: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
        convert_element_type_46: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_47: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_46, torch.float32);  convert_element_type_46 = None
        neg_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_47)
        exp_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_30: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_47, add_30);  convert_element_type_47 = add_30 = None
        convert_element_type_48: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_11: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_48, arg56_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256);  convert_element_type_48 = arg56_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_49: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg57_1, torch.float32);  arg57_1 = None
        unsqueeze_88: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_49, -1);  convert_element_type_49 = None
        unsqueeze_89: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        sub_11: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
        convert_element_type_50: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg58_1, torch.float32);  arg58_1 = None
        add_31: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_50, 1e-05);  convert_element_type_50 = None
        sqrt_11: "f32[256][1]cuda:0" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
        reciprocal_11: "f32[256][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_33: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_90: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
        unsqueeze_91: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_34: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
        unsqueeze_92: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_93: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_35: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
        unsqueeze_94: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_95: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_32: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
        convert_element_type_51: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_52: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_51, torch.float32);  convert_element_type_51 = None
        neg_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_52)
        exp_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_33: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_52, add_33);  convert_element_type_52 = add_33 = None
        convert_element_type_53: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_12: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_53, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_53 = arg61_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_54: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg62_1, torch.float32);  arg62_1 = None
        unsqueeze_96: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
        unsqueeze_97: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        sub_12: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
        convert_element_type_55: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg63_1, torch.float32);  arg63_1 = None
        add_34: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
        sqrt_12: "f32[64][1]cuda:0" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
        reciprocal_12: "f32[64][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_36: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_98: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
        unsqueeze_99: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_37: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
        unsqueeze_100: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_101: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_38: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
        unsqueeze_102: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_103: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_35: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
        convert_element_type_56: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:501 in forward, code: x = x + self.shortcut(shortcut)
        add_36: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_56, add_27);  convert_element_type_56 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_13: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.convolution.default(add_36, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_36 = arg66_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_57: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg67_1, torch.float32);  arg67_1 = None
        unsqueeze_104: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_57, -1);  convert_element_type_57 = None
        unsqueeze_105: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        sub_13: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
        convert_element_type_58: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg68_1, torch.float32);  arg68_1 = None
        add_37: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_58, 1e-05);  convert_element_type_58 = None
        sqrt_13: "f32[256][1]cuda:0" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
        reciprocal_13: "f32[256][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_39: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_106: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
        unsqueeze_107: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_40: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
        unsqueeze_108: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_109: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_41: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
        unsqueeze_110: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_111: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_38: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
        convert_element_type_59: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_60: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_59, torch.float32);  convert_element_type_59 = None
        neg_9: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_60)
        exp_9: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_39: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_60, add_39);  convert_element_type_60 = add_39 = None
        convert_element_type_61: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_14: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_61, arg71_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 256);  convert_element_type_61 = arg71_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_62: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg72_1, torch.float32);  arg72_1 = None
        unsqueeze_112: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
        unsqueeze_113: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        sub_14: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
        convert_element_type_63: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg73_1, torch.float32);  arg73_1 = None
        add_40: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
        sqrt_14: "f32[256][1]cuda:0" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
        reciprocal_14: "f32[256][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_42: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_114: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
        unsqueeze_115: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_43: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
        unsqueeze_116: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_117: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_44: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
        unsqueeze_118: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_119: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_41: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
        convert_element_type_64: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_65: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_64, torch.float32);  convert_element_type_64 = None
        neg_10: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_65)
        exp_10: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_42: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_65, add_42);  convert_element_type_65 = add_42 = None
        convert_element_type_66: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_15: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_66, arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_66 = arg76_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_67: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg77_1, torch.float32);  arg77_1 = None
        unsqueeze_120: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_67, -1);  convert_element_type_67 = None
        unsqueeze_121: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        sub_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
        convert_element_type_68: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg78_1, torch.float32);  arg78_1 = None
        add_43: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_68, 1e-05);  convert_element_type_68 = None
        sqrt_15: "f32[96][1]cuda:0" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
        reciprocal_15: "f32[96][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_45: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_122: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
        unsqueeze_123: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_46: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
        unsqueeze_124: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_125: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_47: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
        unsqueeze_126: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_127: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_44: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
        convert_element_type_69: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_16: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_69, arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg81_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_70: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg82_1, torch.float32);  arg82_1 = None
        unsqueeze_128: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
        unsqueeze_129: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        sub_16: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
        convert_element_type_71: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg83_1, torch.float32);  arg83_1 = None
        add_45: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
        sqrt_16: "f32[96][1]cuda:0" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
        reciprocal_16: "f32[96][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_48: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_130: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
        unsqueeze_131: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_49: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
        unsqueeze_132: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_133: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_50: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
        unsqueeze_134: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_135: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_46: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
        convert_element_type_72: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_73: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_72, torch.float32);  convert_element_type_72 = None
        neg_11: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.neg.default(convert_element_type_73)
        exp_11: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_47: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_73, add_47);  convert_element_type_73 = add_47 = None
        convert_element_type_74: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_17: "bf16[128, 144, 32, 32][147456, 1, 4608, 144]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_74, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_74 = arg86_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        clone: "bf16[128, 144, 32, 32][147456, 1024, 32, 1]cuda:0" = torch.ops.aten.clone.default(convolution_17, memory_format = torch.contiguous_format);  convolution_17 = None
        view: "bf16[294912, 2, 16, 2][64, 32, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [294912, 2, 16, 2]);  clone = None
        permute: "bf16[294912, 16, 2, 2][64, 2, 32, 1]cuda:0" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_1: "bf16[294912, 16, 2, 2][64, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[128, 144, 256, 4][147456, 1024, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [128, 144, 256, 4]);  clone_1 = None
        permute_1: "bf16[128, 4, 256, 144][147456, 1, 4, 1024]cuda:0" = torch.ops.aten.permute.default(view_1, [0, 3, 2, 1]);  view_1 = None
        clone_2: "bf16[128, 4, 256, 144][147456, 36864, 144, 1]cuda:0" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [512, 256, 144]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_75: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_75, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub_17: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_75, getitem_1);  convert_element_type_75 = getitem_1 = None
        add_48: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_51: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt);  sub_17 = rsqrt = None
        mul_52: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, arg87_1);  mul_51 = arg87_1 = None
        add_49: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_52, arg88_1);  mul_52 = arg88_1 = None
        convert_element_type_76: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_3: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_76, [131072, 144]);  convert_element_type_76 = None
        permute_2: "bf16[144, 432][1, 144]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm: "bf16[131072, 432][432, 1]cuda:0" = torch.ops.aten.addmm.default(arg90_1, view_3, permute_2);  arg90_1 = view_3 = permute_2 = None
        view_4: "bf16[512, 256, 432][110592, 432, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [512, 256, 432]);  addmm = None
        view_5: "bf16[512, 256, 3, 4, 36][110592, 432, 144, 36, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [512, 256, 3, 4, 36]);  view_4 = None
        permute_3: "bf16[3, 512, 4, 256, 36][144, 110592, 36, 432, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
        getitem_2: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind[0]
        getitem_3: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind[1]
        getitem_4: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_2, [0, 4], 0.0);  getitem_2 = None
        constant_pad_nd_1: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_3, [0, 4], 0.0);  getitem_3 = None
        constant_pad_nd_2: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_4, [0, 4], 0.0);  getitem_4 = None
        _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(constant_pad_nd, constant_pad_nd_1, constant_pad_nd_2, scale = 0.16666666666666666);  constant_pad_nd = constant_pad_nd_1 = constant_pad_nd_2 = None
        getitem_5: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = _scaled_dot_product_flash_attention[0];  _scaled_dot_product_flash_attention = None
        slice_1: "bf16[512, 4, 256, 36][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.slice.Tensor(getitem_5, -1, 0, 36);  getitem_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_4: "bf16[512, 256, 4, 36][40960, 40, 10240, 1]cuda:0" = torch.ops.aten.permute.default(slice_1, [0, 2, 1, 3]);  slice_1 = None
        clone_3: "bf16[512, 256, 4, 36][36864, 144, 36, 1]cuda:0" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_6: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [512, 256, 144]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_7: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [131072, 144]);  view_6 = None
        permute_5: "bf16[144, 144][1, 144]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_1: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.addmm.default(arg92_1, view_7, permute_5);  arg92_1 = view_7 = permute_5 = None
        view_8: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [512, 256, 144]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_50: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2, view_8);  view_2 = view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_83: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_83, [2], correction = 0, keepdim = True)
        getitem_14: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_1[0]
        getitem_15: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_18: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_83, getitem_15);  convert_element_type_83 = getitem_15 = None
        add_51: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_1: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_53: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_1);  sub_18 = rsqrt_1 = None
        mul_54: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, arg93_1);  mul_53 = arg93_1 = None
        add_52: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_54, arg94_1);  mul_54 = arg94_1 = None
        convert_element_type_84: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_9: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_84, [131072, 144]);  convert_element_type_84 = None
        permute_6: "bf16[144, 288][1, 144]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_2: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.addmm.default(arg96_1, view_9, permute_6);  arg96_1 = view_9 = permute_6 = None
        view_10: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [512, 256, 288]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_88: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        neg_12: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_88)
        exp_12: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_53: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_88, add_53);  convert_element_type_88 = add_53 = None
        convert_element_type_89: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_11: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_89, [131072, 288]);  convert_element_type_89 = None
        permute_7: "bf16[288, 144][1, 288]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_3: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.addmm.default(arg98_1, view_11, permute_7);  arg98_1 = view_11 = permute_7 = None
        view_12: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [512, 256, 144]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_54: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_50, view_12);  add_50 = view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_93: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_93, [2], correction = 0, keepdim = True)
        getitem_16: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_2[0]
        getitem_17: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_19: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_93, getitem_17);  convert_element_type_93 = getitem_17 = None
        add_55: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_2: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_55: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_2);  sub_19 = rsqrt_2 = None
        mul_56: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, arg99_1);  mul_55 = arg99_1 = None
        add_56: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_56, arg100_1);  mul_56 = arg100_1 = None
        convert_element_type_94: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_13: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_94, [131072, 144]);  convert_element_type_94 = None
        permute_8: "bf16[144, 432][1, 144]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_4: "bf16[131072, 432][432, 1]cuda:0" = torch.ops.aten.addmm.default(arg102_1, view_13, permute_8);  arg102_1 = view_13 = permute_8 = None
        view_14: "bf16[512, 256, 432][110592, 432, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [512, 256, 432]);  addmm_4 = None
        view_15: "bf16[512, 256, 3, 4, 36][110592, 432, 144, 36, 1]cuda:0" = torch.ops.aten.reshape.default(view_14, [512, 256, 3, 4, 36]);  view_14 = None
        permute_9: "bf16[3, 512, 4, 256, 36][144, 110592, 36, 432, 1]cuda:0" = torch.ops.aten.permute.default(view_15, [2, 0, 3, 1, 4]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
        getitem_18: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind_1[0]
        getitem_19: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind_1[1]
        getitem_20: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd_3: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_18, [0, 4], 0.0);  getitem_18 = None
        constant_pad_nd_4: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_19, [0, 4], 0.0);  getitem_19 = None
        constant_pad_nd_5: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_20, [0, 4], 0.0);  getitem_20 = None
        _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(constant_pad_nd_3, constant_pad_nd_4, constant_pad_nd_5, scale = 0.16666666666666666);  constant_pad_nd_3 = constant_pad_nd_4 = constant_pad_nd_5 = None
        getitem_21: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = _scaled_dot_product_flash_attention_1[0];  _scaled_dot_product_flash_attention_1 = None
        slice_2: "bf16[512, 4, 256, 36][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.slice.Tensor(getitem_21, -1, 0, 36);  getitem_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_10: "bf16[512, 256, 4, 36][40960, 40, 10240, 1]cuda:0" = torch.ops.aten.permute.default(slice_2, [0, 2, 1, 3]);  slice_2 = None
        clone_7: "bf16[512, 256, 4, 36][36864, 144, 36, 1]cuda:0" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        view_16: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [512, 256, 144]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_17: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(view_16, [131072, 144]);  view_16 = None
        permute_11: "bf16[144, 144][1, 144]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_5: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.addmm.default(arg104_1, view_17, permute_11);  arg104_1 = view_17 = permute_11 = None
        view_18: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [512, 256, 144]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_57: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_54, view_18);  add_54 = view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_101: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_101, [2], correction = 0, keepdim = True)
        getitem_30: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_3[0]
        getitem_31: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_20: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_101, getitem_31);  convert_element_type_101 = getitem_31 = None
        add_58: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_3: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_57: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_3);  sub_20 = rsqrt_3 = None
        mul_58: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg105_1);  mul_57 = arg105_1 = None
        add_59: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg106_1);  mul_58 = arg106_1 = None
        convert_element_type_102: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_19: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_102, [131072, 144]);  convert_element_type_102 = None
        permute_12: "bf16[144, 288][1, 144]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_6: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.addmm.default(arg108_1, view_19, permute_12);  arg108_1 = view_19 = permute_12 = None
        view_20: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [512, 256, 288]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_106: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_20, torch.float32);  view_20 = None
        neg_13: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_106)
        exp_13: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_60: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_106, add_60);  convert_element_type_106 = add_60 = None
        convert_element_type_107: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_21: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_107, [131072, 288]);  convert_element_type_107 = None
        permute_13: "bf16[288, 144][1, 288]cuda:0" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_7: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.addmm.default(arg110_1, view_21, permute_13);  arg110_1 = view_21 = permute_13 = None
        view_22: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [512, 256, 144]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_61: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_57, view_22);  add_57 = view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        convert_element_type_111: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.float32);  add_61 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_111, [2], correction = 0, keepdim = True)
        getitem_32: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_4[0]
        getitem_33: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_21: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_111, getitem_33);  convert_element_type_111 = getitem_33 = None
        add_62: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_4: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_59: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_4);  sub_21 = rsqrt_4 = None
        mul_60: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, arg111_1);  mul_59 = arg111_1 = None
        add_63: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, arg112_1);  mul_60 = arg112_1 = None
        convert_element_type_112: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_23: "bf16[128, 4, 256, 144][147456, 36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_112, [128, 4, 256, -1]);  convert_element_type_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_14: "bf16[128, 144, 256, 4][147456, 1, 144, 36864]cuda:0" = torch.ops.aten.permute.default(view_23, [0, 3, 2, 1]);  view_23 = None
        clone_11: "bf16[128, 144, 256, 4][147456, 1024, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_24: "bf16[294912, 16, 2, 2][64, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [294912, 16, 2, 2]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_15: "bf16[294912, 2, 16, 2][64, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        clone_12: "bf16[294912, 2, 16, 2][64, 32, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_25: "bf16[128, 144, 32, 32][147456, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [128, 144, 32, 32]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_18: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.convolution.default(view_25, arg113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_25 = arg113_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_113: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg114_1, torch.float32);  arg114_1 = None
        unsqueeze_136: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_113, -1);  convert_element_type_113 = None
        unsqueeze_137: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        sub_22: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_137);  convolution_18 = unsqueeze_137 = None
        convert_element_type_114: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg115_1, torch.float32);  arg115_1 = None
        add_64: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_114, 1e-05);  convert_element_type_114 = None
        sqrt_17: "f32[96][1]cuda:0" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
        reciprocal_17: "f32[96][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_61: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        unsqueeze_138: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_61, -1);  mul_61 = None
        unsqueeze_139: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        mul_62: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_139);  sub_22 = unsqueeze_139 = None
        unsqueeze_140: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_141: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_63: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, unsqueeze_141);  mul_62 = unsqueeze_141 = None
        unsqueeze_142: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_143: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_65: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_63, unsqueeze_143);  mul_63 = unsqueeze_143 = None
        convert_element_type_115: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_116: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_115, torch.float32);  convert_element_type_115 = None
        neg_14: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.neg.default(convert_element_type_116)
        exp_14: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_66: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_116, add_66);  convert_element_type_116 = add_66 = None
        convert_element_type_117: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat: "bf16[128, 192, 32, 32][196608, 1, 6144, 192]cuda:0" = torch.ops.aten.cat.default([convert_element_type_69, convert_element_type_117], 1);  convert_element_type_69 = convert_element_type_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_19: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.convolution.default(cat, arg118_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat = arg118_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_118: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg119_1, torch.float32);  arg119_1 = None
        unsqueeze_144: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_118, -1);  convert_element_type_118 = None
        unsqueeze_145: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        sub_23: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_145);  convolution_19 = unsqueeze_145 = None
        convert_element_type_119: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg120_1, torch.float32);  arg120_1 = None
        add_67: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_119, 1e-05);  convert_element_type_119 = None
        sqrt_18: "f32[96][1]cuda:0" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
        reciprocal_18: "f32[96][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_64: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        unsqueeze_146: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_64, -1);  mul_64 = None
        unsqueeze_147: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        mul_65: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_147);  sub_23 = unsqueeze_147 = None
        unsqueeze_148: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_149: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_66: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, unsqueeze_149);  mul_65 = unsqueeze_149 = None
        unsqueeze_150: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_151: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_68: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_66, unsqueeze_151);  mul_66 = unsqueeze_151 = None
        convert_element_type_120: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_121: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_120, torch.float32);  convert_element_type_120 = None
        neg_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.neg.default(convert_element_type_121)
        exp_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_69: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_121, add_69);  convert_element_type_121 = add_69 = None
        convert_element_type_122: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_20: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_122, arg123_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_122 = arg123_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_123: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg124_1, torch.float32);  arg124_1 = None
        unsqueeze_152: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_123, -1);  convert_element_type_123 = None
        unsqueeze_153: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        sub_24: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_153);  convolution_20 = unsqueeze_153 = None
        convert_element_type_124: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg125_1, torch.float32);  arg125_1 = None
        add_70: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_124, 1e-05);  convert_element_type_124 = None
        sqrt_19: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
        reciprocal_19: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_67: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        unsqueeze_154: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_67, -1);  mul_67 = None
        unsqueeze_155: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        mul_68: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_155);  sub_24 = unsqueeze_155 = None
        unsqueeze_156: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_157: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_69: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, unsqueeze_157);  mul_68 = unsqueeze_157 = None
        unsqueeze_158: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_159: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_71: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_159);  mul_69 = unsqueeze_159 = None
        convert_element_type_125: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_126: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_125, torch.float32);  convert_element_type_125 = None
        neg_16: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_126)
        exp_16: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_72: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_126, add_72);  convert_element_type_126 = add_72 = None
        convert_element_type_127: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_21: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_127, arg128_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 384);  convert_element_type_127 = arg128_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_128: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg129_1, torch.float32);  arg129_1 = None
        unsqueeze_160: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_128, -1);  convert_element_type_128 = None
        unsqueeze_161: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        sub_25: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_161);  convolution_21 = unsqueeze_161 = None
        convert_element_type_129: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg130_1, torch.float32);  arg130_1 = None
        add_73: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_129, 1e-05);  convert_element_type_129 = None
        sqrt_20: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
        reciprocal_20: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        mul_70: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        unsqueeze_162: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_70, -1);  mul_70 = None
        unsqueeze_163: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        mul_71: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_163);  sub_25 = unsqueeze_163 = None
        unsqueeze_164: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_165: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_72: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_165);  mul_71 = unsqueeze_165 = None
        unsqueeze_166: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_167: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_74: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_167);  mul_72 = unsqueeze_167 = None
        convert_element_type_130: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_131: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_130, torch.float32);  convert_element_type_130 = None
        neg_17: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_131)
        exp_17: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_75: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        div_17: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_131, add_75);  convert_element_type_131 = add_75 = None
        convert_element_type_132: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_22: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_132, arg133_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_132 = arg133_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_133: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg134_1, torch.float32);  arg134_1 = None
        unsqueeze_168: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_133, -1);  convert_element_type_133 = None
        unsqueeze_169: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        sub_26: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_169);  convolution_22 = unsqueeze_169 = None
        convert_element_type_134: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg135_1, torch.float32);  arg135_1 = None
        add_76: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_134, 1e-05);  convert_element_type_134 = None
        sqrt_21: "f32[128][1]cuda:0" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
        reciprocal_21: "f32[128][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        mul_73: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        unsqueeze_170: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_73, -1);  mul_73 = None
        unsqueeze_171: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        mul_74: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_171);  sub_26 = unsqueeze_171 = None
        unsqueeze_172: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_173: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_75: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, unsqueeze_173);  mul_74 = unsqueeze_173 = None
        unsqueeze_174: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_175: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_77: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_75, unsqueeze_175);  mul_75 = unsqueeze_175 = None
        convert_element_type_135: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_23: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_135, arg138_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg138_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_136: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg139_1, torch.float32);  arg139_1 = None
        unsqueeze_176: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_136, -1);  convert_element_type_136 = None
        unsqueeze_177: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        sub_27: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_177);  convolution_23 = unsqueeze_177 = None
        convert_element_type_137: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float32);  arg140_1 = None
        add_78: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_137, 1e-05);  convert_element_type_137 = None
        sqrt_22: "f32[128][1]cuda:0" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
        reciprocal_22: "f32[128][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        mul_76: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        unsqueeze_178: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_76, -1);  mul_76 = None
        unsqueeze_179: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        mul_77: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_179);  sub_27 = unsqueeze_179 = None
        unsqueeze_180: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_181: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_78: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_181);  mul_77 = unsqueeze_181 = None
        unsqueeze_182: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_183: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_79: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_183);  mul_78 = unsqueeze_183 = None
        convert_element_type_138: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_139: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_138, torch.float32);  convert_element_type_138 = None
        neg_18: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_139)
        exp_18: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_80: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        div_18: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_139, add_80);  convert_element_type_139 = add_80 = None
        convert_element_type_140: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_24: "bf16[128, 192, 16, 16][49152, 1, 3072, 192]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_140, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_140 = arg143_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        clone_13: "bf16[128, 192, 16, 16][49152, 256, 16, 1]cuda:0" = torch.ops.aten.clone.default(convolution_24, memory_format = torch.contiguous_format);  convolution_24 = None
        view_26: "bf16[196608, 2, 8, 2][32, 16, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [196608, 2, 8, 2]);  clone_13 = None
        permute_16: "bf16[196608, 8, 2, 2][32, 2, 16, 1]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_14: "bf16[196608, 8, 2, 2][32, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_27: "bf16[128, 192, 64, 4][49152, 256, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [128, 192, 64, 4]);  clone_14 = None
        permute_17: "bf16[128, 4, 64, 192][49152, 1, 4, 256]cuda:0" = torch.ops.aten.permute.default(view_27, [0, 3, 2, 1]);  view_27 = None
        clone_15: "bf16[128, 4, 64, 192][49152, 12288, 192, 1]cuda:0" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_28: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [512, 64, 192]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_141: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_28, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_141, [2], correction = 0, keepdim = True)
        getitem_34: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_5[0]
        getitem_35: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_28: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_141, getitem_35);  convert_element_type_141 = getitem_35 = None
        add_81: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_5: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        mul_79: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_5);  sub_28 = rsqrt_5 = None
        mul_80: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, arg144_1);  mul_79 = arg144_1 = None
        add_82: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_80, arg145_1);  mul_80 = arg145_1 = None
        convert_element_type_142: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_82, torch.bfloat16);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_29: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_142, [32768, 192]);  convert_element_type_142 = None
        permute_18: "bf16[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_8: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.addmm.default(arg147_1, view_29, permute_18);  arg147_1 = view_29 = permute_18 = None
        view_30: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [512, 64, 576]);  addmm_8 = None
        view_31: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [512, 64, 3, 4, 48]);  view_30 = None
        permute_19: "bf16[3, 512, 4, 64, 48][192, 36864, 48, 576, 1]cuda:0" = torch.ops.aten.permute.default(view_31, [2, 0, 3, 1, 4]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_2 = torch.ops.aten.unbind.int(permute_19);  permute_19 = None
        getitem_36: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_2[0]
        getitem_37: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_2[1]
        getitem_38: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_36, getitem_37, getitem_38, None, False);  getitem_36 = getitem_37 = getitem_38 = None
        getitem_39: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_20: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_39, [0, 2, 1, 3]);  getitem_39 = None
        view_32: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_20, [512, 64, 192]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_33: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_32, [32768, 192]);  view_32 = None
        permute_21: "bf16[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_9: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(arg149_1, view_33, permute_21);  arg149_1 = view_33 = permute_21 = None
        view_34: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [512, 64, 192]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_83: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_28, view_34);  view_28 = view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_149: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_149, [2], correction = 0, keepdim = True)
        getitem_48: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_6[0]
        getitem_49: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_29: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_149, getitem_49);  convert_element_type_149 = getitem_49 = None
        add_84: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_6: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_81: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_6);  sub_29 = rsqrt_6 = None
        mul_82: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, arg150_1);  mul_81 = arg150_1 = None
        add_85: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, arg151_1);  mul_82 = arg151_1 = None
        convert_element_type_150: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_35: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_150, [32768, 192]);  convert_element_type_150 = None
        permute_22: "bf16[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_10: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg153_1, view_35, permute_22);  arg153_1 = view_35 = permute_22 = None
        view_36: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [512, 64, 384]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_154: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_36, torch.float32);  view_36 = None
        neg_19: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_154)
        exp_19: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_86: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_154, add_86);  convert_element_type_154 = add_86 = None
        convert_element_type_155: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_37: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_155, [32768, 384]);  convert_element_type_155 = None
        permute_23: "bf16[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_11: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(arg155_1, view_37, permute_23);  arg155_1 = view_37 = permute_23 = None
        view_38: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [512, 64, 192]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_87: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_83, view_38);  add_83 = view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_159: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_159, [2], correction = 0, keepdim = True)
        getitem_50: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_7[0]
        getitem_51: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_30: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_159, getitem_51);  convert_element_type_159 = getitem_51 = None
        add_88: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_83: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_7);  sub_30 = rsqrt_7 = None
        mul_84: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, arg156_1);  mul_83 = arg156_1 = None
        add_89: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_84, arg157_1);  mul_84 = arg157_1 = None
        convert_element_type_160: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_39: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_160, [32768, 192]);  convert_element_type_160 = None
        permute_24: "bf16[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_12: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.addmm.default(arg159_1, view_39, permute_24);  arg159_1 = view_39 = permute_24 = None
        view_40: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [512, 64, 576]);  addmm_12 = None
        view_41: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_40, [512, 64, 3, 4, 48]);  view_40 = None
        permute_25: "bf16[3, 512, 4, 64, 48][192, 36864, 48, 576, 1]cuda:0" = torch.ops.aten.permute.default(view_41, [2, 0, 3, 1, 4]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_3 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
        getitem_52: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_3[0]
        getitem_53: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_3[1]
        getitem_54: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_52, getitem_53, getitem_54, None, False);  getitem_52 = getitem_53 = getitem_54 = None
        getitem_55: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_26: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
        view_42: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_26, [512, 64, 192]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_43: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [32768, 192]);  view_42 = None
        permute_27: "bf16[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_13: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(arg161_1, view_43, permute_27);  arg161_1 = view_43 = permute_27 = None
        view_44: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [512, 64, 192]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_90: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_87, view_44);  add_87 = view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_167: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_167, [2], correction = 0, keepdim = True)
        getitem_64: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_8[0]
        getitem_65: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_31: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_167, getitem_65);  convert_element_type_167 = getitem_65 = None
        add_91: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_8: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        mul_85: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_8);  sub_31 = rsqrt_8 = None
        mul_86: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, arg162_1);  mul_85 = arg162_1 = None
        add_92: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_86, arg163_1);  mul_86 = arg163_1 = None
        convert_element_type_168: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_92, torch.bfloat16);  add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_45: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_168, [32768, 192]);  convert_element_type_168 = None
        permute_28: "bf16[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_14: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg165_1, view_45, permute_28);  arg165_1 = view_45 = permute_28 = None
        view_46: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [512, 64, 384]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_172: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        neg_20: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_172)
        exp_20: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_93: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        div_20: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_172, add_93);  convert_element_type_172 = add_93 = None
        convert_element_type_173: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_47: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_173, [32768, 384]);  convert_element_type_173 = None
        permute_29: "bf16[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_15: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(arg167_1, view_47, permute_29);  arg167_1 = view_47 = permute_29 = None
        view_48: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [512, 64, 192]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_94: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_90, view_48);  add_90 = view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_177: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_177, [2], correction = 0, keepdim = True)
        getitem_66: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_9[0]
        getitem_67: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_32: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_177, getitem_67);  convert_element_type_177 = getitem_67 = None
        add_95: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_9: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_87: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_9);  sub_32 = rsqrt_9 = None
        mul_88: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, arg168_1);  mul_87 = arg168_1 = None
        add_96: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, arg169_1);  mul_88 = arg169_1 = None
        convert_element_type_178: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16);  add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_49: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_178, [32768, 192]);  convert_element_type_178 = None
        permute_30: "bf16[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_16: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.addmm.default(arg171_1, view_49, permute_30);  arg171_1 = view_49 = permute_30 = None
        view_50: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [512, 64, 576]);  addmm_16 = None
        view_51: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_50, [512, 64, 3, 4, 48]);  view_50 = None
        permute_31: "bf16[3, 512, 4, 64, 48][192, 36864, 48, 576, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [2, 0, 3, 1, 4]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_4 = torch.ops.aten.unbind.int(permute_31);  permute_31 = None
        getitem_68: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_4[0]
        getitem_69: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_4[1]
        getitem_70: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_68, getitem_69, getitem_70, None, False);  getitem_68 = getitem_69 = getitem_70 = None
        getitem_71: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_32: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        view_52: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_32, [512, 64, 192]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_53: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [32768, 192]);  view_52 = None
        permute_33: "bf16[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_17: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(arg173_1, view_53, permute_33);  arg173_1 = view_53 = permute_33 = None
        view_54: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [512, 64, 192]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_97: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_94, view_54);  add_94 = view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_185: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_185, [2], correction = 0, keepdim = True)
        getitem_80: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_10[0]
        getitem_81: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_33: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_185, getitem_81);  convert_element_type_185 = getitem_81 = None
        add_98: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_10: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_89: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_10);  sub_33 = rsqrt_10 = None
        mul_90: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, arg174_1);  mul_89 = arg174_1 = None
        add_99: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, arg175_1);  mul_90 = arg175_1 = None
        convert_element_type_186: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_55: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_186, [32768, 192]);  convert_element_type_186 = None
        permute_34: "bf16[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_18: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg177_1, view_55, permute_34);  arg177_1 = view_55 = permute_34 = None
        view_56: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [512, 64, 384]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_190: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_56, torch.float32);  view_56 = None
        neg_21: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_190)
        exp_21: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_100: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        div_21: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_190, add_100);  convert_element_type_190 = add_100 = None
        convert_element_type_191: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_57: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_191, [32768, 384]);  convert_element_type_191 = None
        permute_35: "bf16[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_19: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(arg179_1, view_57, permute_35);  arg179_1 = view_57 = permute_35 = None
        view_58: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [512, 64, 192]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_101: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_97, view_58);  add_97 = view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_195: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_195, [2], correction = 0, keepdim = True)
        getitem_82: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_11[0]
        getitem_83: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_34: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_195, getitem_83);  convert_element_type_195 = getitem_83 = None
        add_102: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_11: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_91: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_11);  sub_34 = rsqrt_11 = None
        mul_92: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, arg180_1);  mul_91 = arg180_1 = None
        add_103: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, arg181_1);  mul_92 = arg181_1 = None
        convert_element_type_196: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_59: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_196, [32768, 192]);  convert_element_type_196 = None
        permute_36: "bf16[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_20: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.addmm.default(arg183_1, view_59, permute_36);  arg183_1 = view_59 = permute_36 = None
        view_60: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [512, 64, 576]);  addmm_20 = None
        view_61: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_60, [512, 64, 3, 4, 48]);  view_60 = None
        permute_37: "bf16[3, 512, 4, 64, 48][192, 36864, 48, 576, 1]cuda:0" = torch.ops.aten.permute.default(view_61, [2, 0, 3, 1, 4]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_5 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
        getitem_84: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_5[0]
        getitem_85: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_5[1]
        getitem_86: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_84, getitem_85, getitem_86, None, False);  getitem_84 = getitem_85 = getitem_86 = None
        getitem_87: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_38: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_87, [0, 2, 1, 3]);  getitem_87 = None
        view_62: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_38, [512, 64, 192]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_63: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_62, [32768, 192]);  view_62 = None
        permute_39: "bf16[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_21: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(arg185_1, view_63, permute_39);  arg185_1 = view_63 = permute_39 = None
        view_64: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [512, 64, 192]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_104: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, view_64);  add_101 = view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_203: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_203, [2], correction = 0, keepdim = True)
        getitem_96: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_12[0]
        getitem_97: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_35: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_203, getitem_97);  convert_element_type_203 = getitem_97 = None
        add_105: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_12: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        mul_93: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_12);  sub_35 = rsqrt_12 = None
        mul_94: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, arg186_1);  mul_93 = arg186_1 = None
        add_106: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_94, arg187_1);  mul_94 = arg187_1 = None
        convert_element_type_204: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16);  add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_65: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_204, [32768, 192]);  convert_element_type_204 = None
        permute_40: "bf16[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_22: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg189_1, view_65, permute_40);  arg189_1 = view_65 = permute_40 = None
        view_66: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [512, 64, 384]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_208: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_66, torch.float32);  view_66 = None
        neg_22: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_208)
        exp_22: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_107: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_208, add_107);  convert_element_type_208 = add_107 = None
        convert_element_type_209: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_67: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_209, [32768, 384]);  convert_element_type_209 = None
        permute_41: "bf16[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_23: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(arg191_1, view_67, permute_41);  arg191_1 = view_67 = permute_41 = None
        view_68: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [512, 64, 192]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_108: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_104, view_68);  add_104 = view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        convert_element_type_213: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.float32);  add_108 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_213, [2], correction = 0, keepdim = True)
        getitem_98: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_13[0]
        getitem_99: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_36: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_213, getitem_99);  convert_element_type_213 = getitem_99 = None
        add_109: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_13: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        mul_95: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_13);  sub_36 = rsqrt_13 = None
        mul_96: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, arg192_1);  mul_95 = arg192_1 = None
        add_110: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_96, arg193_1);  mul_96 = arg193_1 = None
        convert_element_type_214: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_110, torch.bfloat16);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_69: "bf16[128, 4, 64, 192][49152, 12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_214, [128, 4, 64, -1]);  convert_element_type_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_42: "bf16[128, 192, 64, 4][49152, 1, 192, 12288]cuda:0" = torch.ops.aten.permute.default(view_69, [0, 3, 2, 1]);  view_69 = None
        clone_28: "bf16[128, 192, 64, 4][49152, 256, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        view_70: "bf16[196608, 8, 2, 2][32, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [196608, 8, 2, 2]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_43: "bf16[196608, 2, 8, 2][32, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_29: "bf16[196608, 2, 8, 2][32, 16, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_71: "bf16[128, 192, 16, 16][49152, 256, 16, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [128, 192, 16, 16]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_25: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.convolution.default(view_71, arg194_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_71 = arg194_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_215: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg195_1, torch.float32);  arg195_1 = None
        unsqueeze_184: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_215, -1);  convert_element_type_215 = None
        unsqueeze_185: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        sub_37: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_185);  convolution_25 = unsqueeze_185 = None
        convert_element_type_216: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg196_1, torch.float32);  arg196_1 = None
        add_111: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_216, 1e-05);  convert_element_type_216 = None
        sqrt_23: "f32[128][1]cuda:0" = torch.ops.aten.sqrt.default(add_111);  add_111 = None
        reciprocal_23: "f32[128][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_97: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        unsqueeze_186: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_97, -1);  mul_97 = None
        unsqueeze_187: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        mul_98: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_187);  sub_37 = unsqueeze_187 = None
        unsqueeze_188: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_189: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_99: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_189);  mul_98 = unsqueeze_189 = None
        unsqueeze_190: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_191: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_112: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_191);  mul_99 = unsqueeze_191 = None
        convert_element_type_217: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_218: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_217, torch.float32);  convert_element_type_217 = None
        neg_23: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_218)
        exp_23: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_113: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        div_23: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_218, add_113);  convert_element_type_218 = add_113 = None
        convert_element_type_219: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_1: "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = torch.ops.aten.cat.default([convert_element_type_135, convert_element_type_219], 1);  convert_element_type_135 = convert_element_type_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_26: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.convolution.default(cat_1, arg199_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_1 = arg199_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_220: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg200_1, torch.float32);  arg200_1 = None
        unsqueeze_192: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_220, -1);  convert_element_type_220 = None
        unsqueeze_193: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        sub_38: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_193);  convolution_26 = unsqueeze_193 = None
        convert_element_type_221: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg201_1, torch.float32);  arg201_1 = None
        add_114: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_221, 1e-05);  convert_element_type_221 = None
        sqrt_24: "f32[128][1]cuda:0" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_24: "f32[128][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_100: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        unsqueeze_194: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_100, -1);  mul_100 = None
        unsqueeze_195: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        mul_101: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_195);  sub_38 = unsqueeze_195 = None
        unsqueeze_196: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_197: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_102: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, unsqueeze_197);  mul_101 = unsqueeze_197 = None
        unsqueeze_198: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_199: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_115: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_102, unsqueeze_199);  mul_102 = unsqueeze_199 = None
        convert_element_type_222: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16);  add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_223: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_222, torch.float32);  convert_element_type_222 = None
        neg_24: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_223)
        exp_24: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_116: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        div_24: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_223, add_116);  convert_element_type_223 = add_116 = None
        convert_element_type_224: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_27: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_224, arg204_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_224 = arg204_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_225: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg205_1, torch.float32);  arg205_1 = None
        unsqueeze_200: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_225, -1);  convert_element_type_225 = None
        unsqueeze_201: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        sub_39: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_201);  convolution_27 = unsqueeze_201 = None
        convert_element_type_226: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg206_1, torch.float32);  arg206_1 = None
        add_117: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_226, 1e-05);  convert_element_type_226 = None
        sqrt_25: "f32[512][1]cuda:0" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
        reciprocal_25: "f32[512][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_103: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        unsqueeze_202: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_103, -1);  mul_103 = None
        unsqueeze_203: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        mul_104: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_203);  sub_39 = unsqueeze_203 = None
        unsqueeze_204: "bf16[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_205: "bf16[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_105: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, unsqueeze_205);  mul_104 = unsqueeze_205 = None
        unsqueeze_206: "bf16[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_207: "bf16[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_118: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_105, unsqueeze_207);  mul_105 = unsqueeze_207 = None
        convert_element_type_227: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(add_118, torch.bfloat16);  add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_228: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_227, torch.float32);  convert_element_type_227 = None
        neg_25: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.neg.default(convert_element_type_228)
        exp_25: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_119: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_228, add_119);  convert_element_type_228 = add_119 = None
        convert_element_type_229: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_28: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_229, arg209_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 512);  convert_element_type_229 = arg209_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_230: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg210_1, torch.float32);  arg210_1 = None
        unsqueeze_208: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_230, -1);  convert_element_type_230 = None
        unsqueeze_209: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        sub_40: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_209);  convolution_28 = unsqueeze_209 = None
        convert_element_type_231: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg211_1, torch.float32);  arg211_1 = None
        add_120: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_231, 1e-05);  convert_element_type_231 = None
        sqrt_26: "f32[512][1]cuda:0" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_26: "f32[512][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_106: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        unsqueeze_210: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_106, -1);  mul_106 = None
        unsqueeze_211: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        mul_107: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_211);  sub_40 = unsqueeze_211 = None
        unsqueeze_212: "bf16[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_213: "bf16[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_108: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_213);  mul_107 = unsqueeze_213 = None
        unsqueeze_214: "bf16[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_215: "bf16[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_121: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_108, unsqueeze_215);  mul_108 = unsqueeze_215 = None
        convert_element_type_232: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_233: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_232, torch.float32);  convert_element_type_232 = None
        neg_26: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.neg.default(convert_element_type_233)
        exp_26: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_122: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        div_26: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_233, add_122);  convert_element_type_233 = add_122 = None
        convert_element_type_234: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_29: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_234, arg214_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_234 = arg214_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_235: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg215_1, torch.float32);  arg215_1 = None
        unsqueeze_216: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_235, -1);  convert_element_type_235 = None
        unsqueeze_217: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        sub_41: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_217);  convolution_29 = unsqueeze_217 = None
        convert_element_type_236: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg216_1, torch.float32);  arg216_1 = None
        add_123: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_236, 1e-05);  convert_element_type_236 = None
        sqrt_27: "f32[160][1]cuda:0" = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_27: "f32[160][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_109: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_218: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_109, -1);  mul_109 = None
        unsqueeze_219: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        mul_110: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_219);  sub_41 = unsqueeze_219 = None
        unsqueeze_220: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_221: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_111: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, unsqueeze_221);  mul_110 = unsqueeze_221 = None
        unsqueeze_222: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_223: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_124: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_223);  mul_111 = unsqueeze_223 = None
        convert_element_type_237: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_30: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_237, arg219_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg219_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_238: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg220_1, torch.float32);  arg220_1 = None
        unsqueeze_224: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_238, -1);  convert_element_type_238 = None
        unsqueeze_225: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        sub_42: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_225);  convolution_30 = unsqueeze_225 = None
        convert_element_type_239: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg221_1, torch.float32);  arg221_1 = None
        add_125: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_239, 1e-05);  convert_element_type_239 = None
        sqrt_28: "f32[160][1]cuda:0" = torch.ops.aten.sqrt.default(add_125);  add_125 = None
        reciprocal_28: "f32[160][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_112: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_226: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_112, -1);  mul_112 = None
        unsqueeze_227: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        mul_113: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_227);  sub_42 = unsqueeze_227 = None
        unsqueeze_228: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_229: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_114: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_229);  mul_113 = unsqueeze_229 = None
        unsqueeze_230: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_231: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_126: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_231);  mul_114 = unsqueeze_231 = None
        convert_element_type_240: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_126, torch.bfloat16);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_241: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_240, torch.float32);  convert_element_type_240 = None
        neg_27: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.neg.default(convert_element_type_241)
        exp_27: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_127: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        div_27: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_241, add_127);  convert_element_type_241 = add_127 = None
        convert_element_type_242: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_31: "bf16[128, 240, 8, 8][15360, 1, 1920, 240]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_242, arg224_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_242 = arg224_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        clone_30: "bf16[128, 240, 8, 8][15360, 64, 8, 1]cuda:0" = torch.ops.aten.clone.default(convolution_31, memory_format = torch.contiguous_format);  convolution_31 = None
        view_72: "bf16[122880, 2, 4, 2][16, 8, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [122880, 2, 4, 2]);  clone_30 = None
        permute_44: "bf16[122880, 4, 2, 2][16, 2, 8, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_31: "bf16[122880, 4, 2, 2][16, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_73: "bf16[128, 240, 16, 4][15360, 64, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [128, 240, 16, 4]);  clone_31 = None
        permute_45: "bf16[128, 4, 16, 240][15360, 1, 4, 64]cuda:0" = torch.ops.aten.permute.default(view_73, [0, 3, 2, 1]);  view_73 = None
        clone_32: "bf16[128, 4, 16, 240][15360, 3840, 240, 1]cuda:0" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_74: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [512, 16, 240]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_243: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_74, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_243, [2], correction = 0, keepdim = True)
        getitem_100: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_14[0]
        getitem_101: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_43: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_243, getitem_101);  convert_element_type_243 = getitem_101 = None
        add_128: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_14: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_115: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_14);  sub_43 = rsqrt_14 = None
        mul_116: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, arg225_1);  mul_115 = arg225_1 = None
        add_129: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, arg226_1);  mul_116 = arg226_1 = None
        convert_element_type_244: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_75: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_244, [8192, 240]);  convert_element_type_244 = None
        permute_46: "bf16[240, 720][1, 240]cuda:0" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_24: "bf16[8192, 720][720, 1]cuda:0" = torch.ops.aten.addmm.default(arg228_1, view_75, permute_46);  arg228_1 = view_75 = permute_46 = None
        view_76: "bf16[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [512, 16, 720]);  addmm_24 = None
        view_77: "bf16[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_76, [512, 16, 3, 4, 60]);  view_76 = None
        permute_47: "bf16[3, 512, 4, 16, 60][240, 11520, 60, 720, 1]cuda:0" = torch.ops.aten.permute.default(view_77, [2, 0, 3, 1, 4]);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_6 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
        getitem_102: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_6[0]
        getitem_103: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_6[1]
        getitem_104: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd_6: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_102, [0, 4], 0.0);  getitem_102 = None
        constant_pad_nd_7: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_103, [0, 4], 0.0);  getitem_103 = None
        constant_pad_nd_8: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_104, [0, 4], 0.0);  getitem_104 = None
        _scaled_dot_product_flash_attention_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(constant_pad_nd_6, constant_pad_nd_7, constant_pad_nd_8, scale = 0.12909944487358055);  constant_pad_nd_6 = constant_pad_nd_7 = constant_pad_nd_8 = None
        getitem_105: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_2[0];  _scaled_dot_product_flash_attention_2 = None
        slice_3: "bf16[512, 4, 16, 60][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(getitem_105, -1, 0, 60);  getitem_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_48: "bf16[512, 16, 4, 60][4096, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(slice_3, [0, 2, 1, 3]);  slice_3 = None
        clone_33: "bf16[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_78: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [512, 16, 240]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_79: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(view_78, [8192, 240]);  view_78 = None
        permute_49: "bf16[240, 240][1, 240]cuda:0" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_25: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(arg230_1, view_79, permute_49);  arg230_1 = view_79 = permute_49 = None
        view_80: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [512, 16, 240]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_130: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(view_74, view_80);  view_74 = view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_251: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_130, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_251, [2], correction = 0, keepdim = True)
        getitem_114: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_15[0]
        getitem_115: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_44: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_251, getitem_115);  convert_element_type_251 = getitem_115 = None
        add_131: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
        rsqrt_15: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        mul_117: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_15);  sub_44 = rsqrt_15 = None
        mul_118: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, arg231_1);  mul_117 = arg231_1 = None
        add_132: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_118, arg232_1);  mul_118 = arg232_1 = None
        convert_element_type_252: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_132, torch.bfloat16);  add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_81: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_252, [8192, 240]);  convert_element_type_252 = None
        permute_50: "bf16[240, 480][1, 240]cuda:0" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_26: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.addmm.default(arg234_1, view_81, permute_50);  arg234_1 = view_81 = permute_50 = None
        view_82: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [512, 16, 480]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_256: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_82, torch.float32);  view_82 = None
        neg_28: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_256)
        exp_28: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_133: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        div_28: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_256, add_133);  convert_element_type_256 = add_133 = None
        convert_element_type_257: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_83: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_257, [8192, 480]);  convert_element_type_257 = None
        permute_51: "bf16[480, 240][1, 480]cuda:0" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_27: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(arg236_1, view_83, permute_51);  arg236_1 = view_83 = permute_51 = None
        view_84: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [512, 16, 240]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_134: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_130, view_84);  add_130 = view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_261: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_261, [2], correction = 0, keepdim = True)
        getitem_116: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_16[0]
        getitem_117: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_45: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_261, getitem_117);  convert_element_type_261 = getitem_117 = None
        add_135: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_16: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        mul_119: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_16);  sub_45 = rsqrt_16 = None
        mul_120: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, arg237_1);  mul_119 = arg237_1 = None
        add_136: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, arg238_1);  mul_120 = arg238_1 = None
        convert_element_type_262: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_85: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_262, [8192, 240]);  convert_element_type_262 = None
        permute_52: "bf16[240, 720][1, 240]cuda:0" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_28: "bf16[8192, 720][720, 1]cuda:0" = torch.ops.aten.addmm.default(arg240_1, view_85, permute_52);  arg240_1 = view_85 = permute_52 = None
        view_86: "bf16[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [512, 16, 720]);  addmm_28 = None
        view_87: "bf16[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_86, [512, 16, 3, 4, 60]);  view_86 = None
        permute_53: "bf16[3, 512, 4, 16, 60][240, 11520, 60, 720, 1]cuda:0" = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_7 = torch.ops.aten.unbind.int(permute_53);  permute_53 = None
        getitem_118: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_7[0]
        getitem_119: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_7[1]
        getitem_120: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd_9: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_118, [0, 4], 0.0);  getitem_118 = None
        constant_pad_nd_10: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_119, [0, 4], 0.0);  getitem_119 = None
        constant_pad_nd_11: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_120, [0, 4], 0.0);  getitem_120 = None
        _scaled_dot_product_flash_attention_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(constant_pad_nd_9, constant_pad_nd_10, constant_pad_nd_11, scale = 0.12909944487358055);  constant_pad_nd_9 = constant_pad_nd_10 = constant_pad_nd_11 = None
        getitem_121: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_3[0];  _scaled_dot_product_flash_attention_3 = None
        slice_4: "bf16[512, 4, 16, 60][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(getitem_121, -1, 0, 60);  getitem_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_54: "bf16[512, 16, 4, 60][4096, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(slice_4, [0, 2, 1, 3]);  slice_4 = None
        clone_37: "bf16[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
        view_88: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [512, 16, 240]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_89: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(view_88, [8192, 240]);  view_88 = None
        permute_55: "bf16[240, 240][1, 240]cuda:0" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_29: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(arg242_1, view_89, permute_55);  arg242_1 = view_89 = permute_55 = None
        view_90: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [512, 16, 240]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_137: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, view_90);  add_134 = view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_269: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_269, [2], correction = 0, keepdim = True)
        getitem_130: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_17[0]
        getitem_131: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_46: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_269, getitem_131);  convert_element_type_269 = getitem_131 = None
        add_138: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_17: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_121: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_17);  sub_46 = rsqrt_17 = None
        mul_122: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, arg243_1);  mul_121 = arg243_1 = None
        add_139: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_122, arg244_1);  mul_122 = arg244_1 = None
        convert_element_type_270: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_91: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_270, [8192, 240]);  convert_element_type_270 = None
        permute_56: "bf16[240, 480][1, 240]cuda:0" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_30: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.addmm.default(arg246_1, view_91, permute_56);  arg246_1 = view_91 = permute_56 = None
        view_92: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [512, 16, 480]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_274: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_92, torch.float32);  view_92 = None
        neg_29: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_274)
        exp_29: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_140: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        div_29: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_274, add_140);  convert_element_type_274 = add_140 = None
        convert_element_type_275: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_29, torch.bfloat16);  div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_93: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_275, [8192, 480]);  convert_element_type_275 = None
        permute_57: "bf16[480, 240][1, 480]cuda:0" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_31: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(arg248_1, view_93, permute_57);  arg248_1 = view_93 = permute_57 = None
        view_94: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [512, 16, 240]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_141: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_137, view_94);  add_137 = view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_279: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_141, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_279, [2], correction = 0, keepdim = True)
        getitem_132: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_18[0]
        getitem_133: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_47: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_279, getitem_133);  convert_element_type_279 = getitem_133 = None
        add_142: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_18: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        mul_123: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_18);  sub_47 = rsqrt_18 = None
        mul_124: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, arg249_1);  mul_123 = arg249_1 = None
        add_143: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_124, arg250_1);  mul_124 = arg250_1 = None
        convert_element_type_280: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_95: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_280, [8192, 240]);  convert_element_type_280 = None
        permute_58: "bf16[240, 720][1, 240]cuda:0" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_32: "bf16[8192, 720][720, 1]cuda:0" = torch.ops.aten.addmm.default(arg252_1, view_95, permute_58);  arg252_1 = view_95 = permute_58 = None
        view_96: "bf16[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [512, 16, 720]);  addmm_32 = None
        view_97: "bf16[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_96, [512, 16, 3, 4, 60]);  view_96 = None
        permute_59: "bf16[3, 512, 4, 16, 60][240, 11520, 60, 720, 1]cuda:0" = torch.ops.aten.permute.default(view_97, [2, 0, 3, 1, 4]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_8 = torch.ops.aten.unbind.int(permute_59);  permute_59 = None
        getitem_134: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_8[0]
        getitem_135: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_8[1]
        getitem_136: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_8[2];  unbind_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd_12: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_134, [0, 4], 0.0);  getitem_134 = None
        constant_pad_nd_13: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_135, [0, 4], 0.0);  getitem_135 = None
        constant_pad_nd_14: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_136, [0, 4], 0.0);  getitem_136 = None
        _scaled_dot_product_flash_attention_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(constant_pad_nd_12, constant_pad_nd_13, constant_pad_nd_14, scale = 0.12909944487358055);  constant_pad_nd_12 = constant_pad_nd_13 = constant_pad_nd_14 = None
        getitem_137: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_4[0];  _scaled_dot_product_flash_attention_4 = None
        slice_5: "bf16[512, 4, 16, 60][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(getitem_137, -1, 0, 60);  getitem_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_60: "bf16[512, 16, 4, 60][4096, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(slice_5, [0, 2, 1, 3]);  slice_5 = None
        clone_41: "bf16[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_98: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [512, 16, 240]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_99: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(view_98, [8192, 240]);  view_98 = None
        permute_61: "bf16[240, 240][1, 240]cuda:0" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_33: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(arg254_1, view_99, permute_61);  arg254_1 = view_99 = permute_61 = None
        view_100: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [512, 16, 240]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_144: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_141, view_100);  add_141 = view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_287: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_287, [2], correction = 0, keepdim = True)
        getitem_146: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_19[0]
        getitem_147: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_48: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_287, getitem_147);  convert_element_type_287 = getitem_147 = None
        add_145: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_19: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        mul_125: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_19);  sub_48 = rsqrt_19 = None
        mul_126: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, arg255_1);  mul_125 = arg255_1 = None
        add_146: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_126, arg256_1);  mul_126 = arg256_1 = None
        convert_element_type_288: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_146, torch.bfloat16);  add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_101: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_288, [8192, 240]);  convert_element_type_288 = None
        permute_62: "bf16[240, 480][1, 240]cuda:0" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_34: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.addmm.default(arg258_1, view_101, permute_62);  arg258_1 = view_101 = permute_62 = None
        view_102: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [512, 16, 480]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_292: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_102, torch.float32);  view_102 = None
        neg_30: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_292)
        exp_30: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.exp.default(neg_30);  neg_30 = None
        add_147: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        div_30: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_292, add_147);  convert_element_type_292 = add_147 = None
        convert_element_type_293: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_30, torch.bfloat16);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_103: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_293, [8192, 480]);  convert_element_type_293 = None
        permute_63: "bf16[480, 240][1, 480]cuda:0" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_35: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(arg260_1, view_103, permute_63);  arg260_1 = view_103 = permute_63 = None
        view_104: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [512, 16, 240]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_148: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_144, view_104);  add_144 = view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        convert_element_type_297: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_148, torch.float32);  add_148 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_297, [2], correction = 0, keepdim = True)
        getitem_148: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_20[0]
        getitem_149: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_49: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_297, getitem_149);  convert_element_type_297 = getitem_149 = None
        add_149: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
        rsqrt_20: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_127: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_20);  sub_49 = rsqrt_20 = None
        mul_128: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, arg261_1);  mul_127 = arg261_1 = None
        add_150: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_128, arg262_1);  mul_128 = arg262_1 = None
        convert_element_type_298: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_150, torch.bfloat16);  add_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_105: "bf16[128, 4, 16, 240][15360, 3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_298, [128, 4, 16, -1]);  convert_element_type_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_64: "bf16[128, 240, 16, 4][15360, 1, 240, 3840]cuda:0" = torch.ops.aten.permute.default(view_105, [0, 3, 2, 1]);  view_105 = None
        clone_45: "bf16[128, 240, 16, 4][15360, 64, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_106: "bf16[122880, 4, 2, 2][16, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [122880, 4, 2, 2]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_65: "bf16[122880, 2, 4, 2][16, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_46: "bf16[122880, 2, 4, 2][16, 8, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_107: "bf16[128, 240, 8, 8][15360, 64, 8, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [128, 240, 8, 8]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_32: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.convolution.default(view_107, arg263_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_107 = arg263_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_299: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg264_1, torch.float32);  arg264_1 = None
        unsqueeze_232: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_299, -1);  convert_element_type_299 = None
        unsqueeze_233: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        sub_50: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_233);  convolution_32 = unsqueeze_233 = None
        convert_element_type_300: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg265_1, torch.float32);  arg265_1 = None
        add_151: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_300, 1e-05);  convert_element_type_300 = None
        sqrt_29: "f32[160][1]cuda:0" = torch.ops.aten.sqrt.default(add_151);  add_151 = None
        reciprocal_29: "f32[160][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_129: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_234: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
        unsqueeze_235: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        mul_130: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_235);  sub_50 = unsqueeze_235 = None
        unsqueeze_236: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_237: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_131: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_237);  mul_130 = unsqueeze_237 = None
        unsqueeze_238: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_239: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_152: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_239);  mul_131 = unsqueeze_239 = None
        convert_element_type_301: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_302: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_301, torch.float32);  convert_element_type_301 = None
        neg_31: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.neg.default(convert_element_type_302)
        exp_31: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_153: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        div_31: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_302, add_153);  convert_element_type_302 = add_153 = None
        convert_element_type_303: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16);  div_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_2: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.cat.default([convert_element_type_237, convert_element_type_303], 1);  convert_element_type_237 = convert_element_type_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_33: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.convolution.default(cat_2, arg268_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_2 = arg268_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_304: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg269_1, torch.float32);  arg269_1 = None
        unsqueeze_240: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_304, -1);  convert_element_type_304 = None
        unsqueeze_241: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        sub_51: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_241);  convolution_33 = unsqueeze_241 = None
        convert_element_type_305: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg270_1, torch.float32);  arg270_1 = None
        add_154: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_305, 1e-05);  convert_element_type_305 = None
        sqrt_30: "f32[160][1]cuda:0" = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_30: "f32[160][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_132: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_242: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
        unsqueeze_243: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        mul_133: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_243);  sub_51 = unsqueeze_243 = None
        unsqueeze_244: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_245: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_134: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_245);  mul_133 = unsqueeze_245 = None
        unsqueeze_246: "bf16[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_247: "bf16[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_155: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_247);  mul_134 = unsqueeze_247 = None
        convert_element_type_306: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16);  add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_307: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_306, torch.float32);  convert_element_type_306 = None
        neg_32: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.neg.default(convert_element_type_307)
        exp_32: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_156: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(exp_32, 1);  exp_32 = None
        div_32: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_307, add_156);  convert_element_type_307 = add_156 = None
        convert_element_type_308: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(div_32, torch.bfloat16);  div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_34: "bf16[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_308, arg273_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_308 = arg273_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_309: "f32[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg274_1, torch.float32);  arg274_1 = None
        unsqueeze_248: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_309, -1);  convert_element_type_309 = None
        unsqueeze_249: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        sub_52: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_249);  convolution_34 = unsqueeze_249 = None
        convert_element_type_310: "f32[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg275_1, torch.float32);  arg275_1 = None
        add_157: "f32[640][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_310, 1e-05);  convert_element_type_310 = None
        sqrt_31: "f32[640][1]cuda:0" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_31: "f32[640][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_135: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_250: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
        unsqueeze_251: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        mul_136: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_251);  sub_52 = unsqueeze_251 = None
        unsqueeze_252: "bf16[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_253: "bf16[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_137: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_253);  mul_136 = unsqueeze_253 = None
        unsqueeze_254: "bf16[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_255: "bf16[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_158: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_255);  mul_137 = unsqueeze_255 = None
        convert_element_type_311: "bf16[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.prims.convert_element_type.default(add_158, torch.bfloat16);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_312: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_311, torch.float32);  convert_element_type_311 = None
        neg_33: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.neg.default(convert_element_type_312)
        exp_33: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.exp.default(neg_33);  neg_33 = None
        add_159: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.add.Tensor(exp_33, 1);  exp_33 = None
        div_33: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_312, add_159);  convert_element_type_312 = add_159 = None
        convert_element_type_313: "bf16[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16);  div_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_313, [-1, -2], True);  convert_element_type_313 = None
        as_strided: "bf16[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.as_strided.default(mean, [128, 640, 1, 1], [640, 1, 640, 640]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_108: "bf16[128, 640][640, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided, [128, 640]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_66: "bf16[640, 1000][1, 640]cuda:0" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_36: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg279_1, view_108, permute_66);  arg279_1 = view_108 = permute_66 = None
        return (addmm_36,)
