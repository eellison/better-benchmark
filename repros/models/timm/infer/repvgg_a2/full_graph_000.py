class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 3, 1, 1][3, 1, 3, 3]cuda:0", arg1_1: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", arg2_1: "bf16[64][1]cuda:0", arg3_1: "bf16[64][1]cuda:0", arg4_1: "bf16[64][1]cuda:0", arg5_1: "bf16[64][1]cuda:0", arg6_1: "bf16[64, 3, 3, 3][27, 1, 9, 3]cuda:0", arg7_1: "bf16[64][1]cuda:0", arg8_1: "bf16[64][1]cuda:0", arg9_1: "bf16[64][1]cuda:0", arg10_1: "bf16[64][1]cuda:0", arg11_1: "bf16[96, 64, 1, 1][64, 1, 64, 64]cuda:0", arg12_1: "bf16[96][1]cuda:0", arg13_1: "bf16[96][1]cuda:0", arg14_1: "bf16[96][1]cuda:0", arg15_1: "bf16[96][1]cuda:0", arg16_1: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0", arg17_1: "bf16[96][1]cuda:0", arg18_1: "bf16[96][1]cuda:0", arg19_1: "bf16[96][1]cuda:0", arg20_1: "bf16[96][1]cuda:0", arg21_1: "bf16[96][1]cuda:0", arg22_1: "bf16[96][1]cuda:0", arg23_1: "bf16[96][1]cuda:0", arg24_1: "bf16[96][1]cuda:0", arg25_1: "bf16[96, 96, 1, 1][96, 1, 96, 96]cuda:0", arg26_1: "bf16[96][1]cuda:0", arg27_1: "bf16[96][1]cuda:0", arg28_1: "bf16[96][1]cuda:0", arg29_1: "bf16[96][1]cuda:0", arg30_1: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0", arg31_1: "bf16[96][1]cuda:0", arg32_1: "bf16[96][1]cuda:0", arg33_1: "bf16[96][1]cuda:0", arg34_1: "bf16[96][1]cuda:0", arg35_1: "bf16[192, 96, 1, 1][96, 1, 96, 96]cuda:0", arg36_1: "bf16[192][1]cuda:0", arg37_1: "bf16[192][1]cuda:0", arg38_1: "bf16[192][1]cuda:0", arg39_1: "bf16[192][1]cuda:0", arg40_1: "bf16[192, 96, 3, 3][864, 1, 288, 96]cuda:0", arg41_1: "bf16[192][1]cuda:0", arg42_1: "bf16[192][1]cuda:0", arg43_1: "bf16[192][1]cuda:0", arg44_1: "bf16[192][1]cuda:0", arg45_1: "bf16[192][1]cuda:0", arg46_1: "bf16[192][1]cuda:0", arg47_1: "bf16[192][1]cuda:0", arg48_1: "bf16[192][1]cuda:0", arg49_1: "bf16[192, 192, 1, 1][192, 1, 192, 192]cuda:0", arg50_1: "bf16[192][1]cuda:0", arg51_1: "bf16[192][1]cuda:0", arg52_1: "bf16[192][1]cuda:0", arg53_1: "bf16[192][1]cuda:0", arg54_1: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0", arg55_1: "bf16[192][1]cuda:0", arg56_1: "bf16[192][1]cuda:0", arg57_1: "bf16[192][1]cuda:0", arg58_1: "bf16[192][1]cuda:0", arg59_1: "bf16[192][1]cuda:0", arg60_1: "bf16[192][1]cuda:0", arg61_1: "bf16[192][1]cuda:0", arg62_1: "bf16[192][1]cuda:0", arg63_1: "bf16[192, 192, 1, 1][192, 1, 192, 192]cuda:0", arg64_1: "bf16[192][1]cuda:0", arg65_1: "bf16[192][1]cuda:0", arg66_1: "bf16[192][1]cuda:0", arg67_1: "bf16[192][1]cuda:0", arg68_1: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0", arg69_1: "bf16[192][1]cuda:0", arg70_1: "bf16[192][1]cuda:0", arg71_1: "bf16[192][1]cuda:0", arg72_1: "bf16[192][1]cuda:0", arg73_1: "bf16[192][1]cuda:0", arg74_1: "bf16[192][1]cuda:0", arg75_1: "bf16[192][1]cuda:0", arg76_1: "bf16[192][1]cuda:0", arg77_1: "bf16[192, 192, 1, 1][192, 1, 192, 192]cuda:0", arg78_1: "bf16[192][1]cuda:0", arg79_1: "bf16[192][1]cuda:0", arg80_1: "bf16[192][1]cuda:0", arg81_1: "bf16[192][1]cuda:0", arg82_1: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0", arg83_1: "bf16[192][1]cuda:0", arg84_1: "bf16[192][1]cuda:0", arg85_1: "bf16[192][1]cuda:0", arg86_1: "bf16[192][1]cuda:0", arg87_1: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0", arg88_1: "bf16[384][1]cuda:0", arg89_1: "bf16[384][1]cuda:0", arg90_1: "bf16[384][1]cuda:0", arg91_1: "bf16[384][1]cuda:0", arg92_1: "bf16[384, 192, 3, 3][1728, 1, 576, 192]cuda:0", arg93_1: "bf16[384][1]cuda:0", arg94_1: "bf16[384][1]cuda:0", arg95_1: "bf16[384][1]cuda:0", arg96_1: "bf16[384][1]cuda:0", arg97_1: "bf16[384][1]cuda:0", arg98_1: "bf16[384][1]cuda:0", arg99_1: "bf16[384][1]cuda:0", arg100_1: "bf16[384][1]cuda:0", arg101_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg102_1: "bf16[384][1]cuda:0", arg103_1: "bf16[384][1]cuda:0", arg104_1: "bf16[384][1]cuda:0", arg105_1: "bf16[384][1]cuda:0", arg106_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg107_1: "bf16[384][1]cuda:0", arg108_1: "bf16[384][1]cuda:0", arg109_1: "bf16[384][1]cuda:0", arg110_1: "bf16[384][1]cuda:0", arg111_1: "bf16[384][1]cuda:0", arg112_1: "bf16[384][1]cuda:0", arg113_1: "bf16[384][1]cuda:0", arg114_1: "bf16[384][1]cuda:0", arg115_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg116_1: "bf16[384][1]cuda:0", arg117_1: "bf16[384][1]cuda:0", arg118_1: "bf16[384][1]cuda:0", arg119_1: "bf16[384][1]cuda:0", arg120_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg121_1: "bf16[384][1]cuda:0", arg122_1: "bf16[384][1]cuda:0", arg123_1: "bf16[384][1]cuda:0", arg124_1: "bf16[384][1]cuda:0", arg125_1: "bf16[384][1]cuda:0", arg126_1: "bf16[384][1]cuda:0", arg127_1: "bf16[384][1]cuda:0", arg128_1: "bf16[384][1]cuda:0", arg129_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg130_1: "bf16[384][1]cuda:0", arg131_1: "bf16[384][1]cuda:0", arg132_1: "bf16[384][1]cuda:0", arg133_1: "bf16[384][1]cuda:0", arg134_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg135_1: "bf16[384][1]cuda:0", arg136_1: "bf16[384][1]cuda:0", arg137_1: "bf16[384][1]cuda:0", arg138_1: "bf16[384][1]cuda:0", arg139_1: "bf16[384][1]cuda:0", arg140_1: "bf16[384][1]cuda:0", arg141_1: "bf16[384][1]cuda:0", arg142_1: "bf16[384][1]cuda:0", arg143_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg144_1: "bf16[384][1]cuda:0", arg145_1: "bf16[384][1]cuda:0", arg146_1: "bf16[384][1]cuda:0", arg147_1: "bf16[384][1]cuda:0", arg148_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg149_1: "bf16[384][1]cuda:0", arg150_1: "bf16[384][1]cuda:0", arg151_1: "bf16[384][1]cuda:0", arg152_1: "bf16[384][1]cuda:0", arg153_1: "bf16[384][1]cuda:0", arg154_1: "bf16[384][1]cuda:0", arg155_1: "bf16[384][1]cuda:0", arg156_1: "bf16[384][1]cuda:0", arg157_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg158_1: "bf16[384][1]cuda:0", arg159_1: "bf16[384][1]cuda:0", arg160_1: "bf16[384][1]cuda:0", arg161_1: "bf16[384][1]cuda:0", arg162_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg163_1: "bf16[384][1]cuda:0", arg164_1: "bf16[384][1]cuda:0", arg165_1: "bf16[384][1]cuda:0", arg166_1: "bf16[384][1]cuda:0", arg167_1: "bf16[384][1]cuda:0", arg168_1: "bf16[384][1]cuda:0", arg169_1: "bf16[384][1]cuda:0", arg170_1: "bf16[384][1]cuda:0", arg171_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg172_1: "bf16[384][1]cuda:0", arg173_1: "bf16[384][1]cuda:0", arg174_1: "bf16[384][1]cuda:0", arg175_1: "bf16[384][1]cuda:0", arg176_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg177_1: "bf16[384][1]cuda:0", arg178_1: "bf16[384][1]cuda:0", arg179_1: "bf16[384][1]cuda:0", arg180_1: "bf16[384][1]cuda:0", arg181_1: "bf16[384][1]cuda:0", arg182_1: "bf16[384][1]cuda:0", arg183_1: "bf16[384][1]cuda:0", arg184_1: "bf16[384][1]cuda:0", arg185_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg186_1: "bf16[384][1]cuda:0", arg187_1: "bf16[384][1]cuda:0", arg188_1: "bf16[384][1]cuda:0", arg189_1: "bf16[384][1]cuda:0", arg190_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg191_1: "bf16[384][1]cuda:0", arg192_1: "bf16[384][1]cuda:0", arg193_1: "bf16[384][1]cuda:0", arg194_1: "bf16[384][1]cuda:0", arg195_1: "bf16[384][1]cuda:0", arg196_1: "bf16[384][1]cuda:0", arg197_1: "bf16[384][1]cuda:0", arg198_1: "bf16[384][1]cuda:0", arg199_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg200_1: "bf16[384][1]cuda:0", arg201_1: "bf16[384][1]cuda:0", arg202_1: "bf16[384][1]cuda:0", arg203_1: "bf16[384][1]cuda:0", arg204_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg205_1: "bf16[384][1]cuda:0", arg206_1: "bf16[384][1]cuda:0", arg207_1: "bf16[384][1]cuda:0", arg208_1: "bf16[384][1]cuda:0", arg209_1: "bf16[384][1]cuda:0", arg210_1: "bf16[384][1]cuda:0", arg211_1: "bf16[384][1]cuda:0", arg212_1: "bf16[384][1]cuda:0", arg213_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg214_1: "bf16[384][1]cuda:0", arg215_1: "bf16[384][1]cuda:0", arg216_1: "bf16[384][1]cuda:0", arg217_1: "bf16[384][1]cuda:0", arg218_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg219_1: "bf16[384][1]cuda:0", arg220_1: "bf16[384][1]cuda:0", arg221_1: "bf16[384][1]cuda:0", arg222_1: "bf16[384][1]cuda:0", arg223_1: "bf16[384][1]cuda:0", arg224_1: "bf16[384][1]cuda:0", arg225_1: "bf16[384][1]cuda:0", arg226_1: "bf16[384][1]cuda:0", arg227_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg228_1: "bf16[384][1]cuda:0", arg229_1: "bf16[384][1]cuda:0", arg230_1: "bf16[384][1]cuda:0", arg231_1: "bf16[384][1]cuda:0", arg232_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg233_1: "bf16[384][1]cuda:0", arg234_1: "bf16[384][1]cuda:0", arg235_1: "bf16[384][1]cuda:0", arg236_1: "bf16[384][1]cuda:0", arg237_1: "bf16[384][1]cuda:0", arg238_1: "bf16[384][1]cuda:0", arg239_1: "bf16[384][1]cuda:0", arg240_1: "bf16[384][1]cuda:0", arg241_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg242_1: "bf16[384][1]cuda:0", arg243_1: "bf16[384][1]cuda:0", arg244_1: "bf16[384][1]cuda:0", arg245_1: "bf16[384][1]cuda:0", arg246_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg247_1: "bf16[384][1]cuda:0", arg248_1: "bf16[384][1]cuda:0", arg249_1: "bf16[384][1]cuda:0", arg250_1: "bf16[384][1]cuda:0", arg251_1: "bf16[384][1]cuda:0", arg252_1: "bf16[384][1]cuda:0", arg253_1: "bf16[384][1]cuda:0", arg254_1: "bf16[384][1]cuda:0", arg255_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg256_1: "bf16[384][1]cuda:0", arg257_1: "bf16[384][1]cuda:0", arg258_1: "bf16[384][1]cuda:0", arg259_1: "bf16[384][1]cuda:0", arg260_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg261_1: "bf16[384][1]cuda:0", arg262_1: "bf16[384][1]cuda:0", arg263_1: "bf16[384][1]cuda:0", arg264_1: "bf16[384][1]cuda:0", arg265_1: "bf16[384][1]cuda:0", arg266_1: "bf16[384][1]cuda:0", arg267_1: "bf16[384][1]cuda:0", arg268_1: "bf16[384][1]cuda:0", arg269_1: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", arg270_1: "bf16[384][1]cuda:0", arg271_1: "bf16[384][1]cuda:0", arg272_1: "bf16[384][1]cuda:0", arg273_1: "bf16[384][1]cuda:0", arg274_1: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg275_1: "bf16[384][1]cuda:0", arg276_1: "bf16[384][1]cuda:0", arg277_1: "bf16[384][1]cuda:0", arg278_1: "bf16[384][1]cuda:0", arg279_1: "bf16[1408, 384, 1, 1][384, 1, 384, 384]cuda:0", arg280_1: "bf16[1408][1]cuda:0", arg281_1: "bf16[1408][1]cuda:0", arg282_1: "bf16[1408][1]cuda:0", arg283_1: "bf16[1408][1]cuda:0", arg284_1: "bf16[1408, 384, 3, 3][3456, 1, 1152, 384]cuda:0", arg285_1: "bf16[1408][1]cuda:0", arg286_1: "bf16[1408][1]cuda:0", arg287_1: "bf16[1408][1]cuda:0", arg288_1: "bf16[1408][1]cuda:0", arg289_1: "bf16[1000, 1408][1408, 1]cuda:0", arg290_1: "bf16[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        unsqueeze: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        convert_element_type_1: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
        sqrt: "f32[64][1]cuda:0" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[64][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_1: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.convolution.default(arg1_1, arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_3: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        unsqueeze_8: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, -1);  convert_element_type_3 = None
        unsqueeze_9: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        convert_element_type_4: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        add_2: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_4, 1e-05);  convert_element_type_4 = None
        sqrt_1: "f32[64][1]cuda:0" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal_1: "f32[64][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_4: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_13: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_15: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_3: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
        convert_element_type_5: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:748 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_4: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2, convert_element_type_5);  convert_element_type_2 = convert_element_type_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.relu.default(add_4);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_2: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.convolution.default(relu, arg11_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg11_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_6: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg12_1, torch.float32);  arg12_1 = None
        unsqueeze_16: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
        unsqueeze_17: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_2: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        convert_element_type_7: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        add_5: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
        sqrt_2: "f32[96][1]cuda:0" = torch.ops.aten.sqrt.default(add_5);  add_5 = None
        reciprocal_2: "f32[96][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_6: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
        unsqueeze_19: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_7: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_21: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_8: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
        unsqueeze_22: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_23: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_6: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
        convert_element_type_8: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_3: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.convolution.default(relu, arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_9: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        unsqueeze_24: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_9, -1);  convert_element_type_9 = None
        unsqueeze_25: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_3: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        convert_element_type_10: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg18_1, torch.float32);  arg18_1 = None
        add_7: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_10, 1e-05);  convert_element_type_10 = None
        sqrt_3: "f32[96][1]cuda:0" = torch.ops.aten.sqrt.default(add_7);  add_7 = None
        reciprocal_3: "f32[96][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_9: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_27: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_10: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_29: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_11: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
        unsqueeze_30: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_31: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_8: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
        convert_element_type_11: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:748 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_9: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_8, convert_element_type_11);  convert_element_type_8 = convert_element_type_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_1: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.relu.default(add_9);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_4: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.convolution.default(relu_1, arg25_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg25_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_15: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg26_1, torch.float32);  arg26_1 = None
        unsqueeze_40: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_15, -1);  convert_element_type_15 = None
        unsqueeze_41: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        sub_5: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_41);  convolution_4 = unsqueeze_41 = None
        convert_element_type_16: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg27_1, torch.float32);  arg27_1 = None
        add_12: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_16, 1e-05);  convert_element_type_16 = None
        sqrt_5: "f32[96][1]cuda:0" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
        reciprocal_5: "f32[96][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_15: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_42: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
        unsqueeze_43: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_16: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
        unsqueeze_44: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_45: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_17: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
        unsqueeze_46: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_47: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_13: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
        convert_element_type_17: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_5: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.convolution.default(relu_1, arg30_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg30_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_18: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg31_1, torch.float32);  arg31_1 = None
        unsqueeze_48: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
        unsqueeze_49: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        sub_6: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_49);  convolution_5 = unsqueeze_49 = None
        convert_element_type_19: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg32_1, torch.float32);  arg32_1 = None
        add_14: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
        sqrt_6: "f32[96][1]cuda:0" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
        reciprocal_6: "f32[96][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_18: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_50: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
        unsqueeze_51: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_19: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
        unsqueeze_52: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_53: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_20: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
        unsqueeze_54: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_55: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_15: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
        convert_element_type_20: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_16: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_17, convert_element_type_20);  convert_element_type_17 = convert_element_type_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_12: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg21_1, torch.float32);  arg21_1 = None
        unsqueeze_32: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
        unsqueeze_33: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        sub_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(relu_1, unsqueeze_33);  relu_1 = unsqueeze_33 = None
        convert_element_type_13: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg22_1, torch.float32);  arg22_1 = None
        add_10: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
        sqrt_4: "f32[96][1]cuda:0" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
        reciprocal_4: "f32[96][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_12: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_34: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
        unsqueeze_35: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_13: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
        unsqueeze_36: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_37: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_14: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
        unsqueeze_38: "bf16[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_39: "bf16[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_11: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
        convert_element_type_14: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_17: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(add_16, convert_element_type_14);  add_16 = convert_element_type_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_2: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.relu.default(add_17);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_6: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(relu_2, arg35_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg35_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_21: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg36_1, torch.float32);  arg36_1 = None
        unsqueeze_56: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_21, -1);  convert_element_type_21 = None
        unsqueeze_57: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        sub_7: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_57);  convolution_6 = unsqueeze_57 = None
        convert_element_type_22: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg37_1, torch.float32);  arg37_1 = None
        add_18: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_22, 1e-05);  convert_element_type_22 = None
        sqrt_7: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
        reciprocal_7: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_21: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_58: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
        unsqueeze_59: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_22: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
        unsqueeze_60: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_61: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_23: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
        unsqueeze_62: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_63: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_19: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
        convert_element_type_23: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_7: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(relu_2, arg40_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_24: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg41_1, torch.float32);  arg41_1 = None
        unsqueeze_64: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
        unsqueeze_65: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        sub_8: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_65);  convolution_7 = unsqueeze_65 = None
        convert_element_type_25: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg42_1, torch.float32);  arg42_1 = None
        add_20: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
        sqrt_8: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
        reciprocal_8: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_24: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_66: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
        unsqueeze_67: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_25: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
        unsqueeze_68: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_69: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_26: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
        unsqueeze_70: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_71: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_21: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
        convert_element_type_26: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:748 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_22: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_23, convert_element_type_26);  convert_element_type_23 = convert_element_type_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_3: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.relu.default(add_22);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_8: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(relu_3, arg49_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg49_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_30: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg50_1, torch.float32);  arg50_1 = None
        unsqueeze_80: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
        unsqueeze_81: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        sub_10: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_81);  convolution_8 = unsqueeze_81 = None
        convert_element_type_31: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg51_1, torch.float32);  arg51_1 = None
        add_25: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
        sqrt_10: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
        reciprocal_10: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_30: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_82: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
        unsqueeze_83: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_31: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
        unsqueeze_84: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_85: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_32: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
        unsqueeze_86: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_87: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_26: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
        convert_element_type_32: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_9: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(relu_3, arg54_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg54_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_33: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg55_1, torch.float32);  arg55_1 = None
        unsqueeze_88: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_33, -1);  convert_element_type_33 = None
        unsqueeze_89: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        sub_11: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_89);  convolution_9 = unsqueeze_89 = None
        convert_element_type_34: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg56_1, torch.float32);  arg56_1 = None
        add_27: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_34, 1e-05);  convert_element_type_34 = None
        sqrt_11: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
        reciprocal_11: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_33: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_90: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
        unsqueeze_91: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_34: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
        unsqueeze_92: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_93: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_35: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
        unsqueeze_94: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_95: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_28: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
        convert_element_type_35: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_29: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_32, convert_element_type_35);  convert_element_type_32 = convert_element_type_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_27: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg45_1, torch.float32);  arg45_1 = None
        unsqueeze_72: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_27, -1);  convert_element_type_27 = None
        unsqueeze_73: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        sub_9: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(relu_3, unsqueeze_73);  relu_3 = unsqueeze_73 = None
        convert_element_type_28: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg46_1, torch.float32);  arg46_1 = None
        add_23: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_28, 1e-05);  convert_element_type_28 = None
        sqrt_9: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
        reciprocal_9: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_27: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_74: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
        unsqueeze_75: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_28: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
        unsqueeze_76: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_77: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_29: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
        unsqueeze_78: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_79: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_24: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
        convert_element_type_29: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_30: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_29, convert_element_type_29);  add_29 = convert_element_type_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_4: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.relu.default(add_30);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_10: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(relu_4, arg63_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg63_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_39: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg64_1, torch.float32);  arg64_1 = None
        unsqueeze_104: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_39, -1);  convert_element_type_39 = None
        unsqueeze_105: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        sub_13: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_105);  convolution_10 = unsqueeze_105 = None
        convert_element_type_40: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg65_1, torch.float32);  arg65_1 = None
        add_33: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_40, 1e-05);  convert_element_type_40 = None
        sqrt_13: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
        reciprocal_13: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_39: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_106: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
        unsqueeze_107: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_40: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
        unsqueeze_108: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_109: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_41: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
        unsqueeze_110: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_111: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_34: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
        convert_element_type_41: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_11: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(relu_4, arg68_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg68_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_42: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg69_1, torch.float32);  arg69_1 = None
        unsqueeze_112: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
        unsqueeze_113: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        sub_14: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_113);  convolution_11 = unsqueeze_113 = None
        convert_element_type_43: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg70_1, torch.float32);  arg70_1 = None
        add_35: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
        sqrt_14: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
        reciprocal_14: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_42: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_114: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
        unsqueeze_115: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_43: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
        unsqueeze_116: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_117: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_44: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
        unsqueeze_118: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_119: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_36: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
        convert_element_type_44: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_37: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_41, convert_element_type_44);  convert_element_type_41 = convert_element_type_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_36: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg59_1, torch.float32);  arg59_1 = None
        unsqueeze_96: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
        unsqueeze_97: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        sub_12: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(relu_4, unsqueeze_97);  relu_4 = unsqueeze_97 = None
        convert_element_type_37: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg60_1, torch.float32);  arg60_1 = None
        add_31: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
        sqrt_12: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
        reciprocal_12: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_36: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_98: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
        unsqueeze_99: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_37: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
        unsqueeze_100: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_101: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_38: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
        unsqueeze_102: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_103: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_32: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
        convert_element_type_38: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_38: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_37, convert_element_type_38);  add_37 = convert_element_type_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_5: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.relu.default(add_38);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_12: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(relu_5, arg77_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg77_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_48: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg78_1, torch.float32);  arg78_1 = None
        unsqueeze_128: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
        unsqueeze_129: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        sub_16: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_129);  convolution_12 = unsqueeze_129 = None
        convert_element_type_49: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg79_1, torch.float32);  arg79_1 = None
        add_41: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
        sqrt_16: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
        reciprocal_16: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_48: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_130: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
        unsqueeze_131: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_49: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
        unsqueeze_132: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_133: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_50: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
        unsqueeze_134: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_135: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_42: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
        convert_element_type_50: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_42, torch.bfloat16);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_13: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(relu_5, arg82_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg82_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_51: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg83_1, torch.float32);  arg83_1 = None
        unsqueeze_136: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_51, -1);  convert_element_type_51 = None
        unsqueeze_137: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        sub_17: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_137);  convolution_13 = unsqueeze_137 = None
        convert_element_type_52: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg84_1, torch.float32);  arg84_1 = None
        add_43: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_52, 1e-05);  convert_element_type_52 = None
        sqrt_17: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
        reciprocal_17: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_51: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        unsqueeze_138: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
        unsqueeze_139: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        mul_52: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
        unsqueeze_140: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_141: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_53: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
        unsqueeze_142: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_143: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_44: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
        convert_element_type_53: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_45: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_50, convert_element_type_53);  convert_element_type_50 = convert_element_type_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_45: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg73_1, torch.float32);  arg73_1 = None
        unsqueeze_120: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_45, -1);  convert_element_type_45 = None
        unsqueeze_121: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        sub_15: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(relu_5, unsqueeze_121);  relu_5 = unsqueeze_121 = None
        convert_element_type_46: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg74_1, torch.float32);  arg74_1 = None
        add_39: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_46, 1e-05);  convert_element_type_46 = None
        sqrt_15: "f32[192][1]cuda:0" = torch.ops.aten.sqrt.default(add_39);  add_39 = None
        reciprocal_15: "f32[192][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_45: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_122: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
        unsqueeze_123: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_46: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
        unsqueeze_124: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_125: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_47: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
        unsqueeze_126: "bf16[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_127: "bf16[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_40: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
        convert_element_type_47: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_46: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_45, convert_element_type_47);  add_45 = convert_element_type_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_6: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.relu.default(add_46);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_14: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_6, arg87_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg87_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_54: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg88_1, torch.float32);  arg88_1 = None
        unsqueeze_144: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
        unsqueeze_145: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        sub_18: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_145);  convolution_14 = unsqueeze_145 = None
        convert_element_type_55: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg89_1, torch.float32);  arg89_1 = None
        add_47: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
        sqrt_18: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
        reciprocal_18: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_54: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        unsqueeze_146: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
        unsqueeze_147: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        mul_55: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
        unsqueeze_148: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_149: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_56: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
        unsqueeze_150: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_151: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_48: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
        convert_element_type_56: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_15: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_6, arg92_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_6 = arg92_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_57: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg93_1, torch.float32);  arg93_1 = None
        unsqueeze_152: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_57, -1);  convert_element_type_57 = None
        unsqueeze_153: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        sub_19: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_153);  convolution_15 = unsqueeze_153 = None
        convert_element_type_58: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg94_1, torch.float32);  arg94_1 = None
        add_49: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_58, 1e-05);  convert_element_type_58 = None
        sqrt_19: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_49);  add_49 = None
        reciprocal_19: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_57: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        unsqueeze_154: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
        unsqueeze_155: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        mul_58: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
        unsqueeze_156: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_157: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_59: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
        unsqueeze_158: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_159: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_50: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
        convert_element_type_59: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:748 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_51: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_56, convert_element_type_59);  convert_element_type_56 = convert_element_type_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_7: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_51);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_16: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_7, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg101_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_63: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg102_1, torch.float32);  arg102_1 = None
        unsqueeze_168: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_63, -1);  convert_element_type_63 = None
        unsqueeze_169: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        sub_21: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_169);  convolution_16 = unsqueeze_169 = None
        convert_element_type_64: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg103_1, torch.float32);  arg103_1 = None
        add_54: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_64, 1e-05);  convert_element_type_64 = None
        sqrt_21: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
        reciprocal_21: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        mul_63: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        unsqueeze_170: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
        unsqueeze_171: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        mul_64: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
        unsqueeze_172: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_173: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_65: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
        unsqueeze_174: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_175: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_55: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
        convert_element_type_65: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_17: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_7, arg106_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg106_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_66: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg107_1, torch.float32);  arg107_1 = None
        unsqueeze_176: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
        unsqueeze_177: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        sub_22: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_177);  convolution_17 = unsqueeze_177 = None
        convert_element_type_67: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg108_1, torch.float32);  arg108_1 = None
        add_56: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
        sqrt_22: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
        reciprocal_22: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        mul_66: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        unsqueeze_178: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
        unsqueeze_179: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        mul_67: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
        unsqueeze_180: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_181: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_68: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
        unsqueeze_182: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_183: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_57: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
        convert_element_type_68: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_58: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_65, convert_element_type_68);  convert_element_type_65 = convert_element_type_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_60: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg97_1, torch.float32);  arg97_1 = None
        unsqueeze_160: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
        unsqueeze_161: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        sub_20: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_7, unsqueeze_161);  relu_7 = unsqueeze_161 = None
        convert_element_type_61: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg98_1, torch.float32);  arg98_1 = None
        add_52: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
        sqrt_20: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
        reciprocal_20: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        mul_60: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        unsqueeze_162: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
        unsqueeze_163: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        mul_61: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
        unsqueeze_164: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_165: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_62: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
        unsqueeze_166: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_167: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_53: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
        convert_element_type_62: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_59: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_58, convert_element_type_62);  add_58 = convert_element_type_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_8: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_59);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_18: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_8, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg115_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_72: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg116_1, torch.float32);  arg116_1 = None
        unsqueeze_192: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
        unsqueeze_193: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        sub_24: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_193);  convolution_18 = unsqueeze_193 = None
        convert_element_type_73: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg117_1, torch.float32);  arg117_1 = None
        add_62: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
        sqrt_24: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
        reciprocal_24: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_72: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        unsqueeze_194: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
        unsqueeze_195: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        mul_73: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
        unsqueeze_196: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_197: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_74: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
        unsqueeze_198: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_199: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_63: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
        convert_element_type_74: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_19: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_8, arg120_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg120_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_75: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg121_1, torch.float32);  arg121_1 = None
        unsqueeze_200: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_75, -1);  convert_element_type_75 = None
        unsqueeze_201: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        sub_25: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_201);  convolution_19 = unsqueeze_201 = None
        convert_element_type_76: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg122_1, torch.float32);  arg122_1 = None
        add_64: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_76, 1e-05);  convert_element_type_76 = None
        sqrt_25: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
        reciprocal_25: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_75: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        unsqueeze_202: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
        unsqueeze_203: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        mul_76: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
        unsqueeze_204: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_205: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_77: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
        unsqueeze_206: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_207: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_65: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
        convert_element_type_77: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_66: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_74, convert_element_type_77);  convert_element_type_74 = convert_element_type_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_69: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg111_1, torch.float32);  arg111_1 = None
        unsqueeze_184: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_69, -1);  convert_element_type_69 = None
        unsqueeze_185: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        sub_23: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_8, unsqueeze_185);  relu_8 = unsqueeze_185 = None
        convert_element_type_70: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg112_1, torch.float32);  arg112_1 = None
        add_60: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_70, 1e-05);  convert_element_type_70 = None
        sqrt_23: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
        reciprocal_23: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_69: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        unsqueeze_186: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
        unsqueeze_187: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        mul_70: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
        unsqueeze_188: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_189: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_71: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
        unsqueeze_190: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_191: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_61: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
        convert_element_type_71: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_67: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_66, convert_element_type_71);  add_66 = convert_element_type_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_9: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_67);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_20: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_9, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg129_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_81: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg130_1, torch.float32);  arg130_1 = None
        unsqueeze_216: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_81, -1);  convert_element_type_81 = None
        unsqueeze_217: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        sub_27: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_217);  convolution_20 = unsqueeze_217 = None
        convert_element_type_82: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg131_1, torch.float32);  arg131_1 = None
        add_70: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_82, 1e-05);  convert_element_type_82 = None
        sqrt_27: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
        reciprocal_27: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_81: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_218: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
        unsqueeze_219: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        mul_82: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
        unsqueeze_220: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_221: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_83: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
        unsqueeze_222: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_223: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_71: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
        convert_element_type_83: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_21: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_9, arg134_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg134_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_84: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg135_1, torch.float32);  arg135_1 = None
        unsqueeze_224: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
        unsqueeze_225: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        sub_28: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_225);  convolution_21 = unsqueeze_225 = None
        convert_element_type_85: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg136_1, torch.float32);  arg136_1 = None
        add_72: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
        sqrt_28: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
        reciprocal_28: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_84: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_226: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
        unsqueeze_227: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        mul_85: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
        unsqueeze_228: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_229: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_86: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
        unsqueeze_230: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_231: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_73: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
        convert_element_type_86: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_74: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_83, convert_element_type_86);  convert_element_type_83 = convert_element_type_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_78: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg125_1, torch.float32);  arg125_1 = None
        unsqueeze_208: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
        unsqueeze_209: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        sub_26: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_9, unsqueeze_209);  relu_9 = unsqueeze_209 = None
        convert_element_type_79: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg126_1, torch.float32);  arg126_1 = None
        add_68: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
        sqrt_26: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
        reciprocal_26: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_78: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        unsqueeze_210: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
        unsqueeze_211: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        mul_79: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
        unsqueeze_212: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_213: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_80: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
        unsqueeze_214: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_215: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_69: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
        convert_element_type_80: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_75: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_74, convert_element_type_80);  add_74 = convert_element_type_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_10: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_75);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_22: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_10, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg143_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_90: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg144_1, torch.float32);  arg144_1 = None
        unsqueeze_240: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
        unsqueeze_241: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        sub_30: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_241);  convolution_22 = unsqueeze_241 = None
        convert_element_type_91: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg145_1, torch.float32);  arg145_1 = None
        add_78: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
        sqrt_30: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
        reciprocal_30: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_90: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_242: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
        unsqueeze_243: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        mul_91: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
        unsqueeze_244: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_245: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_92: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
        unsqueeze_246: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_247: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_79: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
        convert_element_type_92: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_23: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_10, arg148_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg148_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_93: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg149_1, torch.float32);  arg149_1 = None
        unsqueeze_248: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_93, -1);  convert_element_type_93 = None
        unsqueeze_249: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        sub_31: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_249);  convolution_23 = unsqueeze_249 = None
        convert_element_type_94: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg150_1, torch.float32);  arg150_1 = None
        add_80: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_94, 1e-05);  convert_element_type_94 = None
        sqrt_31: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
        reciprocal_31: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_93: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_250: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
        unsqueeze_251: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        mul_94: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
        unsqueeze_252: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_253: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_95: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
        unsqueeze_254: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_255: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_81: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
        convert_element_type_95: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_82: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_92, convert_element_type_95);  convert_element_type_92 = convert_element_type_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_87: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg139_1, torch.float32);  arg139_1 = None
        unsqueeze_232: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_87, -1);  convert_element_type_87 = None
        unsqueeze_233: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        sub_29: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_10, unsqueeze_233);  relu_10 = unsqueeze_233 = None
        convert_element_type_88: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float32);  arg140_1 = None
        add_76: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_88, 1e-05);  convert_element_type_88 = None
        sqrt_29: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
        reciprocal_29: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_87: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_234: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
        unsqueeze_235: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        mul_88: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
        unsqueeze_236: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_237: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_89: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
        unsqueeze_238: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_239: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_77: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
        convert_element_type_89: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_83: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_82, convert_element_type_89);  add_82 = convert_element_type_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_11: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_83);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_24: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_11, arg157_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg157_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_99: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg158_1, torch.float32);  arg158_1 = None
        unsqueeze_264: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_99, -1);  convert_element_type_99 = None
        unsqueeze_265: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        sub_33: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_265);  convolution_24 = unsqueeze_265 = None
        convert_element_type_100: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg159_1, torch.float32);  arg159_1 = None
        add_86: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_100, 1e-05);  convert_element_type_100 = None
        sqrt_33: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
        reciprocal_33: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_99: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_266: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
        unsqueeze_267: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        mul_100: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
        unsqueeze_268: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_269: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_101: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
        unsqueeze_270: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_271: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_87: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
        convert_element_type_101: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_25: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_11, arg162_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg162_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_102: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg163_1, torch.float32);  arg163_1 = None
        unsqueeze_272: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
        unsqueeze_273: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        sub_34: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_273);  convolution_25 = unsqueeze_273 = None
        convert_element_type_103: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg164_1, torch.float32);  arg164_1 = None
        add_88: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
        sqrt_34: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
        reciprocal_34: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_102: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_274: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
        unsqueeze_275: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        mul_103: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
        unsqueeze_276: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_277: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_104: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
        unsqueeze_278: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_279: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_89: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
        convert_element_type_104: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_90: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_101, convert_element_type_104);  convert_element_type_101 = convert_element_type_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_96: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg153_1, torch.float32);  arg153_1 = None
        unsqueeze_256: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
        unsqueeze_257: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        sub_32: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_11, unsqueeze_257);  relu_11 = unsqueeze_257 = None
        convert_element_type_97: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg154_1, torch.float32);  arg154_1 = None
        add_84: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
        sqrt_32: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_32: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_96: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_258: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
        unsqueeze_259: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        mul_97: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
        unsqueeze_260: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_261: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_98: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
        unsqueeze_262: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_263: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_85: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
        convert_element_type_98: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_91: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_90, convert_element_type_98);  add_90 = convert_element_type_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_12: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_91);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_26: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_12, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg171_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_108: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg172_1, torch.float32);  arg172_1 = None
        unsqueeze_288: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_108, -1);  convert_element_type_108 = None
        unsqueeze_289: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        sub_36: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_289);  convolution_26 = unsqueeze_289 = None
        convert_element_type_109: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg173_1, torch.float32);  arg173_1 = None
        add_94: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_109, 1e-05);  convert_element_type_109 = None
        sqrt_36: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
        reciprocal_36: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_108: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_290: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
        unsqueeze_291: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        mul_109: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
        unsqueeze_292: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_293: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_110: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
        unsqueeze_294: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_295: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_95: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
        convert_element_type_110: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16);  add_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_27: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_12, arg176_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg176_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_111: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg177_1, torch.float32);  arg177_1 = None
        unsqueeze_296: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_111, -1);  convert_element_type_111 = None
        unsqueeze_297: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        sub_37: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_297);  convolution_27 = unsqueeze_297 = None
        convert_element_type_112: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg178_1, torch.float32);  arg178_1 = None
        add_96: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_112, 1e-05);  convert_element_type_112 = None
        sqrt_37: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
        reciprocal_37: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_111: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_298: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
        unsqueeze_299: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        mul_112: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
        unsqueeze_300: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_301: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_113: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
        unsqueeze_302: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_303: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_97: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
        convert_element_type_113: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_98: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_110, convert_element_type_113);  convert_element_type_110 = convert_element_type_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_105: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg167_1, torch.float32);  arg167_1 = None
        unsqueeze_280: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_105, -1);  convert_element_type_105 = None
        unsqueeze_281: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        sub_35: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_12, unsqueeze_281);  relu_12 = unsqueeze_281 = None
        convert_element_type_106: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg168_1, torch.float32);  arg168_1 = None
        add_92: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_106, 1e-05);  convert_element_type_106 = None
        sqrt_35: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
        reciprocal_35: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_105: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_282: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
        unsqueeze_283: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        mul_106: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
        unsqueeze_284: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_285: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_107: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
        unsqueeze_286: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_287: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_93: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
        convert_element_type_107: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_99: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_98, convert_element_type_107);  add_98 = convert_element_type_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_13: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_99);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_28: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_13, arg185_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg185_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_117: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg186_1, torch.float32);  arg186_1 = None
        unsqueeze_312: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_117, -1);  convert_element_type_117 = None
        unsqueeze_313: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        sub_39: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_313);  convolution_28 = unsqueeze_313 = None
        convert_element_type_118: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg187_1, torch.float32);  arg187_1 = None
        add_102: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_118, 1e-05);  convert_element_type_118 = None
        sqrt_39: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
        reciprocal_39: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_117: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_314: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
        unsqueeze_315: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        mul_118: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
        unsqueeze_316: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_317: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_119: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
        unsqueeze_318: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_319: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_103: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
        convert_element_type_119: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_29: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_13, arg190_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg190_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_120: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg191_1, torch.float32);  arg191_1 = None
        unsqueeze_320: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_120, -1);  convert_element_type_120 = None
        unsqueeze_321: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        sub_40: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_321);  convolution_29 = unsqueeze_321 = None
        convert_element_type_121: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg192_1, torch.float32);  arg192_1 = None
        add_104: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_121, 1e-05);  convert_element_type_121 = None
        sqrt_40: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_104);  add_104 = None
        reciprocal_40: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_120: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_322: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
        unsqueeze_323: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        mul_121: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
        unsqueeze_324: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_325: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_122: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
        unsqueeze_326: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_327: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_105: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
        convert_element_type_122: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_106: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_119, convert_element_type_122);  convert_element_type_119 = convert_element_type_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_114: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg181_1, torch.float32);  arg181_1 = None
        unsqueeze_304: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_114, -1);  convert_element_type_114 = None
        unsqueeze_305: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        sub_38: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_13, unsqueeze_305);  relu_13 = unsqueeze_305 = None
        convert_element_type_115: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg182_1, torch.float32);  arg182_1 = None
        add_100: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_115, 1e-05);  convert_element_type_115 = None
        sqrt_38: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
        reciprocal_38: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_114: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_306: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
        unsqueeze_307: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        mul_115: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
        unsqueeze_308: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_309: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_116: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
        unsqueeze_310: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_311: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_101: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
        convert_element_type_116: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_107: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_106, convert_element_type_116);  add_106 = convert_element_type_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_14: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_107);  add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_30: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_14, arg199_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg199_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_126: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg200_1, torch.float32);  arg200_1 = None
        unsqueeze_336: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_126, -1);  convert_element_type_126 = None
        unsqueeze_337: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        sub_42: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_337);  convolution_30 = unsqueeze_337 = None
        convert_element_type_127: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg201_1, torch.float32);  arg201_1 = None
        add_110: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_127, 1e-05);  convert_element_type_127 = None
        sqrt_42: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
        reciprocal_42: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_126: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_338: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
        unsqueeze_339: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        mul_127: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
        unsqueeze_340: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_341: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_128: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
        unsqueeze_342: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_343: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_111: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
        convert_element_type_128: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_31: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_14, arg204_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg204_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_129: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg205_1, torch.float32);  arg205_1 = None
        unsqueeze_344: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_129, -1);  convert_element_type_129 = None
        unsqueeze_345: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        sub_43: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_345);  convolution_31 = unsqueeze_345 = None
        convert_element_type_130: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg206_1, torch.float32);  arg206_1 = None
        add_112: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_130, 1e-05);  convert_element_type_130 = None
        sqrt_43: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
        reciprocal_43: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_129: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_346: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
        unsqueeze_347: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        mul_130: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
        unsqueeze_348: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_349: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_131: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
        unsqueeze_350: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_351: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_113: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
        convert_element_type_131: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_113, torch.bfloat16);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_114: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_128, convert_element_type_131);  convert_element_type_128 = convert_element_type_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_123: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg195_1, torch.float32);  arg195_1 = None
        unsqueeze_328: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_123, -1);  convert_element_type_123 = None
        unsqueeze_329: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        sub_41: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_14, unsqueeze_329);  relu_14 = unsqueeze_329 = None
        convert_element_type_124: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg196_1, torch.float32);  arg196_1 = None
        add_108: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_124, 1e-05);  convert_element_type_124 = None
        sqrt_41: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
        reciprocal_41: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_123: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_330: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_331: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        mul_124: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
        unsqueeze_332: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_333: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_125: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
        unsqueeze_334: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_335: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_109: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
        convert_element_type_125: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_115: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_114, convert_element_type_125);  add_114 = convert_element_type_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_15: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_115);  add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_32: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_15, arg213_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg213_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_135: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg214_1, torch.float32);  arg214_1 = None
        unsqueeze_360: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_135, -1);  convert_element_type_135 = None
        unsqueeze_361: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        sub_45: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_361);  convolution_32 = unsqueeze_361 = None
        convert_element_type_136: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg215_1, torch.float32);  arg215_1 = None
        add_118: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_136, 1e-05);  convert_element_type_136 = None
        sqrt_45: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_118);  add_118 = None
        reciprocal_45: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_135: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_362: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
        unsqueeze_363: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        mul_136: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
        unsqueeze_364: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_365: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_137: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
        unsqueeze_366: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_367: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_119: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
        convert_element_type_137: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_33: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_15, arg218_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg218_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_138: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg219_1, torch.float32);  arg219_1 = None
        unsqueeze_368: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_138, -1);  convert_element_type_138 = None
        unsqueeze_369: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        sub_46: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_369);  convolution_33 = unsqueeze_369 = None
        convert_element_type_139: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg220_1, torch.float32);  arg220_1 = None
        add_120: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_139, 1e-05);  convert_element_type_139 = None
        sqrt_46: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_46: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_138: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_370: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
        unsqueeze_371: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        mul_139: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
        unsqueeze_372: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_373: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_140: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
        unsqueeze_374: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_375: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_121: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
        convert_element_type_140: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_122: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_137, convert_element_type_140);  convert_element_type_137 = convert_element_type_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_132: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg209_1, torch.float32);  arg209_1 = None
        unsqueeze_352: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_132, -1);  convert_element_type_132 = None
        unsqueeze_353: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        sub_44: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_15, unsqueeze_353);  relu_15 = unsqueeze_353 = None
        convert_element_type_133: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg210_1, torch.float32);  arg210_1 = None
        add_116: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_133, 1e-05);  convert_element_type_133 = None
        sqrt_44: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_44: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_132: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_354: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
        unsqueeze_355: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        mul_133: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
        unsqueeze_356: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_357: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_134: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
        unsqueeze_358: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_359: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_117: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
        convert_element_type_134: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_123: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_122, convert_element_type_134);  add_122 = convert_element_type_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_16: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_123);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_34: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_16, arg227_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg227_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_144: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg228_1, torch.float32);  arg228_1 = None
        unsqueeze_384: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_144, -1);  convert_element_type_144 = None
        unsqueeze_385: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        sub_48: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_385);  convolution_34 = unsqueeze_385 = None
        convert_element_type_145: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg229_1, torch.float32);  arg229_1 = None
        add_126: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_145, 1e-05);  convert_element_type_145 = None
        sqrt_48: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_126);  add_126 = None
        reciprocal_48: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_144: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_386: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
        unsqueeze_387: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        mul_145: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
        unsqueeze_388: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_389: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_146: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
        unsqueeze_390: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
        unsqueeze_391: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_127: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
        convert_element_type_146: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16);  add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_35: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_16, arg232_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg232_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_147: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg233_1, torch.float32);  arg233_1 = None
        unsqueeze_392: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_147, -1);  convert_element_type_147 = None
        unsqueeze_393: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        sub_49: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_393);  convolution_35 = unsqueeze_393 = None
        convert_element_type_148: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg234_1, torch.float32);  arg234_1 = None
        add_128: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_148, 1e-05);  convert_element_type_148 = None
        sqrt_49: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_128);  add_128 = None
        reciprocal_49: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_147: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_394: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
        unsqueeze_395: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        mul_148: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
        unsqueeze_396: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_397: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_149: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
        unsqueeze_398: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_399: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_129: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
        convert_element_type_149: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_130: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_146, convert_element_type_149);  convert_element_type_146 = convert_element_type_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_141: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg223_1, torch.float32);  arg223_1 = None
        unsqueeze_376: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_141, -1);  convert_element_type_141 = None
        unsqueeze_377: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        sub_47: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_16, unsqueeze_377);  relu_16 = unsqueeze_377 = None
        convert_element_type_142: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg224_1, torch.float32);  arg224_1 = None
        add_124: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_142, 1e-05);  convert_element_type_142 = None
        sqrt_47: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
        reciprocal_47: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_141: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_378: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
        unsqueeze_379: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        mul_142: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
        unsqueeze_380: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_381: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_143: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
        unsqueeze_382: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
        unsqueeze_383: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_125: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
        convert_element_type_143: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_131: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_130, convert_element_type_143);  add_130 = convert_element_type_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_17: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_131);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_36: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_17, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg241_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_153: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg242_1, torch.float32);  arg242_1 = None
        unsqueeze_408: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_153, -1);  convert_element_type_153 = None
        unsqueeze_409: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        sub_51: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_409);  convolution_36 = unsqueeze_409 = None
        convert_element_type_154: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg243_1, torch.float32);  arg243_1 = None
        add_134: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_154, 1e-05);  convert_element_type_154 = None
        sqrt_51: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_51: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_153: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_410: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
        unsqueeze_411: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        mul_154: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
        unsqueeze_412: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_413: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_155: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
        unsqueeze_414: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_415: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_135: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
        convert_element_type_155: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_37: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_17, arg246_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg246_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_156: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg247_1, torch.float32);  arg247_1 = None
        unsqueeze_416: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_156, -1);  convert_element_type_156 = None
        unsqueeze_417: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        sub_52: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_417);  convolution_37 = unsqueeze_417 = None
        convert_element_type_157: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg248_1, torch.float32);  arg248_1 = None
        add_136: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_157, 1e-05);  convert_element_type_157 = None
        sqrt_52: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_52: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_156: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_418: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
        unsqueeze_419: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        mul_157: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
        unsqueeze_420: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_421: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_158: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
        unsqueeze_422: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_423: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_137: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
        convert_element_type_158: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_138: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_155, convert_element_type_158);  convert_element_type_155 = convert_element_type_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_150: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg237_1, torch.float32);  arg237_1 = None
        unsqueeze_400: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_150, -1);  convert_element_type_150 = None
        unsqueeze_401: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        sub_50: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_17, unsqueeze_401);  relu_17 = unsqueeze_401 = None
        convert_element_type_151: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg238_1, torch.float32);  arg238_1 = None
        add_132: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_151, 1e-05);  convert_element_type_151 = None
        sqrt_50: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_50: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_150: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_402: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
        unsqueeze_403: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        mul_151: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
        unsqueeze_404: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_405: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_152: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
        unsqueeze_406: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_407: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_133: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
        convert_element_type_152: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_139: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_138, convert_element_type_152);  add_138 = convert_element_type_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_18: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_139);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_38: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_18, arg255_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg255_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_162: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg256_1, torch.float32);  arg256_1 = None
        unsqueeze_432: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_162, -1);  convert_element_type_162 = None
        unsqueeze_433: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        sub_54: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_433);  convolution_38 = unsqueeze_433 = None
        convert_element_type_163: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg257_1, torch.float32);  arg257_1 = None
        add_142: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_163, 1e-05);  convert_element_type_163 = None
        sqrt_54: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_142);  add_142 = None
        reciprocal_54: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_162: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_434: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
        unsqueeze_435: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        mul_163: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
        unsqueeze_436: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
        unsqueeze_437: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_164: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
        unsqueeze_438: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_439: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_143: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
        convert_element_type_164: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_39: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_18, arg260_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg260_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_165: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg261_1, torch.float32);  arg261_1 = None
        unsqueeze_440: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_165, -1);  convert_element_type_165 = None
        unsqueeze_441: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        sub_55: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_441);  convolution_39 = unsqueeze_441 = None
        convert_element_type_166: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg262_1, torch.float32);  arg262_1 = None
        add_144: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_166, 1e-05);  convert_element_type_166 = None
        sqrt_55: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_55: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_165: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_442: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
        unsqueeze_443: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        mul_166: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
        unsqueeze_444: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_445: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_167: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
        unsqueeze_446: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_447: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_145: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
        convert_element_type_167: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.bfloat16);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_146: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_164, convert_element_type_167);  convert_element_type_164 = convert_element_type_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_159: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg251_1, torch.float32);  arg251_1 = None
        unsqueeze_424: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_159, -1);  convert_element_type_159 = None
        unsqueeze_425: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        sub_53: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_18, unsqueeze_425);  relu_18 = unsqueeze_425 = None
        convert_element_type_160: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg252_1, torch.float32);  arg252_1 = None
        add_140: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_160, 1e-05);  convert_element_type_160 = None
        sqrt_53: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
        reciprocal_53: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_159: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_426: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
        unsqueeze_427: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        mul_160: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
        unsqueeze_428: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
        unsqueeze_429: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_161: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
        unsqueeze_430: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_431: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_141: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
        convert_element_type_161: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16);  add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_147: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_146, convert_element_type_161);  add_146 = convert_element_type_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_19: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_147);  add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_40: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_19, arg269_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg269_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_171: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg270_1, torch.float32);  arg270_1 = None
        unsqueeze_456: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_171, -1);  convert_element_type_171 = None
        unsqueeze_457: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        sub_57: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_457);  convolution_40 = unsqueeze_457 = None
        convert_element_type_172: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg271_1, torch.float32);  arg271_1 = None
        add_150: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_172, 1e-05);  convert_element_type_172 = None
        sqrt_57: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_57: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_171: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_458: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_459: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        mul_172: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
        unsqueeze_460: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_461: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_173: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_461);  mul_172 = unsqueeze_461 = None
        unsqueeze_462: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_463: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_151: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_463);  mul_173 = unsqueeze_463 = None
        convert_element_type_173: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_151, torch.bfloat16);  add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_41: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(relu_19, arg274_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg274_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_174: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg275_1, torch.float32);  arg275_1 = None
        unsqueeze_464: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_174, -1);  convert_element_type_174 = None
        unsqueeze_465: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        sub_58: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_465);  convolution_41 = unsqueeze_465 = None
        convert_element_type_175: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg276_1, torch.float32);  arg276_1 = None
        add_152: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_175, 1e-05);  convert_element_type_175 = None
        sqrt_58: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
        reciprocal_58: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_174: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_466: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
        unsqueeze_467: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        mul_175: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_469: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_176: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_469);  mul_175 = unsqueeze_469 = None
        unsqueeze_470: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg278_1, -1);  arg278_1 = None
        unsqueeze_471: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_153: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_471);  mul_176 = unsqueeze_471 = None
        convert_element_type_176: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_153, torch.bfloat16);  add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_154: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_173, convert_element_type_176);  convert_element_type_173 = convert_element_type_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_168: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg265_1, torch.float32);  arg265_1 = None
        unsqueeze_448: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_168, -1);  convert_element_type_168 = None
        unsqueeze_449: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        sub_56: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(relu_19, unsqueeze_449);  relu_19 = unsqueeze_449 = None
        convert_element_type_169: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg266_1, torch.float32);  arg266_1 = None
        add_148: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_169, 1e-05);  convert_element_type_169 = None
        sqrt_56: "f32[384][1]cuda:0" = torch.ops.aten.sqrt.default(add_148);  add_148 = None
        reciprocal_56: "f32[384][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_168: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_450: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
        unsqueeze_451: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        mul_169: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
        unsqueeze_452: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_453: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_170: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_453);  mul_169 = unsqueeze_453 = None
        unsqueeze_454: "bf16[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
        unsqueeze_455: "bf16[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_149: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_455);  mul_170 = unsqueeze_455 = None
        convert_element_type_170: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_155: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_154, convert_element_type_170);  add_154 = convert_element_type_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_20: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.relu.default(add_155);  add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_42: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.convolution.default(relu_20, arg279_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg279_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_177: "f32[1408][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg280_1, torch.float32);  arg280_1 = None
        unsqueeze_472: "f32[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_177, -1);  convert_element_type_177 = None
        unsqueeze_473: "f32[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        sub_59: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_473);  convolution_42 = unsqueeze_473 = None
        convert_element_type_178: "f32[1408][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg281_1, torch.float32);  arg281_1 = None
        add_156: "f32[1408][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_178, 1e-05);  convert_element_type_178 = None
        sqrt_59: "f32[1408][1]cuda:0" = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_59: "f32[1408][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_177: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_474: "f32[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
        unsqueeze_475: "f32[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        mul_178: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476: "bf16[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_477: "bf16[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_179: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_477);  mul_178 = unsqueeze_477 = None
        unsqueeze_478: "bf16[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
        unsqueeze_479: "bf16[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_157: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_479);  mul_179 = unsqueeze_479 = None
        convert_element_type_179: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16);  add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_43: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.convolution.default(relu_20, arg284_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_20 = arg284_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_180: "f32[1408][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg285_1, torch.float32);  arg285_1 = None
        unsqueeze_480: "f32[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_180, -1);  convert_element_type_180 = None
        unsqueeze_481: "f32[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        sub_60: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_481);  convolution_43 = unsqueeze_481 = None
        convert_element_type_181: "f32[1408][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg286_1, torch.float32);  arg286_1 = None
        add_158: "f32[1408][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_181, 1e-05);  convert_element_type_181 = None
        sqrt_60: "f32[1408][1]cuda:0" = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_60: "f32[1408][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_180: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_482: "f32[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
        unsqueeze_483: "f32[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        mul_181: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484: "bf16[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_485: "bf16[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_182: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_485);  mul_181 = unsqueeze_485 = None
        unsqueeze_486: "bf16[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_487: "bf16[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_159: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_487);  mul_182 = unsqueeze_487 = None
        convert_element_type_182: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16);  add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:748 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_160: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_179, convert_element_type_182);  convert_element_type_179 = convert_element_type_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_21: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.relu.default(add_160);  add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean: "bf16[128, 1408, 1, 1][1408, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_21, [-1, -2], True);  relu_21 = None
        as_strided: "bf16[128, 1408, 1, 1][1408, 1, 1408, 1408]cuda:0" = torch.ops.aten.as_strided.default(mean, [128, 1408, 1, 1], [1408, 1, 1408, 1408]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view: "bf16[128, 1408][1408, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided, [128, 1408]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute: "bf16[1408, 1000][1, 1408]cuda:0" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg290_1, view, permute);  arg290_1 = view = permute = None
        return (addmm,)
