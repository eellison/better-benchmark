class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s21)", arg1_1: "bf16[s21, 5][5, 1]cuda:0", arg2_1: "Sym(s21)", arg3_1: "bf16[s21, 3, 128, 128][49152, 16384, 128, 1]cuda:0", arg4_1: "bf16[64, 8, 7, 7][392, 49, 7, 1]cuda:0", arg5_1: "bf16[64][1]cuda:0", arg6_1: "bf16[64][1]cuda:0", arg7_1: "bf16[64][1]cuda:0", arg8_1: "bf16[64][1]cuda:0", arg9_1: "bf16[128, 64, 4, 4][1024, 16, 4, 1]cuda:0", arg10_1: "bf16[128][1]cuda:0", arg11_1: "bf16[128][1]cuda:0", arg12_1: "bf16[128][1]cuda:0", arg13_1: "bf16[128][1]cuda:0", arg14_1: "bf16[256, 128, 4, 4][2048, 16, 4, 1]cuda:0", arg15_1: "bf16[256][1]cuda:0", arg16_1: "bf16[256][1]cuda:0", arg17_1: "bf16[256][1]cuda:0", arg18_1: "bf16[256][1]cuda:0", arg19_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg20_1: "bf16[256][1]cuda:0", arg21_1: "bf16[256][1]cuda:0", arg22_1: "bf16[256][1]cuda:0", arg23_1: "bf16[256][1]cuda:0", arg24_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg25_1: "bf16[256][1]cuda:0", arg26_1: "bf16[256][1]cuda:0", arg27_1: "bf16[256][1]cuda:0", arg28_1: "bf16[256][1]cuda:0", arg29_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg30_1: "bf16[256][1]cuda:0", arg31_1: "bf16[256][1]cuda:0", arg32_1: "bf16[256][1]cuda:0", arg33_1: "bf16[256][1]cuda:0", arg34_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg35_1: "bf16[256][1]cuda:0", arg36_1: "bf16[256][1]cuda:0", arg37_1: "bf16[256][1]cuda:0", arg38_1: "bf16[256][1]cuda:0", arg39_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg40_1: "bf16[256][1]cuda:0", arg41_1: "bf16[256][1]cuda:0", arg42_1: "bf16[256][1]cuda:0", arg43_1: "bf16[256][1]cuda:0", arg44_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg45_1: "bf16[256][1]cuda:0", arg46_1: "bf16[256][1]cuda:0", arg47_1: "bf16[256][1]cuda:0", arg48_1: "bf16[256][1]cuda:0", arg49_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg50_1: "bf16[256][1]cuda:0", arg51_1: "bf16[256][1]cuda:0", arg52_1: "bf16[256][1]cuda:0", arg53_1: "bf16[256][1]cuda:0", arg54_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg55_1: "bf16[256][1]cuda:0", arg56_1: "bf16[256][1]cuda:0", arg57_1: "bf16[256][1]cuda:0", arg58_1: "bf16[256][1]cuda:0", arg59_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg60_1: "bf16[256][1]cuda:0", arg61_1: "bf16[256][1]cuda:0", arg62_1: "bf16[256][1]cuda:0", arg63_1: "bf16[256][1]cuda:0", arg64_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg65_1: "bf16[256][1]cuda:0", arg66_1: "bf16[256][1]cuda:0", arg67_1: "bf16[256][1]cuda:0", arg68_1: "bf16[256][1]cuda:0", arg69_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg70_1: "bf16[256][1]cuda:0", arg71_1: "bf16[256][1]cuda:0", arg72_1: "bf16[256][1]cuda:0", arg73_1: "bf16[256][1]cuda:0", arg74_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg75_1: "bf16[256][1]cuda:0", arg76_1: "bf16[256][1]cuda:0", arg77_1: "bf16[256][1]cuda:0", arg78_1: "bf16[256][1]cuda:0", arg79_1: "bf16[256, 128, 4, 4][2048, 16, 4, 1]cuda:0", arg80_1: "bf16[128][1]cuda:0", arg81_1: "bf16[128][1]cuda:0", arg82_1: "bf16[128][1]cuda:0", arg83_1: "bf16[128][1]cuda:0", arg84_1: "bf16[128, 64, 4, 4][1024, 16, 4, 1]cuda:0", arg85_1: "bf16[64][1]cuda:0", arg86_1: "bf16[64][1]cuda:0", arg87_1: "bf16[64][1]cuda:0", arg88_1: "bf16[64][1]cuda:0", arg89_1: "bf16[3, 64, 7, 7][3136, 49, 7, 1]cuda:0"):
        # File: /torchbench/torchbenchmark/models/pytorch_stargan/model.py:91 in forward, code: c = c.view(c.size(0), c.size(1), 1, 1)
        view: "bf16[s21, 5, 1, 1][5, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(arg1_1, [arg2_1, 5, 1, 1]);  arg1_1 = None

        # File: /torchbench/torchbenchmark/models/pytorch_stargan/model.py:92 in forward, code: c = c.repeat(1, 1, x.size(2), x.size(3))
        repeat: "bf16[s21, 5, 128, 128][81920, 16384, 128, 1]cuda:0" = torch.ops.aten.repeat.default(view, [1, 1, 128, 128]);  view = None

        # File: /torchbench/torchbenchmark/models/pytorch_stargan/model.py:93 in forward, code: x = torch.cat([x, c], dim=1)
        cat: "bf16[s21, 8, 128, 128][131072, 16384, 128, 1]cuda:0" = torch.ops.aten.cat.default([arg3_1, repeat], 1);  arg3_1 = repeat = None

        # File: /torchbench/torchbenchmark/models/pytorch_stargan/model.py:94 in forward, code: return self.main(x)
        convolution: "bf16[s21, 64, 128, 128][1048576, 16384, 128, 1]cuda:0" = torch.ops.aten.convolution.default(cat, arg4_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 1);  cat = arg4_1 = None
        mul_8: "Sym(64 * s21)" = arg2_1 * 64
        view_1: "bf16[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convolution, [1, mul_8, 128, 128]);  convolution = None
        repeat_3: "bf16[64*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg5_1, [arg2_1]);  arg5_1 = None
        convert_element_type: "f32[64*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_3, torch.float32);  repeat_3 = None
        unsqueeze: "f32[64*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[64*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub_9: "f32[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_1, unsqueeze_1);  view_1 = unsqueeze_1 = None
        repeat_4: "bf16[64*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg6_1, [arg2_1]);  arg6_1 = None
        convert_element_type_1: "f32[64*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_4, torch.float32);  repeat_4 = None
        add_32: "f32[64*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
        sqrt: "f32[64*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
        reciprocal: "f32[64*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul_23: "f32[64*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[64*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_23, -1);  mul_23 = None
        unsqueeze_3: "f32[64*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_24: "f32[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_3);  sub_9 = unsqueeze_3 = None
        repeat_1: "bf16[64*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg7_1, [arg2_1]);  arg7_1 = None
        unsqueeze_4: "bf16[64*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_1, -1);  repeat_1 = None
        unsqueeze_5: "bf16[64*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_25: "f32[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_5);  mul_24 = unsqueeze_5 = None
        repeat_2: "bf16[64*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg8_1, [arg2_1]);  arg8_1 = None
        unsqueeze_6: "bf16[64*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_2, -1);  repeat_2 = None
        unsqueeze_7: "bf16[64*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_33: "f32[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, unsqueeze_7);  mul_25 = unsqueeze_7 = None
        convert_element_type_2: "bf16[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None
        view_2: "bf16[s21, 64, 128, 128][1048576, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [arg2_1, 64, 128, 128]);  convert_element_type_2 = None
        relu: "bf16[s21, 64, 128, 128][1048576, 16384, 128, 1]cuda:0" = torch.ops.aten.relu.default(view_2);  view_2 = None
        convolution_1: "bf16[s21, 128, 64, 64][524288, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(relu, arg9_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg9_1 = None
        mul_39: "Sym(128 * s21)" = arg2_1 * 128
        view_5: "bf16[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_1, [1, mul_39, 64, 64]);  convolution_1 = None
        repeat_7: "bf16[128*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg10_1, [arg2_1]);  arg10_1 = None
        convert_element_type_3: "f32[128*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_7, torch.float32);  repeat_7 = None
        unsqueeze_8: "f32[128*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, -1);  convert_element_type_3 = None
        unsqueeze_9: "f32[128*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_22: "f32[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_5, unsqueeze_9);  view_5 = unsqueeze_9 = None
        repeat_8: "bf16[128*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg11_1, [arg2_1]);  arg11_1 = None
        convert_element_type_4: "f32[128*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_8, torch.float32);  repeat_8 = None
        add_74: "f32[128*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_4, 1e-05);  convert_element_type_4 = None
        sqrt_1: "f32[128*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
        reciprocal_1: "f32[128*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_54: "f32[128*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[128*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
        unsqueeze_11: "f32[128*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_55: "f32[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_11);  sub_22 = unsqueeze_11 = None
        repeat_5: "bf16[128*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg12_1, [arg2_1]);  arg12_1 = None
        unsqueeze_12: "bf16[128*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_5, -1);  repeat_5 = None
        unsqueeze_13: "bf16[128*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_56: "f32[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_13);  mul_55 = unsqueeze_13 = None
        repeat_6: "bf16[128*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg13_1, [arg2_1]);  arg13_1 = None
        unsqueeze_14: "bf16[128*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_6, -1);  repeat_6 = None
        unsqueeze_15: "bf16[128*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_75: "f32[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_15);  mul_56 = unsqueeze_15 = None
        convert_element_type_5: "bf16[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16);  add_75 = None
        view_6: "bf16[s21, 128, 64, 64][524288, 4096, 64, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_5, [arg2_1, 128, 64, 64]);  convert_element_type_5 = None
        relu_1: "bf16[s21, 128, 64, 64][524288, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(view_6);  view_6 = None
        convolution_2: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, arg14_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg14_1 = None
        mul_70: "Sym(256 * s21)" = arg2_1 * 256
        view_9: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_2, [1, mul_70, 32, 32]);  convolution_2 = None
        repeat_11: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg15_1, [arg2_1]);  arg15_1 = None
        convert_element_type_6: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_11, torch.float32);  repeat_11 = None
        unsqueeze_16: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
        unsqueeze_17: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_35: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_9, unsqueeze_17);  view_9 = unsqueeze_17 = None
        repeat_12: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg16_1, [arg2_1]);  arg16_1 = None
        convert_element_type_7: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_12, torch.float32);  repeat_12 = None
        add_116: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
        sqrt_2: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_2: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_85: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_85, -1);  mul_85 = None
        unsqueeze_19: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_86: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_19);  sub_35 = unsqueeze_19 = None
        repeat_9: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg17_1, [arg2_1]);  arg17_1 = None
        unsqueeze_20: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_9, -1);  repeat_9 = None
        unsqueeze_21: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_87: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, unsqueeze_21);  mul_86 = unsqueeze_21 = None
        repeat_10: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg18_1, [arg2_1]);  arg18_1 = None
        unsqueeze_22: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_10, -1);  repeat_10 = None
        unsqueeze_23: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_117: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_87, unsqueeze_23);  mul_87 = unsqueeze_23 = None
        convert_element_type_8: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None
        view_10: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_8, [arg2_1, 256, 32, 32]);  convert_element_type_8 = None
        relu_2: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.relu.default(view_10);  view_10 = None

        # File: /torchbench/torchbenchmark/models/pytorch_stargan/model.py:20 in forward, code: return x + self.main(x)
        convolution_3: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, arg19_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg19_1 = None
        view_13: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_3, [1, mul_70, 32, 32]);  convolution_3 = None
        repeat_15: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg20_1, [arg2_1]);  arg20_1 = None
        convert_element_type_9: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_15, torch.float32);  repeat_15 = None
        unsqueeze_24: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_9, -1);  convert_element_type_9 = None
        unsqueeze_25: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_48: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_13, unsqueeze_25);  view_13 = unsqueeze_25 = None
        repeat_16: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg21_1, [arg2_1]);  arg21_1 = None
        convert_element_type_10: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_16, torch.float32);  repeat_16 = None
        add_158: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_10, 1e-05);  convert_element_type_10 = None
        sqrt_3: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_3: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_116: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_116, -1);  mul_116 = None
        unsqueeze_27: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_117: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_27);  sub_48 = unsqueeze_27 = None
        repeat_13: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg22_1, [arg2_1]);  arg22_1 = None
        unsqueeze_28: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_13, -1);  repeat_13 = None
        unsqueeze_29: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_118: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, unsqueeze_29);  mul_117 = unsqueeze_29 = None
        repeat_14: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg23_1, [arg2_1]);  arg23_1 = None
        unsqueeze_30: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_14, -1);  repeat_14 = None
        unsqueeze_31: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_159: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_31);  mul_118 = unsqueeze_31 = None
        convert_element_type_11: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16);  add_159 = None
        view_14: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_11, [arg2_1, 256, 32, 32]);  convert_element_type_11 = None
        relu_3: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.relu.default(view_14);  view_14 = None
        convolution_4: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, arg24_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg24_1 = None
        view_17: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_4, [1, mul_70, 32, 32]);  convolution_4 = None
        repeat_19: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg25_1, [arg2_1]);  arg25_1 = None
        convert_element_type_12: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_19, torch.float32);  repeat_19 = None
        unsqueeze_32: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
        unsqueeze_33: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        sub_61: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_17, unsqueeze_33);  view_17 = unsqueeze_33 = None
        repeat_20: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg26_1, [arg2_1]);  arg26_1 = None
        convert_element_type_13: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_20, torch.float32);  repeat_20 = None
        add_200: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
        sqrt_4: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_200);  add_200 = None
        reciprocal_4: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_147: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_34: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
        unsqueeze_35: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_148: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_35);  sub_61 = unsqueeze_35 = None
        repeat_17: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg27_1, [arg2_1]);  arg27_1 = None
        unsqueeze_36: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_17, -1);  repeat_17 = None
        unsqueeze_37: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_149: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_37);  mul_148 = unsqueeze_37 = None
        repeat_18: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg28_1, [arg2_1]);  arg28_1 = None
        unsqueeze_38: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_18, -1);  repeat_18 = None
        unsqueeze_39: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_201: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_39);  mul_149 = unsqueeze_39 = None
        convert_element_type_14: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_201, torch.bfloat16);  add_201 = None
        view_18: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_14, [arg2_1, 256, 32, 32]);  convert_element_type_14 = None
        add_215: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(relu_2, view_18);  relu_2 = view_18 = None
        convolution_5: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(add_215, arg29_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg29_1 = None
        view_19: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_5, [1, mul_70, 32, 32]);  convolution_5 = None
        repeat_23: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg30_1, [arg2_1]);  arg30_1 = None
        convert_element_type_15: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_23, torch.float32);  repeat_23 = None
        unsqueeze_40: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_15, -1);  convert_element_type_15 = None
        unsqueeze_41: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        sub_73: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_19, unsqueeze_41);  view_19 = unsqueeze_41 = None
        repeat_24: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg31_1, [arg2_1]);  arg31_1 = None
        convert_element_type_16: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_24, torch.float32);  repeat_24 = None
        add_238: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_16, 1e-05);  convert_element_type_16 = None
        sqrt_5: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_238);  add_238 = None
        reciprocal_5: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_176: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_42: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_176, -1);  mul_176 = None
        unsqueeze_43: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_177: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_43);  sub_73 = unsqueeze_43 = None
        repeat_21: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg32_1, [arg2_1]);  arg32_1 = None
        unsqueeze_44: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_21, -1);  repeat_21 = None
        unsqueeze_45: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_178: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, unsqueeze_45);  mul_177 = unsqueeze_45 = None
        repeat_22: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg33_1, [arg2_1]);  arg33_1 = None
        unsqueeze_46: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_22, -1);  repeat_22 = None
        unsqueeze_47: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_239: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_178, unsqueeze_47);  mul_178 = unsqueeze_47 = None
        convert_element_type_17: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_239, torch.bfloat16);  add_239 = None
        view_20: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_17, [arg2_1, 256, 32, 32]);  convert_element_type_17 = None
        relu_4: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.relu.default(view_20);  view_20 = None
        convolution_6: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, arg34_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg34_1 = None
        view_23: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_6, [1, mul_70, 32, 32]);  convolution_6 = None
        repeat_27: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg35_1, [arg2_1]);  arg35_1 = None
        convert_element_type_18: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_27, torch.float32);  repeat_27 = None
        unsqueeze_48: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
        unsqueeze_49: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        sub_86: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_23, unsqueeze_49);  view_23 = unsqueeze_49 = None
        repeat_28: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg36_1, [arg2_1]);  arg36_1 = None
        convert_element_type_19: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_28, torch.float32);  repeat_28 = None
        add_280: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
        sqrt_6: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_280);  add_280 = None
        reciprocal_6: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_207: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_50: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
        unsqueeze_51: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_208: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_51);  sub_86 = unsqueeze_51 = None
        repeat_25: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg37_1, [arg2_1]);  arg37_1 = None
        unsqueeze_52: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_25, -1);  repeat_25 = None
        unsqueeze_53: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_209: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_53);  mul_208 = unsqueeze_53 = None
        repeat_26: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg38_1, [arg2_1]);  arg38_1 = None
        unsqueeze_54: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_26, -1);  repeat_26 = None
        unsqueeze_55: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_281: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_55);  mul_209 = unsqueeze_55 = None
        convert_element_type_20: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_281, torch.bfloat16);  add_281 = None
        view_24: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_20, [arg2_1, 256, 32, 32]);  convert_element_type_20 = None
        add_295: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(add_215, view_24);  add_215 = view_24 = None
        convolution_7: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(add_295, arg39_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg39_1 = None
        view_25: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_7, [1, mul_70, 32, 32]);  convolution_7 = None
        repeat_31: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg40_1, [arg2_1]);  arg40_1 = None
        convert_element_type_21: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_31, torch.float32);  repeat_31 = None
        unsqueeze_56: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_21, -1);  convert_element_type_21 = None
        unsqueeze_57: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        sub_98: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_25, unsqueeze_57);  view_25 = unsqueeze_57 = None
        repeat_32: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg41_1, [arg2_1]);  arg41_1 = None
        convert_element_type_22: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_32, torch.float32);  repeat_32 = None
        add_318: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_22, 1e-05);  convert_element_type_22 = None
        sqrt_7: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_318);  add_318 = None
        reciprocal_7: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_236: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_58: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_236, -1);  mul_236 = None
        unsqueeze_59: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_237: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_59);  sub_98 = unsqueeze_59 = None
        repeat_29: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg42_1, [arg2_1]);  arg42_1 = None
        unsqueeze_60: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_29, -1);  repeat_29 = None
        unsqueeze_61: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_238: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, unsqueeze_61);  mul_237 = unsqueeze_61 = None
        repeat_30: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg43_1, [arg2_1]);  arg43_1 = None
        unsqueeze_62: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_30, -1);  repeat_30 = None
        unsqueeze_63: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_319: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_238, unsqueeze_63);  mul_238 = unsqueeze_63 = None
        convert_element_type_23: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_319, torch.bfloat16);  add_319 = None
        view_26: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_23, [arg2_1, 256, 32, 32]);  convert_element_type_23 = None
        relu_5: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.relu.default(view_26);  view_26 = None
        convolution_8: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, arg44_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg44_1 = None
        view_29: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_8, [1, mul_70, 32, 32]);  convolution_8 = None
        repeat_35: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg45_1, [arg2_1]);  arg45_1 = None
        convert_element_type_24: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_35, torch.float32);  repeat_35 = None
        unsqueeze_64: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
        unsqueeze_65: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        sub_111: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_29, unsqueeze_65);  view_29 = unsqueeze_65 = None
        repeat_36: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg46_1, [arg2_1]);  arg46_1 = None
        convert_element_type_25: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_36, torch.float32);  repeat_36 = None
        add_360: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
        sqrt_8: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_360);  add_360 = None
        reciprocal_8: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_267: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_66: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
        unsqueeze_67: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_268: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_67);  sub_111 = unsqueeze_67 = None
        repeat_33: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg47_1, [arg2_1]);  arg47_1 = None
        unsqueeze_68: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_33, -1);  repeat_33 = None
        unsqueeze_69: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_269: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_69);  mul_268 = unsqueeze_69 = None
        repeat_34: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg48_1, [arg2_1]);  arg48_1 = None
        unsqueeze_70: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_34, -1);  repeat_34 = None
        unsqueeze_71: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_361: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_71);  mul_269 = unsqueeze_71 = None
        convert_element_type_26: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_361, torch.bfloat16);  add_361 = None
        view_30: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_26, [arg2_1, 256, 32, 32]);  convert_element_type_26 = None
        add_375: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(add_295, view_30);  add_295 = view_30 = None
        convolution_9: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(add_375, arg49_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg49_1 = None
        view_31: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_9, [1, mul_70, 32, 32]);  convolution_9 = None
        repeat_39: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg50_1, [arg2_1]);  arg50_1 = None
        convert_element_type_27: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_39, torch.float32);  repeat_39 = None
        unsqueeze_72: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_27, -1);  convert_element_type_27 = None
        unsqueeze_73: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        sub_123: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_31, unsqueeze_73);  view_31 = unsqueeze_73 = None
        repeat_40: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg51_1, [arg2_1]);  arg51_1 = None
        convert_element_type_28: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_40, torch.float32);  repeat_40 = None
        add_398: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_28, 1e-05);  convert_element_type_28 = None
        sqrt_9: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_398);  add_398 = None
        reciprocal_9: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_296: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_74: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_296, -1);  mul_296 = None
        unsqueeze_75: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_297: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_75);  sub_123 = unsqueeze_75 = None
        repeat_37: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg52_1, [arg2_1]);  arg52_1 = None
        unsqueeze_76: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_37, -1);  repeat_37 = None
        unsqueeze_77: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_298: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, unsqueeze_77);  mul_297 = unsqueeze_77 = None
        repeat_38: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg53_1, [arg2_1]);  arg53_1 = None
        unsqueeze_78: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_38, -1);  repeat_38 = None
        unsqueeze_79: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_399: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_79);  mul_298 = unsqueeze_79 = None
        convert_element_type_29: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_399, torch.bfloat16);  add_399 = None
        view_32: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_29, [arg2_1, 256, 32, 32]);  convert_element_type_29 = None
        relu_6: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.relu.default(view_32);  view_32 = None
        convolution_10: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu_6, arg54_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_6 = arg54_1 = None
        view_35: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_10, [1, mul_70, 32, 32]);  convolution_10 = None
        repeat_43: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg55_1, [arg2_1]);  arg55_1 = None
        convert_element_type_30: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_43, torch.float32);  repeat_43 = None
        unsqueeze_80: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
        unsqueeze_81: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        sub_136: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_35, unsqueeze_81);  view_35 = unsqueeze_81 = None
        repeat_44: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg56_1, [arg2_1]);  arg56_1 = None
        convert_element_type_31: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_44, torch.float32);  repeat_44 = None
        add_440: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
        sqrt_10: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_440);  add_440 = None
        reciprocal_10: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_327: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_82: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
        unsqueeze_83: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_328: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_83);  sub_136 = unsqueeze_83 = None
        repeat_41: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg57_1, [arg2_1]);  arg57_1 = None
        unsqueeze_84: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_41, -1);  repeat_41 = None
        unsqueeze_85: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_329: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_85);  mul_328 = unsqueeze_85 = None
        repeat_42: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg58_1, [arg2_1]);  arg58_1 = None
        unsqueeze_86: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_42, -1);  repeat_42 = None
        unsqueeze_87: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_441: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_329, unsqueeze_87);  mul_329 = unsqueeze_87 = None
        convert_element_type_32: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_441, torch.bfloat16);  add_441 = None
        view_36: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_32, [arg2_1, 256, 32, 32]);  convert_element_type_32 = None
        add_455: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(add_375, view_36);  add_375 = view_36 = None
        convolution_11: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(add_455, arg59_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg59_1 = None
        view_37: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_11, [1, mul_70, 32, 32]);  convolution_11 = None
        repeat_47: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg60_1, [arg2_1]);  arg60_1 = None
        convert_element_type_33: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_47, torch.float32);  repeat_47 = None
        unsqueeze_88: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_33, -1);  convert_element_type_33 = None
        unsqueeze_89: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        sub_148: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_37, unsqueeze_89);  view_37 = unsqueeze_89 = None
        repeat_48: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg61_1, [arg2_1]);  arg61_1 = None
        convert_element_type_34: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_48, torch.float32);  repeat_48 = None
        add_478: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_34, 1e-05);  convert_element_type_34 = None
        sqrt_11: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_478);  add_478 = None
        reciprocal_11: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_356: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_90: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_356, -1);  mul_356 = None
        unsqueeze_91: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_357: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_91);  sub_148 = unsqueeze_91 = None
        repeat_45: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg62_1, [arg2_1]);  arg62_1 = None
        unsqueeze_92: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_45, -1);  repeat_45 = None
        unsqueeze_93: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_358: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_93);  mul_357 = unsqueeze_93 = None
        repeat_46: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg63_1, [arg2_1]);  arg63_1 = None
        unsqueeze_94: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_46, -1);  repeat_46 = None
        unsqueeze_95: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_479: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_358, unsqueeze_95);  mul_358 = unsqueeze_95 = None
        convert_element_type_35: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_479, torch.bfloat16);  add_479 = None
        view_38: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [arg2_1, 256, 32, 32]);  convert_element_type_35 = None
        relu_7: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.relu.default(view_38);  view_38 = None
        convolution_12: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, arg64_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg64_1 = None
        view_41: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_12, [1, mul_70, 32, 32]);  convolution_12 = None
        repeat_51: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg65_1, [arg2_1]);  arg65_1 = None
        convert_element_type_36: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_51, torch.float32);  repeat_51 = None
        unsqueeze_96: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
        unsqueeze_97: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        sub_161: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_41, unsqueeze_97);  view_41 = unsqueeze_97 = None
        repeat_52: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg66_1, [arg2_1]);  arg66_1 = None
        convert_element_type_37: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_52, torch.float32);  repeat_52 = None
        add_520: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
        sqrt_12: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_520);  add_520 = None
        reciprocal_12: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_387: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_98: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_387, -1);  mul_387 = None
        unsqueeze_99: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_388: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_99);  sub_161 = unsqueeze_99 = None
        repeat_49: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg67_1, [arg2_1]);  arg67_1 = None
        unsqueeze_100: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_49, -1);  repeat_49 = None
        unsqueeze_101: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_389: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_101);  mul_388 = unsqueeze_101 = None
        repeat_50: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg68_1, [arg2_1]);  arg68_1 = None
        unsqueeze_102: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_50, -1);  repeat_50 = None
        unsqueeze_103: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_521: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_389, unsqueeze_103);  mul_389 = unsqueeze_103 = None
        convert_element_type_38: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_521, torch.bfloat16);  add_521 = None
        view_42: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_38, [arg2_1, 256, 32, 32]);  convert_element_type_38 = None
        add_535: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(add_455, view_42);  add_455 = view_42 = None
        convolution_13: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(add_535, arg69_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg69_1 = None
        view_43: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_13, [1, mul_70, 32, 32]);  convolution_13 = None
        repeat_55: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg70_1, [arg2_1]);  arg70_1 = None
        convert_element_type_39: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_55, torch.float32);  repeat_55 = None
        unsqueeze_104: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_39, -1);  convert_element_type_39 = None
        unsqueeze_105: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        sub_173: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_43, unsqueeze_105);  view_43 = unsqueeze_105 = None
        repeat_56: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg71_1, [arg2_1]);  arg71_1 = None
        convert_element_type_40: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_56, torch.float32);  repeat_56 = None
        add_558: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_40, 1e-05);  convert_element_type_40 = None
        sqrt_13: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_558);  add_558 = None
        reciprocal_13: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_416: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_106: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_416, -1);  mul_416 = None
        unsqueeze_107: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_417: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_107);  sub_173 = unsqueeze_107 = None
        repeat_53: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg72_1, [arg2_1]);  arg72_1 = None
        unsqueeze_108: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_53, -1);  repeat_53 = None
        unsqueeze_109: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_418: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_417, unsqueeze_109);  mul_417 = unsqueeze_109 = None
        repeat_54: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg73_1, [arg2_1]);  arg73_1 = None
        unsqueeze_110: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_54, -1);  repeat_54 = None
        unsqueeze_111: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_559: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_418, unsqueeze_111);  mul_418 = unsqueeze_111 = None
        convert_element_type_41: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_559, torch.bfloat16);  add_559 = None
        view_44: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_41, [arg2_1, 256, 32, 32]);  convert_element_type_41 = None
        relu_8: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.relu.default(view_44);  view_44 = None
        convolution_14: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, arg74_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_8 = arg74_1 = None
        view_47: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_14, [1, mul_70, 32, 32]);  convolution_14 = mul_70 = None
        repeat_59: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg75_1, [arg2_1]);  arg75_1 = None
        convert_element_type_42: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_59, torch.float32);  repeat_59 = None
        unsqueeze_112: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
        unsqueeze_113: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        sub_186: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_47, unsqueeze_113);  view_47 = unsqueeze_113 = None
        repeat_60: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg76_1, [arg2_1]);  arg76_1 = None
        convert_element_type_43: "f32[256*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_60, torch.float32);  repeat_60 = None
        add_600: "f32[256*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
        sqrt_14: "f32[256*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_600);  add_600 = None
        reciprocal_14: "f32[256*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_447: "f32[256*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_114: "f32[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_447, -1);  mul_447 = None
        unsqueeze_115: "f32[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_448: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_115);  sub_186 = unsqueeze_115 = None
        repeat_57: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg77_1, [arg2_1]);  arg77_1 = None
        unsqueeze_116: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_57, -1);  repeat_57 = None
        unsqueeze_117: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_449: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_117);  mul_448 = unsqueeze_117 = None
        repeat_58: "bf16[256*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg78_1, [arg2_1]);  arg78_1 = None
        unsqueeze_118: "bf16[256*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_58, -1);  repeat_58 = None
        unsqueeze_119: "bf16[256*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_601: "f32[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_449, unsqueeze_119);  mul_449 = unsqueeze_119 = None
        convert_element_type_44: "bf16[1, 256*s21, 32, 32][262144*s21, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_601, torch.bfloat16);  add_601 = None
        view_48: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_44, [arg2_1, 256, 32, 32]);  convert_element_type_44 = None
        add_615: "bf16[s21, 256, 32, 32][262144, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(add_535, view_48);  add_535 = view_48 = None

        # File: /torchbench/torchbenchmark/models/pytorch_stargan/model.py:94 in forward, code: return self.main(x)
        convolution_15: "bf16[s21, 128, 64, 64][524288, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(add_615, arg79_1, None, [2, 2], [1, 1], [1, 1], True, [0, 0], 1);  add_615 = arg79_1 = None
        view_49: "bf16[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_15, [1, mul_39, 64, 64]);  convolution_15 = mul_39 = None
        repeat_63: "bf16[128*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg80_1, [arg2_1]);  arg80_1 = None
        convert_element_type_45: "f32[128*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_63, torch.float32);  repeat_63 = None
        unsqueeze_120: "f32[128*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_45, -1);  convert_element_type_45 = None
        unsqueeze_121: "f32[128*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        sub_198: "f32[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_49, unsqueeze_121);  view_49 = unsqueeze_121 = None
        repeat_64: "bf16[128*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg81_1, [arg2_1]);  arg81_1 = None
        convert_element_type_46: "f32[128*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_64, torch.float32);  repeat_64 = None
        add_638: "f32[128*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_46, 1e-05);  convert_element_type_46 = None
        sqrt_15: "f32[128*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_638);  add_638 = None
        reciprocal_15: "f32[128*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_476: "f32[128*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_122: "f32[128*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_476, -1);  mul_476 = None
        unsqueeze_123: "f32[128*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_477: "f32[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_123);  sub_198 = unsqueeze_123 = None
        repeat_61: "bf16[128*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg82_1, [arg2_1]);  arg82_1 = None
        unsqueeze_124: "bf16[128*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_61, -1);  repeat_61 = None
        unsqueeze_125: "bf16[128*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_478: "f32[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_477, unsqueeze_125);  mul_477 = unsqueeze_125 = None
        repeat_62: "bf16[128*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg83_1, [arg2_1]);  arg83_1 = None
        unsqueeze_126: "bf16[128*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_62, -1);  repeat_62 = None
        unsqueeze_127: "bf16[128*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_639: "f32[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_478, unsqueeze_127);  mul_478 = unsqueeze_127 = None
        convert_element_type_47: "bf16[1, 128*s21, 64, 64][524288*s21, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_639, torch.bfloat16);  add_639 = None
        view_50: "bf16[s21, 128, 64, 64][524288, 4096, 64, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_47, [arg2_1, 128, 64, 64]);  convert_element_type_47 = None
        relu_9: "bf16[s21, 128, 64, 64][524288, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(view_50);  view_50 = None
        convolution_16: "bf16[s21, 64, 128, 128][1048576, 16384, 128, 1]cuda:0" = torch.ops.aten.convolution.default(relu_9, arg84_1, None, [2, 2], [1, 1], [1, 1], True, [0, 0], 1);  relu_9 = arg84_1 = None
        view_53: "bf16[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_16, [1, mul_8, 128, 128]);  convolution_16 = mul_8 = None
        repeat_67: "bf16[64*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg85_1, [arg2_1]);  arg85_1 = None
        convert_element_type_48: "f32[64*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_67, torch.float32);  repeat_67 = None
        unsqueeze_128: "f32[64*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
        unsqueeze_129: "f32[64*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        sub_211: "f32[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_53, unsqueeze_129);  view_53 = unsqueeze_129 = None
        repeat_68: "bf16[64*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg86_1, [arg2_1]);  arg86_1 = None
        convert_element_type_49: "f32[64*s21][1]cuda:0" = torch.ops.prims.convert_element_type.default(repeat_68, torch.float32);  repeat_68 = None
        add_680: "f32[64*s21][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
        sqrt_16: "f32[64*s21][1]cuda:0" = torch.ops.aten.sqrt.default(add_680);  add_680 = None
        reciprocal_16: "f32[64*s21][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_507: "f32[64*s21][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_130: "f32[64*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_507, -1);  mul_507 = None
        unsqueeze_131: "f32[64*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_508: "f32[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_131);  sub_211 = unsqueeze_131 = None
        repeat_65: "bf16[64*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg87_1, [arg2_1]);  arg87_1 = None
        unsqueeze_132: "bf16[64*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_65, -1);  repeat_65 = None
        unsqueeze_133: "bf16[64*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_509: "f32[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_508, unsqueeze_133);  mul_508 = unsqueeze_133 = None
        repeat_66: "bf16[64*s21][1]cuda:0" = torch.ops.aten.repeat.default(arg88_1, [arg2_1]);  arg88_1 = None
        unsqueeze_134: "bf16[64*s21, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat_66, -1);  repeat_66 = None
        unsqueeze_135: "bf16[64*s21, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_681: "f32[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_509, unsqueeze_135);  mul_509 = unsqueeze_135 = None
        convert_element_type_50: "bf16[1, 64*s21, 128, 128][1048576*s21, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_681, torch.bfloat16);  add_681 = None
        view_54: "bf16[s21, 64, 128, 128][1048576, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_50, [arg2_1, 64, 128, 128]);  convert_element_type_50 = arg2_1 = None
        relu_10: "bf16[s21, 64, 128, 128][1048576, 16384, 128, 1]cuda:0" = torch.ops.aten.relu.default(view_54);  view_54 = None
        convolution_17: "bf16[s21, 3, 128, 128][49152, 16384, 128, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, arg89_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 1);  relu_10 = arg89_1 = None
        tanh: "bf16[s21, 3, 128, 128][49152, 16384, 128, 1]cuda:0" = torch.ops.aten.tanh.default(convolution_17);  convolution_17 = None
        return (tanh,)
