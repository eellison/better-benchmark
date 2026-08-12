class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 3, 4, 4][48, 16, 4, 1]cuda:0", arg1_1: "bf16[256, 3, 64, 64][12288, 4096, 64, 1]cuda:0", arg2_1: "bf16[128, 64, 4, 4][1024, 16, 4, 1]cuda:0", arg3_1: "bf16[128][1]cuda:0", arg4_1: "bf16[128][1]cuda:0", arg5_1: "bf16[128][1]cuda:0", arg6_1: "bf16[128][1]cuda:0", arg7_1: "bf16[256, 128, 4, 4][2048, 16, 4, 1]cuda:0", arg8_1: "bf16[256][1]cuda:0", arg9_1: "bf16[256][1]cuda:0", arg10_1: "bf16[256][1]cuda:0", arg11_1: "bf16[256][1]cuda:0", arg12_1: "bf16[512, 256, 4, 4][4096, 16, 4, 1]cuda:0", arg13_1: "bf16[512][1]cuda:0", arg14_1: "bf16[512][1]cuda:0", arg15_1: "bf16[512][1]cuda:0", arg16_1: "bf16[512][1]cuda:0", arg17_1: "bf16[1, 512, 4, 4][8192, 16, 4, 1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dcgan/__init__.py:128 in forward, code: return self.main(input)
        convolution: "bf16[256, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        convert_element_type: "f32[256, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        gt: "b8[256, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type, 0)
        mul: "f32[256, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, 0.2)
        where: "f32[256, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(gt, convert_element_type, mul);  gt = convert_element_type = mul = None
        convert_element_type_1: "bf16[256, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        convolution_1: "bf16[256, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, arg2_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_1 = arg2_1 = None
        convert_element_type_2: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        unsqueeze: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
        unsqueeze_1: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[256, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1);  convolution_1 = unsqueeze_1 = None
        convert_element_type_3: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        add: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
        sqrt: "f32[128][1]cuda:0" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[128][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul_1: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1, -1);  mul_1 = None
        unsqueeze_3: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_2: "f32[256, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_5: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_3: "f32[256, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, unsqueeze_5);  mul_2 = unsqueeze_5 = None
        unsqueeze_6: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_7: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[256, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, unsqueeze_7);  mul_3 = unsqueeze_7 = None
        convert_element_type_4: "bf16[256, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_5: "f32[256, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        gt_1: "b8[256, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_5, 0)
        mul_4: "f32[256, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.2)
        where_1: "f32[256, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(gt_1, convert_element_type_5, mul_4);  gt_1 = convert_element_type_5 = mul_4 = None
        convert_element_type_6: "bf16[256, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        convolution_2: "bf16[256, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_6, arg7_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_6 = arg7_1 = None
        convert_element_type_7: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        unsqueeze_8: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_7, -1);  convert_element_type_7 = None
        unsqueeze_9: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[256, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_9);  convolution_2 = unsqueeze_9 = None
        convert_element_type_8: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        add_2: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_8, 1e-05);  convert_element_type_8 = None
        sqrt_1: "f32[256][1]cuda:0" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal_1: "f32[256][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_5: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_5, -1);  mul_5 = None
        unsqueeze_11: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_6: "f32[256, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_13: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_7: "f32[256, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, unsqueeze_13);  mul_6 = unsqueeze_13 = None
        unsqueeze_14: "bf16[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_15: "bf16[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_3: "f32[256, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_7, unsqueeze_15);  mul_7 = unsqueeze_15 = None
        convert_element_type_9: "bf16[256, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        convert_element_type_10: "f32[256, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
        gt_2: "b8[256, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_10, 0)
        mul_8: "f32[256, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.2)
        where_2: "f32[256, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(gt_2, convert_element_type_10, mul_8);  gt_2 = convert_element_type_10 = mul_8 = None
        convert_element_type_11: "bf16[256, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.bfloat16);  where_2 = None
        convolution_3: "bf16[256, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_11, arg12_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_11 = arg12_1 = None
        convert_element_type_12: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        unsqueeze_16: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
        unsqueeze_17: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_2: "f32[256, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_17);  convolution_3 = unsqueeze_17 = None
        convert_element_type_13: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg14_1, torch.float32);  arg14_1 = None
        add_4: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
        sqrt_2: "f32[512][1]cuda:0" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
        reciprocal_2: "f32[512][1]cuda:0" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_9: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_19: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_10: "f32[256, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20: "bf16[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_21: "bf16[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_11: "f32[256, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_21);  mul_10 = unsqueeze_21 = None
        unsqueeze_22: "bf16[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_23: "bf16[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_5: "f32[256, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_23);  mul_11 = unsqueeze_23 = None
        convert_element_type_14: "bf16[256, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        convert_element_type_15: "f32[256, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        gt_3: "b8[256, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_15, 0)
        mul_12: "f32[256, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.2)
        where_3: "f32[256, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(gt_3, convert_element_type_15, mul_12);  gt_3 = convert_element_type_15 = mul_12 = None
        convert_element_type_16: "bf16[256, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.bfloat16);  where_3 = None
        convolution_4: "bf16[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_16, arg17_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_16 = arg17_1 = None
        sigmoid: "bf16[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sigmoid.default(convolution_4);  convolution_4 = None
        return (sigmoid,)
