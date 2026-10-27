class GraphModule(torch.nn.Module):
    def forward(self, primals_7: "f32[128][1]cuda:0", primals_13: "f32[256][1]cuda:0", primals_19: "f32[512][1]cuda:0", convert_element_type: "bf16[64, 3, 4, 4][48, 16, 4, 1]cuda:0", convert_element_type_1: "bf16[32, 3, 64, 64][12288, 4096, 64, 1]cuda:0", convert_element_type_3: "bf16[32, 64, 32, 32][65536, 1024, 32, 1]cuda:0", convert_element_type_4: "bf16[128, 64, 4, 4][1024, 16, 4, 1]cuda:0", convolution_1: "bf16[32, 128, 16, 16][32768, 256, 16, 1]cuda:0", squeeze_1: "f32[128][1]cuda:0", convert_element_type_8: "bf16[32, 128, 16, 16][32768, 256, 16, 1]cuda:0", convert_element_type_9: "bf16[256, 128, 4, 4][2048, 16, 4, 1]cuda:0", convolution_2: "bf16[32, 256, 8, 8][16384, 64, 8, 1]cuda:0", squeeze_4: "f32[256][1]cuda:0", convert_element_type_13: "bf16[32, 256, 8, 8][16384, 64, 8, 1]cuda:0", convert_element_type_14: "bf16[512, 256, 4, 4][4096, 16, 4, 1]cuda:0", convolution_3: "bf16[32, 512, 4, 4][8192, 16, 4, 1]cuda:0", squeeze_7: "f32[512][1]cuda:0", convert_element_type_18: "bf16[32, 512, 4, 4][8192, 16, 4, 1]cuda:0", convert_element_type_19: "bf16[1, 512, 4, 4][8192, 16, 4, 1]cuda:0", sigmoid: "bf16[32, 1, 1, 1][1, 1, 1, 1]cuda:0", unsqueeze_14: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_26: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_38: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", tangents_1: "bf16[32, 1, 1, 1][1, 1, 1, 1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dcgan/__init__.py:128 in forward, code: return self.main(input)
        convert_element_type_20: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
        convert_element_type_21: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32);  sigmoid = None
        sub_3: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_21)
        mul_25: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_21, sub_3);  convert_element_type_21 = sub_3 = None
        mul_26: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, mul_25);  convert_element_type_20 = mul_25 = None
        convert_element_type_22: "bf16[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_22, convert_element_type_18, convert_element_type_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_22 = convert_element_type_19 = None
        getitem_6: "bf16[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = convolution_backward[0]
        getitem_7: "bf16[1, 512, 4, 4][8192, 16, 4, 1]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_23: "f32[1, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
        convert_element_type_24: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_6, torch.float32);  getitem_6 = None
        convert_element_type_25: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_18, torch.float32);  convert_element_type_18 = None
        gt_4: "b8[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_25, 0);  convert_element_type_25 = None
        mul_27: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.2)
        where_4: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(gt_4, convert_element_type_24, mul_27);  gt_4 = convert_element_type_24 = mul_27 = None
        convert_element_type_26: "bf16[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.bfloat16);  where_4 = None
        convert_element_type_27: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_26, torch.float32);  convert_element_type_26 = None
        sum_1: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_27, [0, 2, 3])
        convert_element_type_15: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_4: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_15, unsqueeze_14);  convert_element_type_15 = unsqueeze_14 = None
        mul_28: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_27, sub_4)
        sum_2: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_28, [0, 2, 3]);  mul_28 = None
        mul_29: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_1, 0.001953125)
        unsqueeze_15: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_29, 0);  mul_29 = None
        unsqueeze_16: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None
        unsqueeze_17: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 3);  unsqueeze_16 = None
        mul_30: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.001953125)
        mul_31: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_32: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
        unsqueeze_18: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_32, 0);  mul_32 = None
        unsqueeze_19: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2);  unsqueeze_18 = None
        unsqueeze_20: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 3);  unsqueeze_19 = None
        mul_33: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_19);  primals_19 = None
        unsqueeze_21: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_33, 0);  mul_33 = None
        unsqueeze_22: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 2);  unsqueeze_21 = None
        unsqueeze_23: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 3);  unsqueeze_22 = None
        mul_34: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_20);  sub_4 = unsqueeze_20 = None
        sub_6: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_27, mul_34);  convert_element_type_27 = mul_34 = None
        sub_7: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_6, unsqueeze_17);  sub_6 = unsqueeze_17 = None
        mul_35: "f32[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_23);  sub_7 = unsqueeze_23 = None
        mul_36: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, squeeze_7);  sum_2 = squeeze_7 = None
        convert_element_type_29: "bf16[32, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_29, convert_element_type_13, convert_element_type_14, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_29 = convert_element_type_14 = None
        getitem_9: "bf16[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = convolution_backward_1[0]
        getitem_10: "bf16[512, 256, 4, 4][4096, 16, 4, 1]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_30: "f32[512, 256, 4, 4][4096, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
        convert_element_type_31: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_9, torch.float32);  getitem_9 = None
        convert_element_type_32: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        gt_5: "b8[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_32, 0);  convert_element_type_32 = None
        mul_37: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_31, 0.2)
        where_5: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(gt_5, convert_element_type_31, mul_37);  gt_5 = convert_element_type_31 = mul_37 = None
        convert_element_type_33: "bf16[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.bfloat16);  where_5 = None
        convert_element_type_34: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_33, torch.float32);  convert_element_type_33 = None
        sum_3: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_34, [0, 2, 3])
        convert_element_type_10: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sub_8: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_10, unsqueeze_26);  convert_element_type_10 = unsqueeze_26 = None
        mul_38: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, sub_8)
        sum_4: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_38, [0, 2, 3]);  mul_38 = None
        mul_39: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.00048828125)
        unsqueeze_27: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_39, 0);  mul_39 = None
        unsqueeze_28: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 2);  unsqueeze_27 = None
        unsqueeze_29: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 3);  unsqueeze_28 = None
        mul_40: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.00048828125)
        mul_41: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_42: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
        unsqueeze_30: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_42, 0);  mul_42 = None
        unsqueeze_31: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 2);  unsqueeze_30 = None
        unsqueeze_32: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_31, 3);  unsqueeze_31 = None
        mul_43: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_13);  primals_13 = None
        unsqueeze_33: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_43, 0);  mul_43 = None
        unsqueeze_34: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 2);  unsqueeze_33 = None
        unsqueeze_35: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 3);  unsqueeze_34 = None
        mul_44: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_32);  sub_8 = unsqueeze_32 = None
        sub_10: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_34, mul_44);  convert_element_type_34 = mul_44 = None
        sub_11: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_10, unsqueeze_29);  sub_10 = unsqueeze_29 = None
        mul_45: "f32[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_35);  sub_11 = unsqueeze_35 = None
        mul_46: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, squeeze_4);  sum_4 = squeeze_4 = None
        convert_element_type_36: "bf16[32, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_36, convert_element_type_8, convert_element_type_9, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_36 = convert_element_type_9 = None
        getitem_12: "bf16[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = convolution_backward_2[0]
        getitem_13: "bf16[256, 128, 4, 4][2048, 16, 4, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_37: "f32[256, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
        convert_element_type_38: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_12, torch.float32);  getitem_12 = None
        convert_element_type_39: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        gt_6: "b8[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_39, 0);  convert_element_type_39 = None
        mul_47: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.2)
        where_6: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(gt_6, convert_element_type_38, mul_47);  gt_6 = convert_element_type_38 = mul_47 = None
        convert_element_type_40: "bf16[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.bfloat16);  where_6 = None
        convert_element_type_41: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_40, torch.float32);  convert_element_type_40 = None
        sum_5: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_41, [0, 2, 3])
        convert_element_type_5: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_12: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_5, unsqueeze_38);  convert_element_type_5 = unsqueeze_38 = None
        mul_48: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_41, sub_12)
        sum_6: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_48, [0, 2, 3]);  mul_48 = None
        mul_49: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.0001220703125)
        unsqueeze_39: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_49, 0);  mul_49 = None
        unsqueeze_40: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 2);  unsqueeze_39 = None
        unsqueeze_41: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 3);  unsqueeze_40 = None
        mul_50: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.0001220703125)
        mul_51: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_52: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        unsqueeze_42: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_52, 0);  mul_52 = None
        unsqueeze_43: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
        unsqueeze_44: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 3);  unsqueeze_43 = None
        mul_53: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_7);  primals_7 = None
        unsqueeze_45: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_53, 0);  mul_53 = None
        unsqueeze_46: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 2);  unsqueeze_45 = None
        unsqueeze_47: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 3);  unsqueeze_46 = None
        mul_54: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_44);  sub_12 = unsqueeze_44 = None
        sub_14: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_41, mul_54);  convert_element_type_41 = mul_54 = None
        sub_15: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_14, unsqueeze_41);  sub_14 = unsqueeze_41 = None
        mul_55: "f32[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_47);  sub_15 = unsqueeze_47 = None
        mul_56: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, squeeze_1);  sum_6 = squeeze_1 = None
        convert_element_type_43: "bf16[32, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_43, convert_element_type_3, convert_element_type_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_43 = convert_element_type_4 = None
        getitem_15: "bf16[32, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = convolution_backward_3[0]
        getitem_16: "bf16[128, 64, 4, 4][1024, 16, 4, 1]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_44: "f32[128, 64, 4, 4][1024, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_16, torch.float32);  getitem_16 = None
        convert_element_type_45: "f32[32, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_15, torch.float32);  getitem_15 = None
        convert_element_type_46: "f32[32, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        gt_7: "b8[32, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_46, 0);  convert_element_type_46 = None
        mul_57: "f32[32, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_45, 0.2)
        where_7: "f32[32, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(gt_7, convert_element_type_45, mul_57);  gt_7 = convert_element_type_45 = mul_57 = None
        convert_element_type_47: "bf16[32, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.bfloat16);  where_7 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_47, convert_element_type_1, convert_element_type, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_47 = convert_element_type_1 = convert_element_type = None
        getitem_19: "bf16[64, 3, 4, 4][48, 16, 4, 1]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_48: "f32[64, 3, 4, 4][48, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
        return (convert_element_type_48, None, convert_element_type_44, None, None, None, mul_56, sum_5, convert_element_type_37, None, None, None, mul_46, sum_3, convert_element_type_30, None, None, None, mul_36, sum_1, convert_element_type_23)
