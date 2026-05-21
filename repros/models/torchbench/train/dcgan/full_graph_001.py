class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 4, 4]", primals_2: "f32[1024, 3, 64, 64]", primals_3: "f32[128, 64, 4, 4]", primals_7: "f32[128]", primals_9: "f32[256, 128, 4, 4]", primals_13: "f32[256]", primals_15: "f32[512, 256, 4, 4]", primals_19: "f32[512]", primals_21: "f32[1, 512, 4, 4]", where: "f32[1024, 64, 32, 32]", convolution_1: "f32[1024, 128, 16, 16]", squeeze_1: "f32[128]", where_1: "f32[1024, 128, 16, 16]", convolution_2: "f32[1024, 256, 8, 8]", squeeze_4: "f32[256]", where_2: "f32[1024, 256, 8, 8]", convolution_3: "f32[1024, 512, 4, 4]", squeeze_7: "f32[512]", where_3: "f32[1024, 512, 4, 4]", sigmoid: "f32[1024, 1, 1, 1]", unsqueeze_14: "f32[1, 512, 1, 1]", unsqueeze_26: "f32[1, 256, 1, 1]", unsqueeze_38: "f32[1, 128, 1, 1]", tangents_1: "f32[1024, 1, 1, 1]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dcgan/__init__.py:128 in forward, code: return self.main(input)
        sub_3: "f32[1024, 1, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_25: "f32[1024, 1, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid, sub_3);  sigmoid = sub_3 = None
        mul_26: "f32[1024, 1, 1, 1]" = torch.ops.aten.mul.Tensor(tangents_1, mul_25);  tangents_1 = mul_25 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_26, where_3, primals_21, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_26 = primals_21 = None
        getitem_6: "f32[1024, 512, 4, 4]" = convolution_backward[0]
        getitem_7: "f32[1, 512, 4, 4]" = convolution_backward[1];  convolution_backward = None
        gt_4: "b8[1024, 512, 4, 4]" = torch.ops.aten.gt.Scalar(where_3, 0);  where_3 = None
        mul_27: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(getitem_6, 0.2)
        where_4: "f32[1024, 512, 4, 4]" = torch.ops.aten.where.self(gt_4, getitem_6, mul_27);  gt_4 = getitem_6 = mul_27 = None
        sum_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_4: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_14);  convolution_3 = unsqueeze_14 = None
        mul_28: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(where_4, sub_4)
        sum_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_28, [0, 2, 3]);  mul_28 = None
        mul_29: "f32[512]" = torch.ops.aten.mul.Tensor(sum_1, 6.103515625e-05)
        unsqueeze_15: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_29, 0);  mul_29 = None
        unsqueeze_16: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None
        unsqueeze_17: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 3);  unsqueeze_16 = None
        mul_30: "f32[512]" = torch.ops.aten.mul.Tensor(sum_2, 6.103515625e-05)
        mul_31: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_32: "f32[512]" = torch.ops.aten.mul.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
        unsqueeze_18: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_32, 0);  mul_32 = None
        unsqueeze_19: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2);  unsqueeze_18 = None
        unsqueeze_20: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 3);  unsqueeze_19 = None
        mul_33: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_19);  primals_19 = None
        unsqueeze_21: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_33, 0);  mul_33 = None
        unsqueeze_22: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 2);  unsqueeze_21 = None
        unsqueeze_23: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 3);  unsqueeze_22 = None
        mul_34: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_20);  sub_4 = unsqueeze_20 = None
        sub_6: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(where_4, mul_34);  where_4 = mul_34 = None
        sub_7: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(sub_6, unsqueeze_17);  sub_6 = unsqueeze_17 = None
        mul_35: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_23);  sub_7 = unsqueeze_23 = None
        mul_36: "f32[512]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_7);  sum_2 = squeeze_7 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_35, where_2, primals_15, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_35 = primals_15 = None
        getitem_9: "f32[1024, 256, 8, 8]" = convolution_backward_1[0]
        getitem_10: "f32[512, 256, 4, 4]" = convolution_backward_1[1];  convolution_backward_1 = None
        gt_5: "b8[1024, 256, 8, 8]" = torch.ops.aten.gt.Scalar(where_2, 0);  where_2 = None
        mul_37: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_9, 0.2)
        where_5: "f32[1024, 256, 8, 8]" = torch.ops.aten.where.self(gt_5, getitem_9, mul_37);  gt_5 = getitem_9 = mul_37 = None
        sum_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_8: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_26);  convolution_2 = unsqueeze_26 = None
        mul_38: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(where_5, sub_8)
        sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_38, [0, 2, 3]);  mul_38 = None
        mul_39: "f32[256]" = torch.ops.aten.mul.Tensor(sum_3, 1.52587890625e-05)
        unsqueeze_27: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_39, 0);  mul_39 = None
        unsqueeze_28: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 2);  unsqueeze_27 = None
        unsqueeze_29: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 3);  unsqueeze_28 = None
        mul_40: "f32[256]" = torch.ops.aten.mul.Tensor(sum_4, 1.52587890625e-05)
        mul_41: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_42: "f32[256]" = torch.ops.aten.mul.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
        unsqueeze_30: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_42, 0);  mul_42 = None
        unsqueeze_31: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 2);  unsqueeze_30 = None
        unsqueeze_32: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, 3);  unsqueeze_31 = None
        mul_43: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_13);  primals_13 = None
        unsqueeze_33: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_43, 0);  mul_43 = None
        unsqueeze_34: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 2);  unsqueeze_33 = None
        unsqueeze_35: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 3);  unsqueeze_34 = None
        mul_44: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_32);  sub_8 = unsqueeze_32 = None
        sub_10: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(where_5, mul_44);  where_5 = mul_44 = None
        sub_11: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(sub_10, unsqueeze_29);  sub_10 = unsqueeze_29 = None
        mul_45: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_35);  sub_11 = unsqueeze_35 = None
        mul_46: "f32[256]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_4);  sum_4 = squeeze_4 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_45, where_1, primals_9, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_45 = primals_9 = None
        getitem_12: "f32[1024, 128, 16, 16]" = convolution_backward_2[0]
        getitem_13: "f32[256, 128, 4, 4]" = convolution_backward_2[1];  convolution_backward_2 = None
        gt_6: "b8[1024, 128, 16, 16]" = torch.ops.aten.gt.Scalar(where_1, 0);  where_1 = None
        mul_47: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_12, 0.2)
        where_6: "f32[1024, 128, 16, 16]" = torch.ops.aten.where.self(gt_6, getitem_12, mul_47);  gt_6 = getitem_12 = mul_47 = None
        sum_5: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_12: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_38);  convolution_1 = unsqueeze_38 = None
        mul_48: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(where_6, sub_12)
        sum_6: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_48, [0, 2, 3]);  mul_48 = None
        mul_49: "f32[128]" = torch.ops.aten.mul.Tensor(sum_5, 3.814697265625e-06)
        unsqueeze_39: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_49, 0);  mul_49 = None
        unsqueeze_40: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 2);  unsqueeze_39 = None
        unsqueeze_41: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 3);  unsqueeze_40 = None
        mul_50: "f32[128]" = torch.ops.aten.mul.Tensor(sum_6, 3.814697265625e-06)
        mul_51: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_52: "f32[128]" = torch.ops.aten.mul.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        unsqueeze_42: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_52, 0);  mul_52 = None
        unsqueeze_43: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
        unsqueeze_44: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 3);  unsqueeze_43 = None
        mul_53: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_7);  primals_7 = None
        unsqueeze_45: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_53, 0);  mul_53 = None
        unsqueeze_46: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 2);  unsqueeze_45 = None
        unsqueeze_47: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 3);  unsqueeze_46 = None
        mul_54: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_44);  sub_12 = unsqueeze_44 = None
        sub_14: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(where_6, mul_54);  where_6 = mul_54 = None
        sub_15: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_14, unsqueeze_41);  sub_14 = unsqueeze_41 = None
        mul_55: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_47);  sub_15 = unsqueeze_47 = None
        mul_56: "f32[128]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_1);  sum_6 = squeeze_1 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_55, where, primals_3, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_55 = primals_3 = None
        getitem_15: "f32[1024, 64, 32, 32]" = convolution_backward_3[0]
        getitem_16: "f32[128, 64, 4, 4]" = convolution_backward_3[1];  convolution_backward_3 = None
        gt_7: "b8[1024, 64, 32, 32]" = torch.ops.aten.gt.Scalar(where, 0);  where = None
        mul_57: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_15, 0.2)
        where_7: "f32[1024, 64, 32, 32]" = torch.ops.aten.where.self(gt_7, getitem_15, mul_57);  gt_7 = getitem_15 = mul_57 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_7, primals_2, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  where_7 = primals_2 = primals_1 = None
        getitem_19: "f32[64, 3, 4, 4]" = convolution_backward_4[1];  convolution_backward_4 = None
        return (getitem_19, None, getitem_16, None, None, None, mul_56, sum_5, getitem_13, None, None, None, mul_46, sum_3, getitem_10, None, None, None, mul_36, sum_1, getitem_7)
