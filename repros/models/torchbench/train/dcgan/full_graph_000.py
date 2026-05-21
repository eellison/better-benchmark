class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 4, 4]", primals_2: "f32[1024, 3, 64, 64]", primals_3: "f32[128, 64, 4, 4]", primals_4: "i64[]", primals_5: "f32[128]", primals_6: "f32[128]", primals_7: "f32[128]", primals_8: "f32[128]", primals_9: "f32[256, 128, 4, 4]", primals_10: "i64[]", primals_11: "f32[256]", primals_12: "f32[256]", primals_13: "f32[256]", primals_14: "f32[256]", primals_15: "f32[512, 256, 4, 4]", primals_16: "i64[]", primals_17: "f32[512]", primals_18: "f32[512]", primals_19: "f32[512]", primals_20: "f32[512]", primals_21: "f32[1, 512, 4, 4]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dcgan/__init__.py:128 in forward, code: return self.main(input)
        convolution: "f32[1024, 64, 32, 32]" = torch.ops.aten.convolution.default(primals_2, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        gt: "b8[1024, 64, 32, 32]" = torch.ops.aten.gt.Scalar(convolution, 0)
        mul: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(convolution, 0.2)
        where: "f32[1024, 64, 32, 32]" = torch.ops.aten.where.self(gt, convolution, mul);  gt = convolution = mul = None
        convolution_1: "f32[1024, 128, 16, 16]" = torch.ops.aten.convolution.default(where, primals_3, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_4, 1)
        var_mean = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_1)
        mul_1: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_2: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_3: "f32[128]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_2: "f32[128]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        squeeze_2: "f32[128]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_4: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000038147118175);  squeeze_2 = None
        mul_5: "f32[128]" = torch.ops.aten.mul.Tensor(mul_4, 0.1);  mul_4 = None
        mul_6: "f32[128]" = torch.ops.aten.mul.Tensor(primals_6, 0.9)
        add_3: "f32[128]" = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
        unsqueeze: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_1: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_7: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_1);  mul_1 = unsqueeze_1 = None
        unsqueeze_2: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
        unsqueeze_3: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[1024, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_7, unsqueeze_3);  mul_7 = unsqueeze_3 = None
        gt_1: "b8[1024, 128, 16, 16]" = torch.ops.aten.gt.Scalar(add_4, 0)
        mul_8: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_4, 0.2)
        where_1: "f32[1024, 128, 16, 16]" = torch.ops.aten.where.self(gt_1, add_4, mul_8);  gt_1 = add_4 = mul_8 = None
        convolution_2: "f32[1024, 256, 8, 8]" = torch.ops.aten.convolution.default(where_1, primals_9, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_10, 1)
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 256, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 256, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_3)
        mul_9: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_10: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_11: "f32[256]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_7: "f32[256]" = torch.ops.aten.add.Tensor(mul_10, mul_11);  mul_10 = mul_11 = None
        squeeze_5: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_12: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000152590218967);  squeeze_5 = None
        mul_13: "f32[256]" = torch.ops.aten.mul.Tensor(mul_12, 0.1);  mul_12 = None
        mul_14: "f32[256]" = torch.ops.aten.mul.Tensor(primals_12, 0.9)
        add_8: "f32[256]" = torch.ops.aten.add.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
        unsqueeze_4: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
        unsqueeze_5: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_15: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(mul_9, unsqueeze_5);  mul_9 = unsqueeze_5 = None
        unsqueeze_6: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
        unsqueeze_7: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[1024, 256, 8, 8]" = torch.ops.aten.add.Tensor(mul_15, unsqueeze_7);  mul_15 = unsqueeze_7 = None
        gt_2: "b8[1024, 256, 8, 8]" = torch.ops.aten.gt.Scalar(add_9, 0)
        mul_16: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(add_9, 0.2)
        where_2: "f32[1024, 256, 8, 8]" = torch.ops.aten.where.self(gt_2, add_9, mul_16);  gt_2 = add_9 = mul_16 = None
        convolution_3: "f32[1024, 512, 4, 4]" = torch.ops.aten.convolution.default(where_2, primals_15, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_16, 1)
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 512, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 512, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_5)
        mul_17: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_7: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_18: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_19: "f32[512]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_12: "f32[512]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        squeeze_8: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_20: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000610388817677);  squeeze_8 = None
        mul_21: "f32[512]" = torch.ops.aten.mul.Tensor(mul_20, 0.1);  mul_20 = None
        mul_22: "f32[512]" = torch.ops.aten.mul.Tensor(primals_18, 0.9)
        add_13: "f32[512]" = torch.ops.aten.add.Tensor(mul_21, mul_22);  mul_21 = mul_22 = None
        unsqueeze_8: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
        unsqueeze_9: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_23: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_9);  mul_17 = unsqueeze_9 = None
        unsqueeze_10: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
        unsqueeze_11: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[1024, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_11);  mul_23 = unsqueeze_11 = None
        gt_3: "b8[1024, 512, 4, 4]" = torch.ops.aten.gt.Scalar(add_14, 0)
        mul_24: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(add_14, 0.2)
        where_3: "f32[1024, 512, 4, 4]" = torch.ops.aten.where.self(gt_3, add_14, mul_24);  gt_3 = add_14 = mul_24 = None
        convolution_4: "f32[1024, 1, 1, 1]" = torch.ops.aten.convolution.default(where_3, primals_21, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        sigmoid: "f32[1024, 1, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4);  convolution_4 = None
        unsqueeze_12: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_13: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        unsqueeze_14: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None
        unsqueeze_24: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_25: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
        unsqueeze_26: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 3);  unsqueeze_25 = None
        unsqueeze_36: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_37: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None
        unsqueeze_38: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 3);  unsqueeze_37 = None

        # No stacktrace found for following nodes
        copy_: "i64[]" = torch.ops.aten.copy_.default(primals_4, add);  primals_4 = add = copy_ = None
        copy__1: "f32[128]" = torch.ops.aten.copy_.default(primals_5, add_2);  primals_5 = add_2 = copy__1 = None
        copy__2: "f32[128]" = torch.ops.aten.copy_.default(primals_6, add_3);  primals_6 = add_3 = copy__2 = None
        copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_10, add_5);  primals_10 = add_5 = copy__3 = None
        copy__4: "f32[256]" = torch.ops.aten.copy_.default(primals_11, add_7);  primals_11 = add_7 = copy__4 = None
        copy__5: "f32[256]" = torch.ops.aten.copy_.default(primals_12, add_8);  primals_12 = add_8 = copy__5 = None
        copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_16, add_10);  primals_16 = add_10 = copy__6 = None
        copy__7: "f32[512]" = torch.ops.aten.copy_.default(primals_17, add_12);  primals_17 = add_12 = copy__7 = None
        copy__8: "f32[512]" = torch.ops.aten.copy_.default(primals_18, add_13);  primals_18 = add_13 = copy__8 = None
        return (sigmoid, primals_1, primals_2, primals_3, primals_7, primals_9, primals_13, primals_15, primals_19, primals_21, where, convolution_1, squeeze_1, where_1, convolution_2, squeeze_4, where_2, convolution_3, squeeze_7, where_3, sigmoid, unsqueeze_14, unsqueeze_26, unsqueeze_38)
