class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 1, 3, 3][9, 9, 3, 1]cuda:0", arg1_1: "bf16[64][1]cuda:0", arg2_1: "bf16[5, 1, 28, 28][784, 784, 28, 1]cuda:0", arg3_1: "bf16[64][1]cuda:0", arg4_1: "bf16[64][1]cuda:0", arg5_1: "bf16[64][1]cuda:0", arg6_1: "bf16[64][1]cuda:0", arg7_1: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0", arg8_1: "bf16[64][1]cuda:0", arg9_1: "bf16[64][1]cuda:0", arg10_1: "bf16[64][1]cuda:0", arg11_1: "bf16[64][1]cuda:0", arg12_1: "bf16[64][1]cuda:0", arg13_1: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0", arg14_1: "bf16[64][1]cuda:0", arg15_1: "bf16[64][1]cuda:0", arg16_1: "bf16[64][1]cuda:0", arg17_1: "bf16[64][1]cuda:0", arg18_1: "bf16[64][1]cuda:0", arg19_1: "bf16[64, 64, 2, 2][256, 4, 2, 1]cuda:0", arg20_1: "bf16[64][1]cuda:0", arg21_1: "bf16[64][1]cuda:0", arg22_1: "bf16[64][1]cuda:0", arg23_1: "bf16[64][1]cuda:0", arg24_1: "bf16[64][1]cuda:0", arg25_1: "bf16[5, 64][64, 1]cuda:0", arg26_1: "bf16[5][1]cuda:0", arg27_1: "i64[5][1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:170 in forward, code: x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
        convolution: "bf16[5, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:203 in forward, code: x = F.relu(x, inplace=param[0])
        relu: "bf16[5, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution);  convolution = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        convert_element_type: "f32[5, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        squeeze: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "bf16[64][1]cuda:0" = torch.ops.aten.mul.Tensor(arg5_1, 0.9)
        add_1: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        convert_element_type_2: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        squeeze_2: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3])
        mul_3: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0011848341232228);  squeeze_2 = None
        mul_4: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "bf16[64][1]cuda:0" = torch.ops.aten.mul.Tensor(arg6_1, 0.9)
        add_2: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        convert_element_type_3: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        sub: "f32[5, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.sub.Tensor(relu, getitem_1);  relu = getitem_1 = None
        add: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[5, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        unsqueeze: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[5, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[5, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_1: "bf16[5, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:170 in forward, code: x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
        convolution_1: "bf16[5, 64, 6, 6][2304, 36, 6, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, arg7_1, arg8_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_1 = arg7_1 = arg8_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:203 in forward, code: x = F.relu(x, inplace=param[0])
        relu_1: "bf16[5, 64, 6, 6][2304, 36, 6, 1]cuda:0" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        convert_element_type_4: "f32[5, 64, 6, 6][2304, 36, 6, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_4 = None
        getitem_2: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        squeeze_3: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3])
        mul_8: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1);  squeeze_3 = None
        mul_9: "bf16[64][1]cuda:0" = torch.ops.aten.mul.Tensor(arg11_1, 0.9)
        add_5: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        convert_element_type_6: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        squeeze_5: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3])
        mul_10: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, 1.005586592178771);  squeeze_5 = None
        mul_11: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "bf16[64][1]cuda:0" = torch.ops.aten.mul.Tensor(arg12_1, 0.9)
        add_6: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        convert_element_type_7: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        sub_1: "f32[5, 64, 6, 6][2304, 36, 6, 1]cuda:0" = torch.ops.aten.sub.Tensor(relu_1, getitem_3);  relu_1 = getitem_3 = None
        add_4: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_7: "f32[5, 64, 6, 6][2304, 36, 6, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        unsqueeze_4: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_5: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[5, 64, 6, 6][2304, 36, 6, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_7: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_7: "f32[5, 64, 6, 6][2304, 36, 6, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_5: "bf16[5, 64, 6, 6][2304, 36, 6, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:170 in forward, code: x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
        convolution_2: "bf16[5, 64, 2, 2][256, 4, 2, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_5, arg13_1, arg14_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_5 = arg13_1 = arg14_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:203 in forward, code: x = F.relu(x, inplace=param[0])
        relu_2: "bf16[5, 64, 2, 2][256, 4, 2, 1]cuda:0" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        convert_element_type_8: "f32[5, 64, 2, 2][256, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_2, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_8 = None
        getitem_4: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        squeeze_6: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3])
        mul_15: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_16: "bf16[64][1]cuda:0" = torch.ops.aten.mul.Tensor(arg17_1, 0.9)
        add_9: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        convert_element_type_10: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        squeeze_8: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3])
        mul_17: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0526315789473684);  squeeze_8 = None
        mul_18: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "bf16[64][1]cuda:0" = torch.ops.aten.mul.Tensor(arg18_1, 0.9)
        add_10: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        convert_element_type_11: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None
        sub_2: "f32[5, 64, 2, 2][256, 4, 2, 1]cuda:0" = torch.ops.aten.sub.Tensor(relu_2, getitem_5);  relu_2 = getitem_5 = None
        add_8: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_14: "f32[5, 64, 2, 2][256, 4, 2, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        unsqueeze_8: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_9: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[5, 64, 2, 2][256, 4, 2, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_11: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_11: "f32[5, 64, 2, 2][256, 4, 2, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_9: "bf16[5, 64, 2, 2][256, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:170 in forward, code: x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
        convolution_3: "bf16[5, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_9, arg19_1, arg20_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_9 = arg19_1 = arg20_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:203 in forward, code: x = F.relu(x, inplace=param[0])
        relu_3: "bf16[5, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        convert_element_type_12: "f32[5, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_3, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_12, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_12 = None
        getitem_6: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        squeeze_9: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3])
        mul_22: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1);  squeeze_9 = None
        mul_23: "bf16[64][1]cuda:0" = torch.ops.aten.mul.Tensor(arg23_1, 0.9)
        add_13: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        convert_element_type_14: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        squeeze_11: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3])
        mul_24: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, 1.25);  squeeze_11 = None
        mul_25: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "bf16[64][1]cuda:0" = torch.ops.aten.mul.Tensor(arg24_1, 0.9)
        add_14: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        convert_element_type_15: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/meta.py:169 in torch_dynamo_resume_in_finetunning_at_165, code: loss = F.cross_entropy(logits, y_spt)
        ne_1: "b8[5][1]cuda:0" = torch.ops.aten.ne.Scalar(arg27_1, -100)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        sub_3: "f32[5, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(relu_3, getitem_7);  relu_3 = getitem_7 = None
        add_12: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_21: "f32[5, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        unsqueeze_12: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_13: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[5, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_15: "bf16[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_15: "f32[5, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convert_element_type_13: "bf16[5, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:198 in forward, code: x = x.view(x.size(0), -1)
        view: "bf16[5, 64][64, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_13, [5, -1]);  convert_element_type_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:181 in forward, code: x = F.linear(x, w, b)
        permute: "bf16[64, 5][1, 64]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm: "bf16[5, 5][5, 1]cuda:0" = torch.ops.aten.addmm.default(arg26_1, view, permute);  arg26_1 = view = permute = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/meta.py:169 in torch_dynamo_resume_in_finetunning_at_165, code: loss = F.cross_entropy(logits, y_spt)
        convert_element_type_19: "f32[5, 5][5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        amax: "f32[5, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_19, [1], True)
        sub_4: "f32[5, 5][5, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_19, amax);  convert_element_type_19 = amax = None
        exp: "f32[5, 5][5, 1]cuda:0" = torch.ops.aten.exp.default(sub_4)
        sum_1: "f32[5, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[5, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_5: "f32[5, 5][5, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_4, log);  sub_4 = log = None
        convert_element_type_20: "bf16[5, 5][5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_5, torch.bfloat16);  sub_5 = None
        ne: "b8[5][1]cuda:0" = torch.ops.aten.ne.Scalar(arg27_1, -100)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[5][1]cuda:0" = torch.ops.aten.where.self(ne, arg27_1, full_default);  ne = full_default = None
        unsqueeze_16: "i64[5, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "bf16[5, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_20, 1, unsqueeze_16);  convert_element_type_20 = unsqueeze_16 = None
        squeeze_12: "bf16[5][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[5][1]cuda:0" = torch.ops.aten.neg.default(squeeze_12);  squeeze_12 = None
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[5][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        sum_3: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[5][1]cuda:0" = torch.ops.aten.ne.Scalar(arg27_1, -100);  arg27_1 = None
        sum_2: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_21: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.bfloat16);  sum_2 = None
        div: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_21);  sum_3 = convert_element_type_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        copy_: "bf16[64][1]cuda:0" = torch.ops.aten.copy_.default(arg5_1, convert_element_type_2);  arg5_1 = convert_element_type_2 = copy_ = None
        copy__1: "bf16[64][1]cuda:0" = torch.ops.aten.copy_.default(arg6_1, convert_element_type_3);  arg6_1 = convert_element_type_3 = copy__1 = None
        copy__2: "bf16[64][1]cuda:0" = torch.ops.aten.copy_.default(arg11_1, convert_element_type_6);  arg11_1 = convert_element_type_6 = copy__2 = None
        copy__3: "bf16[64][1]cuda:0" = torch.ops.aten.copy_.default(arg12_1, convert_element_type_7);  arg12_1 = convert_element_type_7 = copy__3 = None
        copy__4: "bf16[64][1]cuda:0" = torch.ops.aten.copy_.default(arg17_1, convert_element_type_10);  arg17_1 = convert_element_type_10 = copy__4 = None
        copy__5: "bf16[64][1]cuda:0" = torch.ops.aten.copy_.default(arg18_1, convert_element_type_11);  arg18_1 = convert_element_type_11 = copy__5 = None
        copy__6: "bf16[64][1]cuda:0" = torch.ops.aten.copy_.default(arg23_1, convert_element_type_14);  arg23_1 = convert_element_type_14 = copy__6 = None
        copy__7: "bf16[64][1]cuda:0" = torch.ops.aten.copy_.default(arg24_1, convert_element_type_15);  arg24_1 = convert_element_type_15 = copy__7 = None
        return (div, addmm)
