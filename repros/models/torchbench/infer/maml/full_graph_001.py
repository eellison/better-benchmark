class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 1, 3, 3]", arg1_1: "f32[64]", arg2_1: "f32[5, 1, 28, 28]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64]", arg7_1: "f32[64, 64, 3, 3]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[64]", arg13_1: "f32[64, 64, 3, 3]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[64]", arg17_1: "f32[64]", arg18_1: "f32[64]", arg19_1: "f32[64, 64, 2, 2]", arg20_1: "f32[64]", arg21_1: "f32[64]", arg22_1: "f32[64]", arg23_1: "f32[64]", arg24_1: "f32[64]", arg25_1: "f32[5, 64]", arg26_1: "f32[5]", arg27_1: "i64[5]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:170 in forward, code: x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
        convolution: "f32[5, 64, 13, 13]" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:203 in forward, code: x = F.relu(x, inplace=param[0])
        relu: "f32[5, 64, 13, 13]" = torch.ops.aten.relu.default(convolution);  convolution = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        var_mean = torch.ops.aten.var_mean.correction(relu, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
        squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(arg5_1, 0.9)
        add_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3])
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0011848341232228);  squeeze_2 = None
        mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(arg6_1, 0.9)
        add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        sub: "f32[5, 64, 13, 13]" = torch.ops.aten.sub.Tensor(relu, getitem_1);  relu = getitem_1 = None
        add: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[5, 64, 13, 13]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[5, 64, 13, 13]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[5, 64, 13, 13]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:170 in forward, code: x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
        convolution_1: "f32[5, 64, 6, 6]" = torch.ops.aten.convolution.default(add_3, arg7_1, arg8_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  add_3 = arg7_1 = arg8_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:203 in forward, code: x = F.relu(x, inplace=param[0])
        relu_1: "f32[5, 64, 6, 6]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        var_mean_1 = torch.ops.aten.var_mean.correction(relu_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3])
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1);  squeeze_3 = None
        mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(arg11_1, 0.9)
        add_5: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3])
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.005586592178771);  squeeze_5 = None
        mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(arg12_1, 0.9)
        add_6: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        sub_1: "f32[5, 64, 6, 6]" = torch.ops.aten.sub.Tensor(relu_1, getitem_3);  relu_1 = getitem_3 = None
        add_4: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_7: "f32[5, 64, 6, 6]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[5, 64, 6, 6]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_7: "f32[5, 64, 6, 6]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:170 in forward, code: x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
        convolution_2: "f32[5, 64, 2, 2]" = torch.ops.aten.convolution.default(add_7, arg13_1, arg14_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  add_7 = arg13_1 = arg14_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:203 in forward, code: x = F.relu(x, inplace=param[0])
        relu_2: "f32[5, 64, 2, 2]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        var_mean_2 = torch.ops.aten.var_mean.correction(relu_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3])
        mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(arg17_1, 0.9)
        add_9: "f32[64]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3])
        mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0526315789473684);  squeeze_8 = None
        mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(arg18_1, 0.9)
        add_10: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        sub_2: "f32[5, 64, 2, 2]" = torch.ops.aten.sub.Tensor(relu_2, getitem_5);  relu_2 = getitem_5 = None
        add_8: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_14: "f32[5, 64, 2, 2]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[5, 64, 2, 2]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_11: "f32[5, 64, 2, 2]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:170 in forward, code: x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
        convolution_3: "f32[5, 64, 1, 1]" = torch.ops.aten.convolution.default(add_11, arg19_1, arg20_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_11 = arg19_1 = arg20_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:203 in forward, code: x = F.relu(x, inplace=param[0])
        relu_3: "f32[5, 64, 1, 1]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        var_mean_3 = torch.ops.aten.var_mean.correction(relu_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 64, 1, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 64, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        squeeze_9: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3])
        mul_22: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1);  squeeze_9 = None
        mul_23: "f32[64]" = torch.ops.aten.mul.Tensor(arg23_1, 0.9)
        add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3])
        mul_24: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.25);  squeeze_11 = None
        mul_25: "f32[64]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[64]" = torch.ops.aten.mul.Tensor(arg24_1, 0.9)
        add_14: "f32[64]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/meta.py:169 in torch_dynamo_resume_in_finetunning_at_165, code: loss = F.cross_entropy(logits, y_spt)
        ne_1: "b8[5]" = torch.ops.aten.ne.Scalar(arg27_1, -100)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        sub_3: "f32[5, 64, 1, 1]" = torch.ops.aten.sub.Tensor(relu_3, getitem_7);  relu_3 = getitem_7 = None
        add_12: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_21: "f32[5, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[5, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_15: "f32[5, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:198 in forward, code: x = x.view(x.size(0), -1)
        view: "f32[5, 64]" = torch.ops.aten.reshape.default(add_15, [5, -1]);  add_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:181 in forward, code: x = F.linear(x, w, b)
        permute: "f32[64, 5]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm: "f32[5, 5]" = torch.ops.aten.addmm.default(arg26_1, view, permute);  arg26_1 = view = permute = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/meta.py:169 in torch_dynamo_resume_in_finetunning_at_165, code: loss = F.cross_entropy(logits, y_spt)
        amax: "f32[5, 1]" = torch.ops.aten.amax.default(addmm, [1], True)
        sub_4: "f32[5, 5]" = torch.ops.aten.sub.Tensor(addmm, amax);  amax = None
        exp: "f32[5, 5]" = torch.ops.aten.exp.default(sub_4)
        sum_1: "f32[5, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[5, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_5: "f32[5, 5]" = torch.ops.aten.sub.Tensor(sub_4, log);  sub_4 = log = None
        ne: "b8[5]" = torch.ops.aten.ne.Scalar(arg27_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[5]" = torch.ops.aten.where.self(ne, arg27_1, full_default);  ne = full_default = None
        unsqueeze_16: "i64[5, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[5, 1]" = torch.ops.aten.gather.default(sub_5, 1, unsqueeze_16);  sub_5 = unsqueeze_16 = None
        squeeze_12: "f32[5]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[5]" = torch.ops.aten.neg.default(squeeze_12);  squeeze_12 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[5]" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[5]" = torch.ops.aten.ne.Scalar(arg27_1, -100);  arg27_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = convert_element_type = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/maml/learner.py:190 in forward, code: x = F.batch_norm(
        copy_: "f32[64]" = torch.ops.aten.copy_.default(arg5_1, add_1);  arg5_1 = add_1 = copy_ = None
        copy__1: "f32[64]" = torch.ops.aten.copy_.default(arg6_1, add_2);  arg6_1 = add_2 = copy__1 = None
        copy__2: "f32[64]" = torch.ops.aten.copy_.default(arg11_1, add_5);  arg11_1 = add_5 = copy__2 = None
        copy__3: "f32[64]" = torch.ops.aten.copy_.default(arg12_1, add_6);  arg12_1 = add_6 = copy__3 = None
        copy__4: "f32[64]" = torch.ops.aten.copy_.default(arg17_1, add_9);  arg17_1 = add_9 = copy__4 = None
        copy__5: "f32[64]" = torch.ops.aten.copy_.default(arg18_1, add_10);  arg18_1 = add_10 = copy__5 = None
        copy__6: "f32[64]" = torch.ops.aten.copy_.default(arg23_1, add_13);  arg23_1 = add_13 = copy__6 = None
        copy__7: "f32[64]" = torch.ops.aten.copy_.default(arg24_1, add_14);  arg24_1 = add_14 = copy__7 = None
        return (div, addmm)
