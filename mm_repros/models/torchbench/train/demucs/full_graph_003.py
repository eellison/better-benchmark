class GraphModule(torch.nn.Module):
    def forward(self, add: "bf16[4, 2048, 92][2048, 1, 8192]cuda:0", convert_element_type_1: "bf16[4096, 2048, 3][6144, 3, 1]cuda:0", convolution: "bf16[4, 4096, 90][368640, 90, 1]cuda:0", glu: "bf16[4, 2048, 90][184320, 90, 1]cuda:0", convert_element_type_3: "bf16[2048, 1024, 8][8192, 8, 1]cuda:0", add_1: "bf16[4, 1024, 364][372736, 364, 1]cuda:0", convert_element_type_5: "bf16[2048, 1024, 3][3072, 3, 1]cuda:0", convolution_2: "bf16[4, 2048, 362][741376, 362, 1]cuda:0", glu_1: "bf16[4, 1024, 362][370688, 362, 1]cuda:0", convert_element_type_7: "bf16[1024, 512, 8][4096, 8, 1]cuda:0", add_2: "bf16[4, 512, 1452][743424, 1452, 1]cuda:0", convert_element_type_9: "bf16[1024, 512, 3][1536, 3, 1]cuda:0", convolution_4: "bf16[4, 1024, 1450][1484800, 1450, 1]cuda:0", glu_2: "bf16[4, 512, 1450][742400, 1450, 1]cuda:0", convert_element_type_11: "bf16[512, 256, 8][2048, 8, 1]cuda:0", add_3: "bf16[4, 256, 5804][1485824, 5804, 1]cuda:0", convert_element_type_13: "bf16[512, 256, 3][768, 3, 1]cuda:0", convolution_6: "bf16[4, 512, 5802][2970624, 5802, 1]cuda:0", glu_3: "bf16[4, 256, 5802][1485312, 5802, 1]cuda:0", convert_element_type_15: "bf16[256, 128, 8][1024, 8, 1]cuda:0", add_4: "bf16[4, 128, 23212][2971136, 23212, 1]cuda:0", convert_element_type_17: "bf16[256, 128, 3][384, 3, 1]cuda:0", convolution_8: "bf16[4, 256, 23210][5941760, 23210, 1]cuda:0", glu_4: "bf16[4, 128, 23210][2970880, 23210, 1]cuda:0", convert_element_type_19: "bf16[128, 64, 8][512, 8, 1]cuda:0", add_5: "bf16[4, 64, 92844][5942016, 92844, 1]cuda:0", convert_element_type_21: "bf16[128, 64, 3][192, 3, 1]cuda:0", convolution_10: "bf16[4, 128, 92842][11883776, 92842, 1]cuda:0", glu_5: "bf16[4, 64, 92842][5941888, 92842, 1]cuda:0", convert_element_type_23: "bf16[64, 8, 8][64, 8, 1]cuda:0", le: "b8[4, 64, 92844][5947392, 92928, 1]cuda:0", le_1: "b8[4, 128, 23212][2981888, 23296, 1]cuda:0", le_2: "b8[4, 256, 5804][1507328, 5888, 1]cuda:0", le_3: "b8[4, 512, 1452][786432, 1536, 1]cuda:0", le_4: "b8[4, 1024, 364][372736, 364, 1]cuda:0", tangents_1: "bf16[4, 4, 2, 371372][2970976, 742744, 371372, 1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:232 in torch_dynamo_resume_in_forward_at_220, code: x = x.view(x.size(0), self.sources, self.audio_channels, x.size(-1))
        view_1: "bf16[4, 8, 371372][2970976, 371372, 1]cuda:0" = torch.ops.aten.reshape.default(tangents_1, [4, 8, 371372]);  tangents_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        sum_1: "bf16[8][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1, [0, 2])
        convolution_backward = torch.ops.aten.convolution_backward.default(view_1, glu_5, convert_element_type_23, [8], [4], [0], [1], True, [0], 1, [True, True, False]);  view_1 = glu_5 = convert_element_type_23 = None
        getitem: "bf16[4, 64, 92842][5941888, 92842, 1]cuda:0" = convolution_backward[0]
        getitem_1: "bf16[64, 8, 8][64, 8, 1]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_24: "f32[64, 8, 8][64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
        convert_element_type_25: "f32[8][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        convert_element_type_26: "f32[4, 64, 92842][5941888, 92842, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        convert_element_type_27: "f32[4, 128, 92842][11883776, 92842, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        slice_6: "f32[4, 64, 92842][11883776, 92842, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_27, 1, 0, 64)
        slice_7: "f32[4, 64, 92842][11883776, 92842, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_27, 1, 64, 128);  convert_element_type_27 = None
        sigmoid: "f32[4, 64, 92842][5941888, 92842, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_7);  slice_7 = None
        sub: "f32[4, 64, 92842][5941888, 92842, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid)
        mul: "f32[4, 64, 92842][5941888, 92842, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, sigmoid);  sub = None
        mul_1: "f32[4, 64, 92842][5941888, 92842, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, slice_6);  mul = slice_6 = None
        mul_2: "f32[4, 64, 92842][5941888, 92842, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, convert_element_type_26);  mul_1 = None
        mul_3: "f32[4, 64, 92842][5941888, 92842, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid, convert_element_type_26);  sigmoid = convert_element_type_26 = None
        cat: "f32[4, 128, 92842][11883776, 92842, 1]cuda:0" = torch.ops.aten.cat.default([mul_3, mul_2], 1);  mul_3 = mul_2 = None
        convert_element_type_28: "bf16[4, 128, 92842][11883776, 92842, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat, torch.bfloat16);  cat = None
        sum_2: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_28, [0, 2])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_28, add_5, convert_element_type_21, [128], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_28 = add_5 = convert_element_type_21 = None
        getitem_3: "bf16[4, 64, 92844][5942016, 92844, 1]cuda:0" = convolution_backward_1[0]
        getitem_4: "bf16[128, 64, 3][192, 3, 1]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_29: "f32[128, 64, 3][192, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_4, torch.float32);  getitem_4 = None
        convert_element_type_30: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.full.default([4, 64, 95696], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default, getitem_3, 2, 1426, -1426);  full_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[4, 64, 92844][5942016, 92844, 1]cuda:0" = torch.ops.aten.where.self(le, full_default_1, getitem_3);  le = getitem_3 = None
        sum_3: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where, [0, 2])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where, glu_4, convert_element_type_19, [64], [4], [0], [1], True, [0], 1, [True, True, False]);  where = glu_4 = convert_element_type_19 = None
        getitem_6: "bf16[4, 128, 23210][2970880, 23210, 1]cuda:0" = convolution_backward_2[0]
        getitem_7: "bf16[128, 64, 8][512, 8, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_31: "f32[128, 64, 8][512, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
        convert_element_type_32: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None
        convert_element_type_33: "f32[4, 128, 23210][2970880, 23210, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_6, torch.float32);  getitem_6 = None
        convert_element_type_34: "f32[4, 256, 23210][5941760, 23210, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        slice_8: "f32[4, 128, 23210][5941760, 23210, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_34, 1, 0, 128)
        slice_9: "f32[4, 128, 23210][5941760, 23210, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_34, 1, 128, 256);  convert_element_type_34 = None
        sigmoid_1: "f32[4, 128, 23210][2970880, 23210, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_9);  slice_9 = None
        sub_1: "f32[4, 128, 23210][2970880, 23210, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid_1)
        mul_4: "f32[4, 128, 23210][2970880, 23210, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, sigmoid_1);  sub_1 = None
        mul_5: "f32[4, 128, 23210][2970880, 23210, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, slice_8);  mul_4 = slice_8 = None
        mul_6: "f32[4, 128, 23210][2970880, 23210, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, convert_element_type_33);  mul_5 = None
        mul_7: "f32[4, 128, 23210][2970880, 23210, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_1, convert_element_type_33);  sigmoid_1 = convert_element_type_33 = None
        cat_1: "f32[4, 256, 23210][5941760, 23210, 1]cuda:0" = torch.ops.aten.cat.default([mul_7, mul_6], 1);  mul_7 = mul_6 = None
        convert_element_type_35: "bf16[4, 256, 23210][5941760, 23210, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_1, torch.bfloat16);  cat_1 = None
        sum_4: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_35, [0, 2])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_35, add_4, convert_element_type_17, [256], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_35 = add_4 = convert_element_type_17 = None
        getitem_9: "bf16[4, 128, 23212][2971136, 23212, 1]cuda:0" = convolution_backward_3[0]
        getitem_10: "bf16[256, 128, 3][384, 3, 1]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_36: "f32[256, 128, 3][384, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
        convert_element_type_37: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_4, torch.float32);  sum_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_2: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.full.default([4, 128, 23923], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_1: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_2, getitem_9, 2, 355, -356);  full_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        where_1: "bf16[4, 128, 23212][2971136, 23212, 1]cuda:0" = torch.ops.aten.where.self(le_1, full_default_1, getitem_9);  le_1 = getitem_9 = None
        sum_5: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_1, glu_3, convert_element_type_15, [128], [4], [0], [1], True, [0], 1, [True, True, False]);  where_1 = glu_3 = convert_element_type_15 = None
        getitem_12: "bf16[4, 256, 5802][1485312, 5802, 1]cuda:0" = convolution_backward_4[0]
        getitem_13: "bf16[256, 128, 8][1024, 8, 1]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_38: "f32[256, 128, 8][1024, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
        convert_element_type_39: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_5, torch.float32);  sum_5 = None
        convert_element_type_40: "f32[4, 256, 5802][1485312, 5802, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_12, torch.float32);  getitem_12 = None
        convert_element_type_41: "f32[4, 512, 5802][2970624, 5802, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        slice_10: "f32[4, 256, 5802][2970624, 5802, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_41, 1, 0, 256)
        slice_11: "f32[4, 256, 5802][2970624, 5802, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_41, 1, 256, 512);  convert_element_type_41 = None
        sigmoid_2: "f32[4, 256, 5802][1485312, 5802, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_11);  slice_11 = None
        sub_2: "f32[4, 256, 5802][1485312, 5802, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid_2)
        mul_8: "f32[4, 256, 5802][1485312, 5802, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, sigmoid_2);  sub_2 = None
        mul_9: "f32[4, 256, 5802][1485312, 5802, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, slice_10);  mul_8 = slice_10 = None
        mul_10: "f32[4, 256, 5802][1485312, 5802, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, convert_element_type_40);  mul_9 = None
        mul_11: "f32[4, 256, 5802][1485312, 5802, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_2, convert_element_type_40);  sigmoid_2 = convert_element_type_40 = None
        cat_2: "f32[4, 512, 5802][2970624, 5802, 1]cuda:0" = torch.ops.aten.cat.default([mul_11, mul_10], 1);  mul_11 = mul_10 = None
        convert_element_type_42: "bf16[4, 512, 5802][2970624, 5802, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_2, torch.bfloat16);  cat_2 = None
        sum_6: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_42, [0, 2])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_42, add_3, convert_element_type_13, [512], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_42 = add_3 = convert_element_type_13 = None
        getitem_15: "bf16[4, 256, 5804][1485824, 5804, 1]cuda:0" = convolution_backward_5[0]
        getitem_16: "bf16[512, 256, 3][768, 3, 1]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_43: "f32[512, 256, 3][768, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_16, torch.float32);  getitem_16 = None
        convert_element_type_44: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_6, torch.float32);  sum_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_4: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.full.default([4, 256, 5979], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_2: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_4, getitem_15, 2, 87, -88);  full_default_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        where_2: "bf16[4, 256, 5804][1485824, 5804, 1]cuda:0" = torch.ops.aten.where.self(le_2, full_default_1, getitem_15);  le_2 = getitem_15 = None
        sum_7: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_2, glu_2, convert_element_type_11, [256], [4], [0], [1], True, [0], 1, [True, True, False]);  where_2 = glu_2 = convert_element_type_11 = None
        getitem_18: "bf16[4, 512, 1450][742400, 1450, 1]cuda:0" = convolution_backward_6[0]
        getitem_19: "bf16[512, 256, 8][2048, 8, 1]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_45: "f32[512, 256, 8][2048, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
        convert_element_type_46: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None
        convert_element_type_47: "f32[4, 512, 1450][742400, 1450, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_18, torch.float32);  getitem_18 = None
        convert_element_type_48: "f32[4, 1024, 1450][1484800, 1450, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        slice_12: "f32[4, 512, 1450][1484800, 1450, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_48, 1, 0, 512)
        slice_13: "f32[4, 512, 1450][1484800, 1450, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_48, 1, 512, 1024);  convert_element_type_48 = None
        sigmoid_3: "f32[4, 512, 1450][742400, 1450, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_13);  slice_13 = None
        sub_3: "f32[4, 512, 1450][742400, 1450, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid_3)
        mul_12: "f32[4, 512, 1450][742400, 1450, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, sigmoid_3);  sub_3 = None
        mul_13: "f32[4, 512, 1450][742400, 1450, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, slice_12);  mul_12 = slice_12 = None
        mul_14: "f32[4, 512, 1450][742400, 1450, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, convert_element_type_47);  mul_13 = None
        mul_15: "f32[4, 512, 1450][742400, 1450, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_3, convert_element_type_47);  sigmoid_3 = convert_element_type_47 = None
        cat_3: "f32[4, 1024, 1450][1484800, 1450, 1]cuda:0" = torch.ops.aten.cat.default([mul_15, mul_14], 1);  mul_15 = mul_14 = None
        convert_element_type_49: "bf16[4, 1024, 1450][1484800, 1450, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_3, torch.bfloat16);  cat_3 = None
        sum_8: "bf16[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_49, [0, 2])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_49, add_2, convert_element_type_9, [1024], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_49 = add_2 = convert_element_type_9 = None
        getitem_21: "bf16[4, 512, 1452][743424, 1452, 1]cuda:0" = convolution_backward_7[0]
        getitem_22: "bf16[1024, 512, 3][1536, 3, 1]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_50: "f32[1024, 512, 3][1536, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_22, torch.float32);  getitem_22 = None
        convert_element_type_51: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_6: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.full.default([4, 512, 1493], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_3: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_6, getitem_21, 2, 20, -21);  full_default_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        where_3: "bf16[4, 512, 1452][743424, 1452, 1]cuda:0" = torch.ops.aten.where.self(le_3, full_default_1, getitem_21);  le_3 = getitem_21 = None
        sum_9: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_3, glu_1, convert_element_type_7, [512], [4], [0], [1], True, [0], 1, [True, True, False]);  where_3 = glu_1 = convert_element_type_7 = None
        getitem_24: "bf16[4, 1024, 362][370688, 362, 1]cuda:0" = convolution_backward_8[0]
        getitem_25: "bf16[1024, 512, 8][4096, 8, 1]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_52: "f32[1024, 512, 8][4096, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
        convert_element_type_53: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_9, torch.float32);  sum_9 = None
        convert_element_type_54: "f32[4, 1024, 362][370688, 362, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_24, torch.float32);  getitem_24 = None
        convert_element_type_55: "f32[4, 2048, 362][741376, 362, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        slice_14: "f32[4, 1024, 362][741376, 362, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_55, 1, 0, 1024)
        slice_15: "f32[4, 1024, 362][741376, 362, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_55, 1, 1024, 2048);  convert_element_type_55 = None
        sigmoid_4: "f32[4, 1024, 362][370688, 362, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_15);  slice_15 = None
        sub_4: "f32[4, 1024, 362][370688, 362, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid_4)
        mul_16: "f32[4, 1024, 362][370688, 362, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, sigmoid_4);  sub_4 = None
        mul_17: "f32[4, 1024, 362][370688, 362, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, slice_14);  mul_16 = slice_14 = None
        mul_18: "f32[4, 1024, 362][370688, 362, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, convert_element_type_54);  mul_17 = None
        mul_19: "f32[4, 1024, 362][370688, 362, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_4, convert_element_type_54);  sigmoid_4 = convert_element_type_54 = None
        cat_4: "f32[4, 2048, 362][741376, 362, 1]cuda:0" = torch.ops.aten.cat.default([mul_19, mul_18], 1);  mul_19 = mul_18 = None
        convert_element_type_56: "bf16[4, 2048, 362][741376, 362, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_4, torch.bfloat16);  cat_4 = None
        sum_10: "bf16[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_56, [0, 2])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_56, add_1, convert_element_type_5, [2048], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_56 = add_1 = convert_element_type_5 = None
        getitem_27: "bf16[4, 1024, 364][372736, 364, 1]cuda:0" = convolution_backward_9[0]
        getitem_28: "bf16[2048, 1024, 3][3072, 3, 1]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_57: "f32[2048, 1024, 3][3072, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_28, torch.float32);  getitem_28 = None
        convert_element_type_58: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_10, torch.float32);  sum_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_8: "bf16[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.full.default([4, 1024, 372], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_4: "bf16[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_8, getitem_27, 2, 4, -4);  full_default_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        where_4: "bf16[4, 1024, 364][372736, 364, 1]cuda:0" = torch.ops.aten.where.self(le_4, full_default_1, getitem_27);  le_4 = full_default_1 = getitem_27 = None
        sum_11: "bf16[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(where_4, glu, convert_element_type_3, [1024], [4], [0], [1], True, [0], 1, [True, True, False]);  where_4 = glu = convert_element_type_3 = None
        getitem_30: "bf16[4, 2048, 90][184320, 90, 1]cuda:0" = convolution_backward_10[0]
        getitem_31: "bf16[2048, 1024, 8][8192, 8, 1]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_59: "f32[2048, 1024, 8][8192, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
        convert_element_type_60: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_11, torch.float32);  sum_11 = None
        convert_element_type_61: "f32[4, 2048, 90][184320, 90, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_30, torch.float32);  getitem_30 = None
        convert_element_type_62: "f32[4, 4096, 90][368640, 90, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        slice_16: "f32[4, 2048, 90][368640, 90, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_62, 1, 0, 2048)
        slice_17: "f32[4, 2048, 90][368640, 90, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_62, 1, 2048, 4096);  convert_element_type_62 = None
        sigmoid_5: "f32[4, 2048, 90][184320, 90, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_17);  slice_17 = None
        sub_5: "f32[4, 2048, 90][184320, 90, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid_5)
        mul_20: "f32[4, 2048, 90][184320, 90, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, sigmoid_5);  sub_5 = None
        mul_21: "f32[4, 2048, 90][184320, 90, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, slice_16);  mul_20 = slice_16 = None
        mul_22: "f32[4, 2048, 90][184320, 90, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, convert_element_type_61);  mul_21 = None
        mul_23: "f32[4, 2048, 90][184320, 90, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_5, convert_element_type_61);  sigmoid_5 = convert_element_type_61 = None
        cat_5: "f32[4, 4096, 90][368640, 90, 1]cuda:0" = torch.ops.aten.cat.default([mul_23, mul_22], 1);  mul_23 = mul_22 = None
        convert_element_type_63: "bf16[4, 4096, 90][368640, 90, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_5, torch.bfloat16);  cat_5 = None
        sum_12: "bf16[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_63, [0, 2])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_63, add, convert_element_type_1, [4096], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_63 = add = convert_element_type_1 = None
        getitem_33: "bf16[4, 2048, 92][188416, 92, 1]cuda:0" = convolution_backward_11[0]
        getitem_34: "bf16[4096, 2048, 3][6144, 3, 1]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_64: "f32[4096, 2048, 3][6144, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_34, torch.float32);  getitem_34 = None
        convert_element_type_65: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_12, torch.float32);  sum_12 = None
        return (getitem_33, getitem_33, convert_element_type_64, convert_element_type_65, convert_element_type_59, convert_element_type_60, slice_scatter_4, convert_element_type_57, convert_element_type_58, convert_element_type_52, convert_element_type_53, slice_scatter_3, convert_element_type_50, convert_element_type_51, convert_element_type_45, convert_element_type_46, slice_scatter_2, convert_element_type_43, convert_element_type_44, convert_element_type_38, convert_element_type_39, slice_scatter_1, convert_element_type_36, convert_element_type_37, convert_element_type_31, convert_element_type_32, slice_scatter, convert_element_type_29, convert_element_type_30, convert_element_type_24, convert_element_type_25)
