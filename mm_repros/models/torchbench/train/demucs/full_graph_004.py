class GraphModule(torch.nn.Module):
    def forward(self, convert_element_type_1: "bf16[64, 2, 8][16, 8, 1]cuda:0", convert_element_type_2: "bf16[4, 2, 382788][765576, 382788, 1]cuda:0", relu: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0", convert_element_type_4: "bf16[128, 64, 1][64, 1, 1]cuda:0", convolution_1: "bf16[4, 128, 95696][12249088, 95696, 1]cuda:0", glu: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0", convert_element_type_6: "bf16[128, 64, 8][512, 8, 1]cuda:0", relu_1: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0", convert_element_type_8: "bf16[256, 128, 1][128, 1, 1]cuda:0", convolution_3: "bf16[4, 256, 23923][6124288, 23923, 1]cuda:0", glu_1: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0", convert_element_type_10: "bf16[256, 128, 8][1024, 8, 1]cuda:0", relu_2: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0", convert_element_type_12: "bf16[512, 256, 1][256, 1, 1]cuda:0", convolution_5: "bf16[4, 512, 5979][3061248, 5979, 1]cuda:0", glu_2: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0", convert_element_type_14: "bf16[512, 256, 8][2048, 8, 1]cuda:0", relu_3: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0", convert_element_type_16: "bf16[1024, 512, 1][512, 1, 1]cuda:0", convolution_7: "bf16[4, 1024, 1493][1528832, 1493, 1]cuda:0", glu_3: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0", convert_element_type_18: "bf16[1024, 512, 8][4096, 8, 1]cuda:0", relu_4: "bf16[4, 1024, 372][380928, 372, 1]cuda:0", convert_element_type_20: "bf16[2048, 1024, 1][1024, 1, 1]cuda:0", convolution_9: "bf16[4, 2048, 372][761856, 372, 1]cuda:0", glu_4: "bf16[4, 1024, 372][380928, 372, 1]cuda:0", convert_element_type_22: "bf16[2048, 1024, 8][8192, 8, 1]cuda:0", relu_5: "bf16[4, 2048, 92][188416, 92, 1]cuda:0", convert_element_type_24: "bf16[4096, 2048, 1][2048, 1, 1]cuda:0", convolution_11: "bf16[4, 4096, 92][376832, 92, 1]cuda:0", tangents_1: "bf16[4, 2048, 92][188416, 92, 1]cuda:0", tangents_2: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0", tangents_3: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0", tangents_4: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0", tangents_5: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0", tangents_6: "bf16[4, 1024, 372][380928, 372, 1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:215 in forward, code: x = encode(x)
        convert_element_type_25: "f32[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
        convert_element_type_26: "f32[4, 4096, 92][376832, 92, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        slice_1: "f32[4, 2048, 92][376832, 92, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_26, 1, 0, 2048)
        slice_2: "f32[4, 2048, 92][376832, 92, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_26, 1, 2048, 4096);  convert_element_type_26 = None
        sigmoid: "f32[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_2);  slice_2 = None
        sub: "f32[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid)
        mul: "f32[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, sigmoid);  sub = None
        mul_1: "f32[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, slice_1);  mul = slice_1 = None
        mul_2: "f32[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, convert_element_type_25);  mul_1 = None
        mul_3: "f32[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid, convert_element_type_25);  sigmoid = convert_element_type_25 = None
        cat: "f32[4, 4096, 92][376832, 92, 1]cuda:0" = torch.ops.aten.cat.default([mul_3, mul_2], 1);  mul_3 = mul_2 = None
        convert_element_type_27: "bf16[4, 4096, 92][376832, 92, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat, torch.bfloat16);  cat = None
        sum_1: "bf16[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_27, [0, 2])
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_27, relu_5, convert_element_type_24, [4096], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_27 = convert_element_type_24 = None
        getitem: "bf16[4, 2048, 92][188416, 92, 1]cuda:0" = convolution_backward[0]
        getitem_1: "bf16[4096, 2048, 1][2048, 1, 1]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_28: "f32[4096, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
        convert_element_type_29: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        le: "b8[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, getitem);  le = getitem = None
        sum_2: "bf16[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where, [0, 2])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(where, glu_4, convert_element_type_22, [2048], [4], [0], [1], False, [0], 1, [True, True, False]);  where = glu_4 = convert_element_type_22 = None
        getitem_3: "bf16[4, 1024, 372][380928, 372, 1]cuda:0" = convolution_backward_1[0]
        getitem_4: "bf16[2048, 1024, 8][8192, 8, 1]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        add: "bf16[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_6, getitem_3);  tangents_6 = getitem_3 = None
        convert_element_type_30: "f32[2048, 1024, 8][8192, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_4, torch.float32);  getitem_4 = None
        convert_element_type_31: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        convert_element_type_32: "f32[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        convert_element_type_33: "f32[4, 2048, 372][761856, 372, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        slice_3: "f32[4, 1024, 372][761856, 372, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_33, 1, 0, 1024)
        slice_4: "f32[4, 1024, 372][761856, 372, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_33, 1, 1024, 2048);  convert_element_type_33 = None
        sigmoid_1: "f32[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_4);  slice_4 = None
        sub_1: "f32[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid_1)
        mul_4: "f32[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, sigmoid_1);  sub_1 = None
        mul_5: "f32[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, slice_3);  mul_4 = slice_3 = None
        mul_6: "f32[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, convert_element_type_32);  mul_5 = None
        mul_7: "f32[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_1, convert_element_type_32);  sigmoid_1 = convert_element_type_32 = None
        cat_1: "f32[4, 2048, 372][761856, 372, 1]cuda:0" = torch.ops.aten.cat.default([mul_7, mul_6], 1);  mul_7 = mul_6 = None
        convert_element_type_34: "bf16[4, 2048, 372][761856, 372, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_1, torch.bfloat16);  cat_1 = None
        sum_3: "bf16[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_34, [0, 2])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_34, relu_4, convert_element_type_20, [2048], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_34 = convert_element_type_20 = None
        getitem_6: "bf16[4, 1024, 372][380928, 372, 1]cuda:0" = convolution_backward_2[0]
        getitem_7: "bf16[2048, 1024, 1][1024, 1, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_35: "f32[2048, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
        convert_element_type_36: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None
        le_1: "b8[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_1: "bf16[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.where.self(le_1, full_default, getitem_6);  le_1 = getitem_6 = None
        sum_4: "bf16[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_1, glu_3, convert_element_type_18, [1024], [4], [0], [1], False, [0], 1, [True, True, False]);  where_1 = glu_3 = convert_element_type_18 = None
        getitem_9: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0" = convolution_backward_3[0]
        getitem_10: "bf16[1024, 512, 8][4096, 8, 1]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        add_1: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_5, getitem_9);  tangents_5 = getitem_9 = None
        convert_element_type_37: "f32[1024, 512, 8][4096, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
        convert_element_type_38: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_4, torch.float32);  sum_4 = None
        convert_element_type_39: "f32[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        convert_element_type_40: "f32[4, 1024, 1493][1528832, 1493, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        slice_5: "f32[4, 512, 1493][1528832, 1493, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_40, 1, 0, 512)
        slice_6: "f32[4, 512, 1493][1528832, 1493, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_40, 1, 512, 1024);  convert_element_type_40 = None
        sigmoid_2: "f32[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_6);  slice_6 = None
        sub_2: "f32[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid_2)
        mul_8: "f32[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, sigmoid_2);  sub_2 = None
        mul_9: "f32[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, slice_5);  mul_8 = slice_5 = None
        mul_10: "f32[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, convert_element_type_39);  mul_9 = None
        mul_11: "f32[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_2, convert_element_type_39);  sigmoid_2 = convert_element_type_39 = None
        cat_2: "f32[4, 1024, 1493][1528832, 1493, 1]cuda:0" = torch.ops.aten.cat.default([mul_11, mul_10], 1);  mul_11 = mul_10 = None
        convert_element_type_41: "bf16[4, 1024, 1493][1528832, 1493, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_2, torch.bfloat16);  cat_2 = None
        sum_5: "bf16[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_41, [0, 2])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_41, relu_3, convert_element_type_16, [1024], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_41 = convert_element_type_16 = None
        getitem_12: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0" = convolution_backward_4[0]
        getitem_13: "bf16[1024, 512, 1][512, 1, 1]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_42: "f32[1024, 512, 1][512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
        convert_element_type_43: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_5, torch.float32);  sum_5 = None
        le_2: "b8[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_2: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.where.self(le_2, full_default, getitem_12);  le_2 = getitem_12 = None
        sum_6: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_2, glu_2, convert_element_type_14, [512], [4], [0], [1], False, [0], 1, [True, True, False]);  where_2 = glu_2 = convert_element_type_14 = None
        getitem_15: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0" = convolution_backward_5[0]
        getitem_16: "bf16[512, 256, 8][2048, 8, 1]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        add_2: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_4, getitem_15);  tangents_4 = getitem_15 = None
        convert_element_type_44: "f32[512, 256, 8][2048, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_16, torch.float32);  getitem_16 = None
        convert_element_type_45: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_6, torch.float32);  sum_6 = None
        convert_element_type_46: "f32[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2, torch.float32);  add_2 = None
        convert_element_type_47: "f32[4, 512, 5979][3061248, 5979, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        slice_7: "f32[4, 256, 5979][3061248, 5979, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_47, 1, 0, 256)
        slice_8: "f32[4, 256, 5979][3061248, 5979, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_47, 1, 256, 512);  convert_element_type_47 = None
        sigmoid_3: "f32[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_8);  slice_8 = None
        sub_3: "f32[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid_3)
        mul_12: "f32[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, sigmoid_3);  sub_3 = None
        mul_13: "f32[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, slice_7);  mul_12 = slice_7 = None
        mul_14: "f32[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, convert_element_type_46);  mul_13 = None
        mul_15: "f32[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_3, convert_element_type_46);  sigmoid_3 = convert_element_type_46 = None
        cat_3: "f32[4, 512, 5979][3061248, 5979, 1]cuda:0" = torch.ops.aten.cat.default([mul_15, mul_14], 1);  mul_15 = mul_14 = None
        convert_element_type_48: "bf16[4, 512, 5979][3061248, 5979, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_3, torch.bfloat16);  cat_3 = None
        sum_7: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_48, [0, 2])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_48, relu_2, convert_element_type_12, [512], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_48 = convert_element_type_12 = None
        getitem_18: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0" = convolution_backward_6[0]
        getitem_19: "bf16[512, 256, 1][256, 1, 1]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_49: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
        convert_element_type_50: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None
        le_3: "b8[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_3: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.where.self(le_3, full_default, getitem_18);  le_3 = getitem_18 = None
        sum_8: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_3, glu_1, convert_element_type_10, [256], [4], [0], [1], False, [0], 1, [True, True, False]);  where_3 = glu_1 = convert_element_type_10 = None
        getitem_21: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0" = convolution_backward_7[0]
        getitem_22: "bf16[256, 128, 8][1024, 8, 1]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        add_3: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_3, getitem_21);  tangents_3 = getitem_21 = None
        convert_element_type_51: "f32[256, 128, 8][1024, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_22, torch.float32);  getitem_22 = None
        convert_element_type_52: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
        convert_element_type_53: "f32[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        convert_element_type_54: "f32[4, 256, 23923][6124288, 23923, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        slice_9: "f32[4, 128, 23923][6124288, 23923, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_54, 1, 0, 128)
        slice_10: "f32[4, 128, 23923][6124288, 23923, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_54, 1, 128, 256);  convert_element_type_54 = None
        sigmoid_4: "f32[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_10);  slice_10 = None
        sub_4: "f32[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid_4)
        mul_16: "f32[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, sigmoid_4);  sub_4 = None
        mul_17: "f32[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, slice_9);  mul_16 = slice_9 = None
        mul_18: "f32[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, convert_element_type_53);  mul_17 = None
        mul_19: "f32[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_4, convert_element_type_53);  sigmoid_4 = convert_element_type_53 = None
        cat_4: "f32[4, 256, 23923][6124288, 23923, 1]cuda:0" = torch.ops.aten.cat.default([mul_19, mul_18], 1);  mul_19 = mul_18 = None
        convert_element_type_55: "bf16[4, 256, 23923][6124288, 23923, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_4, torch.bfloat16);  cat_4 = None
        sum_9: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_55, [0, 2])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_55, relu_1, convert_element_type_8, [256], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_55 = convert_element_type_8 = None
        getitem_24: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0" = convolution_backward_8[0]
        getitem_25: "bf16[256, 128, 1][128, 1, 1]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_56: "f32[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
        convert_element_type_57: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_9, torch.float32);  sum_9 = None
        le_4: "b8[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_4: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.where.self(le_4, full_default, getitem_24);  le_4 = getitem_24 = None
        sum_10: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_4, glu, convert_element_type_6, [128], [4], [0], [1], False, [0], 1, [True, True, False]);  where_4 = glu = convert_element_type_6 = None
        getitem_27: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0" = convolution_backward_9[0]
        getitem_28: "bf16[128, 64, 8][512, 8, 1]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        add_4: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, getitem_27);  tangents_2 = getitem_27 = None
        convert_element_type_58: "f32[128, 64, 8][512, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_28, torch.float32);  getitem_28 = None
        convert_element_type_59: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_10, torch.float32);  sum_10 = None
        convert_element_type_60: "f32[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        convert_element_type_61: "f32[4, 128, 95696][12249088, 95696, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        slice_11: "f32[4, 64, 95696][12249088, 95696, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_61, 1, 0, 64)
        slice_12: "f32[4, 64, 95696][12249088, 95696, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_61, 1, 64, 128);  convert_element_type_61 = None
        sigmoid_5: "f32[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.sigmoid.default(slice_12);  slice_12 = None
        sub_5: "f32[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, sigmoid_5)
        mul_20: "f32[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, sigmoid_5);  sub_5 = None
        mul_21: "f32[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, slice_11);  mul_20 = slice_11 = None
        mul_22: "f32[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, convert_element_type_60);  mul_21 = None
        mul_23: "f32[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_5, convert_element_type_60);  sigmoid_5 = convert_element_type_60 = None
        cat_5: "f32[4, 128, 95696][12249088, 95696, 1]cuda:0" = torch.ops.aten.cat.default([mul_23, mul_22], 1);  mul_23 = mul_22 = None
        convert_element_type_62: "bf16[4, 128, 95696][12249088, 95696, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_5, torch.bfloat16);  cat_5 = None
        sum_11: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_62, [0, 2])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_62, relu, convert_element_type_4, [128], [1], [0], [1], False, [0], 1, [True, True, False]);  convert_element_type_62 = convert_element_type_4 = None
        getitem_30: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0" = convolution_backward_10[0]
        getitem_31: "bf16[128, 64, 1][64, 1, 1]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_63: "f32[128, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
        convert_element_type_64: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_11, torch.float32);  sum_11 = None
        le_5: "b8[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_5: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.where.self(le_5, full_default, getitem_30);  le_5 = full_default = getitem_30 = None
        sum_12: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_5, convert_element_type_2, convert_element_type_1, [64], [4], [0], [1], False, [0], 1, [False, True, False]);  where_5 = convert_element_type_2 = convert_element_type_1 = None
        getitem_34: "bf16[64, 2, 8][16, 8, 1]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_65: "f32[64, 2, 8][16, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_34, torch.float32);  getitem_34 = None
        convert_element_type_66: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_12, torch.float32);  sum_12 = None
        return (convert_element_type_65, convert_element_type_66, None, convert_element_type_63, convert_element_type_64, convert_element_type_58, convert_element_type_59, convert_element_type_56, convert_element_type_57, convert_element_type_51, convert_element_type_52, convert_element_type_49, convert_element_type_50, convert_element_type_44, convert_element_type_45, convert_element_type_42, convert_element_type_43, convert_element_type_37, convert_element_type_38, convert_element_type_35, convert_element_type_36, convert_element_type_30, convert_element_type_31, convert_element_type_28, convert_element_type_29)
