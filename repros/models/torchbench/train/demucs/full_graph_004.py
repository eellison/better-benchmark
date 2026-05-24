import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 2, 8]", primals_3: "f32[64, 2, 382788]", primals_4: "f32[128, 64, 1]", primals_6: "f32[128, 64, 8]", primals_8: "f32[256, 128, 1]", primals_10: "f32[256, 128, 8]", primals_12: "f32[512, 256, 1]", primals_14: "f32[512, 256, 8]", primals_16: "f32[1024, 512, 1]", primals_18: "f32[1024, 512, 8]", primals_20: "f32[2048, 1024, 1]", primals_22: "f32[2048, 1024, 8]", primals_24: "f32[4096, 2048, 1]", relu: "f32[64, 64, 95696]", convolution_1: "f32[64, 128, 95696]", glu: "f32[64, 64, 95696]", relu_1: "f32[64, 128, 23923]", convolution_3: "f32[64, 256, 23923]", glu_1: "f32[64, 128, 23923]", relu_2: "f32[64, 256, 5979]", convolution_5: "f32[64, 512, 5979]", glu_2: "f32[64, 256, 5979]", relu_3: "f32[64, 512, 1493]", convolution_7: "f32[64, 1024, 1493]", glu_3: "f32[64, 512, 1493]", relu_4: "f32[64, 1024, 372]", convolution_9: "f32[64, 2048, 372]", glu_4: "f32[64, 1024, 372]", relu_5: "f32[64, 2048, 92]", convolution_11: "f32[64, 4096, 92]", tangents_1: "f32[64, 2048, 92]", tangents_2: "f32[64, 64, 95696]", tangents_3: "f32[64, 128, 23923]", tangents_4: "f32[64, 256, 5979]", tangents_5: "f32[64, 512, 1493]", tangents_6: "f32[64, 1024, 372]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:215 in forward, code: x = encode(x)
        slice_1: "f32[64, 2048, 92]" = torch.ops.aten.slice.Tensor(convolution_11, 1, 0, 2048)
        slice_2: "f32[64, 2048, 92]" = torch.ops.aten.slice.Tensor(convolution_11, 1, 2048, 4096);  convolution_11 = None
        sigmoid: "f32[64, 2048, 92]" = torch.ops.aten.sigmoid.default(slice_2);  slice_2 = None
        sub: "f32[64, 2048, 92]" = torch.ops.aten.sub.Tensor(1.0, sigmoid)
        mul: "f32[64, 2048, 92]" = torch.ops.aten.mul.Tensor(sub, sigmoid);  sub = None
        mul_1: "f32[64, 2048, 92]" = torch.ops.aten.mul.Tensor(mul, slice_1);  mul = slice_1 = None
        mul_2: "f32[64, 2048, 92]" = torch.ops.aten.mul.Tensor(mul_1, tangents_1);  mul_1 = None
        mul_3: "f32[64, 2048, 92]" = torch.ops.aten.mul.Tensor(sigmoid, tangents_1);  sigmoid = tangents_1 = None
        cat: "f32[64, 4096, 92]" = torch.ops.aten.cat.default([mul_3, mul_2], 1);  mul_3 = mul_2 = None
        sum_1: "f32[4096]" = torch.ops.aten.sum.dim_IntList(cat, [0, 2])
        convolution_backward = torch.ops.aten.convolution_backward.default(cat, relu_5, primals_24, [4096], [1], [0], [1], False, [0], 1, [True, True, False]);  cat = primals_24 = None
        getitem: "f32[64, 2048, 92]" = convolution_backward[0]
        getitem_1: "f32[4096, 2048, 1]" = convolution_backward[1];  convolution_backward = None
        le: "b8[64, 2048, 92]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[64, 2048, 92]" = torch.ops.aten.where.self(le, full_default, getitem);  le = getitem = None
        sum_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(where, glu_4, primals_22, [2048], [4], [0], [1], False, [0], 1, [True, True, False]);  where = glu_4 = primals_22 = None
        getitem_3: "f32[64, 1024, 372]" = convolution_backward_1[0]
        getitem_4: "f32[2048, 1024, 8]" = convolution_backward_1[1];  convolution_backward_1 = None
        add: "f32[64, 1024, 372]" = torch.ops.aten.add.Tensor(tangents_6, getitem_3);  tangents_6 = getitem_3 = None
        slice_3: "f32[64, 1024, 372]" = torch.ops.aten.slice.Tensor(convolution_9, 1, 0, 1024)
        slice_4: "f32[64, 1024, 372]" = torch.ops.aten.slice.Tensor(convolution_9, 1, 1024, 2048);  convolution_9 = None
        sigmoid_1: "f32[64, 1024, 372]" = torch.ops.aten.sigmoid.default(slice_4);  slice_4 = None
        sub_1: "f32[64, 1024, 372]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_1)
        mul_4: "f32[64, 1024, 372]" = torch.ops.aten.mul.Tensor(sub_1, sigmoid_1);  sub_1 = None
        mul_5: "f32[64, 1024, 372]" = torch.ops.aten.mul.Tensor(mul_4, slice_3);  mul_4 = slice_3 = None
        mul_6: "f32[64, 1024, 372]" = torch.ops.aten.mul.Tensor(mul_5, add);  mul_5 = None
        mul_7: "f32[64, 1024, 372]" = torch.ops.aten.mul.Tensor(sigmoid_1, add);  sigmoid_1 = add = None
        cat_1: "f32[64, 2048, 372]" = torch.ops.aten.cat.default([mul_7, mul_6], 1);  mul_7 = mul_6 = None
        sum_3: "f32[2048]" = torch.ops.aten.sum.dim_IntList(cat_1, [0, 2])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(cat_1, relu_4, primals_20, [2048], [1], [0], [1], False, [0], 1, [True, True, False]);  cat_1 = primals_20 = None
        getitem_6: "f32[64, 1024, 372]" = convolution_backward_2[0]
        getitem_7: "f32[2048, 1024, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
        le_1: "b8[64, 1024, 372]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_1: "f32[64, 1024, 372]" = torch.ops.aten.where.self(le_1, full_default, getitem_6);  le_1 = getitem_6 = None
        sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_1, glu_3, primals_18, [1024], [4], [0], [1], False, [0], 1, [True, True, False]);  where_1 = glu_3 = primals_18 = None
        getitem_9: "f32[64, 512, 1493]" = convolution_backward_3[0]
        getitem_10: "f32[1024, 512, 8]" = convolution_backward_3[1];  convolution_backward_3 = None
        add_1: "f32[64, 512, 1493]" = torch.ops.aten.add.Tensor(tangents_5, getitem_9);  tangents_5 = getitem_9 = None
        slice_5: "f32[64, 512, 1493]" = torch.ops.aten.slice.Tensor(convolution_7, 1, 0, 512)
        slice_6: "f32[64, 512, 1493]" = torch.ops.aten.slice.Tensor(convolution_7, 1, 512, 1024);  convolution_7 = None
        sigmoid_2: "f32[64, 512, 1493]" = torch.ops.aten.sigmoid.default(slice_6);  slice_6 = None
        sub_2: "f32[64, 512, 1493]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_2)
        mul_8: "f32[64, 512, 1493]" = torch.ops.aten.mul.Tensor(sub_2, sigmoid_2);  sub_2 = None
        mul_9: "f32[64, 512, 1493]" = torch.ops.aten.mul.Tensor(mul_8, slice_5);  mul_8 = slice_5 = None
        mul_10: "f32[64, 512, 1493]" = torch.ops.aten.mul.Tensor(mul_9, add_1);  mul_9 = None
        mul_11: "f32[64, 512, 1493]" = torch.ops.aten.mul.Tensor(sigmoid_2, add_1);  sigmoid_2 = add_1 = None
        cat_2: "f32[64, 1024, 1493]" = torch.ops.aten.cat.default([mul_11, mul_10], 1);  mul_11 = mul_10 = None
        sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(cat_2, [0, 2])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(cat_2, relu_3, primals_16, [1024], [1], [0], [1], False, [0], 1, [True, True, False]);  cat_2 = primals_16 = None
        getitem_12: "f32[64, 512, 1493]" = convolution_backward_4[0]
        getitem_13: "f32[1024, 512, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
        le_2: "b8[64, 512, 1493]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_2: "f32[64, 512, 1493]" = torch.ops.aten.where.self(le_2, full_default, getitem_12);  le_2 = getitem_12 = None
        sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_2, glu_2, primals_14, [512], [4], [0], [1], False, [0], 1, [True, True, False]);  where_2 = glu_2 = primals_14 = None
        getitem_15: "f32[64, 256, 5979]" = convolution_backward_5[0]
        getitem_16: "f32[512, 256, 8]" = convolution_backward_5[1];  convolution_backward_5 = None
        add_2: "f32[64, 256, 5979]" = torch.ops.aten.add.Tensor(tangents_4, getitem_15);  tangents_4 = getitem_15 = None
        slice_7: "f32[64, 256, 5979]" = torch.ops.aten.slice.Tensor(convolution_5, 1, 0, 256)
        slice_8: "f32[64, 256, 5979]" = torch.ops.aten.slice.Tensor(convolution_5, 1, 256, 512);  convolution_5 = None
        sigmoid_3: "f32[64, 256, 5979]" = torch.ops.aten.sigmoid.default(slice_8);  slice_8 = None
        sub_3: "f32[64, 256, 5979]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_3)
        mul_12: "f32[64, 256, 5979]" = torch.ops.aten.mul.Tensor(sub_3, sigmoid_3);  sub_3 = None
        mul_13: "f32[64, 256, 5979]" = torch.ops.aten.mul.Tensor(mul_12, slice_7);  mul_12 = slice_7 = None
        mul_14: "f32[64, 256, 5979]" = torch.ops.aten.mul.Tensor(mul_13, add_2);  mul_13 = None
        mul_15: "f32[64, 256, 5979]" = torch.ops.aten.mul.Tensor(sigmoid_3, add_2);  sigmoid_3 = add_2 = None
        cat_3: "f32[64, 512, 5979]" = torch.ops.aten.cat.default([mul_15, mul_14], 1);  mul_15 = mul_14 = None
        sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(cat_3, [0, 2])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(cat_3, relu_2, primals_12, [512], [1], [0], [1], False, [0], 1, [True, True, False]);  cat_3 = primals_12 = None
        getitem_18: "f32[64, 256, 5979]" = convolution_backward_6[0]
        getitem_19: "f32[512, 256, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
        le_3: "b8[64, 256, 5979]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_3: "f32[64, 256, 5979]" = torch.ops.aten.where.self(le_3, full_default, getitem_18);  le_3 = getitem_18 = None
        sum_8: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_3, glu_1, primals_10, [256], [4], [0], [1], False, [0], 1, [True, True, False]);  where_3 = glu_1 = primals_10 = None
        getitem_21: "f32[64, 128, 23923]" = convolution_backward_7[0]
        getitem_22: "f32[256, 128, 8]" = convolution_backward_7[1];  convolution_backward_7 = None
        add_3: "f32[64, 128, 23923]" = torch.ops.aten.add.Tensor(tangents_3, getitem_21);  tangents_3 = getitem_21 = None
        slice_9: "f32[64, 128, 23923]" = torch.ops.aten.slice.Tensor(convolution_3, 1, 0, 128)
        slice_10: "f32[64, 128, 23923]" = torch.ops.aten.slice.Tensor(convolution_3, 1, 128, 256);  convolution_3 = None
        sigmoid_4: "f32[64, 128, 23923]" = torch.ops.aten.sigmoid.default(slice_10);  slice_10 = None
        sub_4: "f32[64, 128, 23923]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_4)
        mul_16: "f32[64, 128, 23923]" = torch.ops.aten.mul.Tensor(sub_4, sigmoid_4);  sub_4 = None
        mul_17: "f32[64, 128, 23923]" = torch.ops.aten.mul.Tensor(mul_16, slice_9);  mul_16 = slice_9 = None
        mul_18: "f32[64, 128, 23923]" = torch.ops.aten.mul.Tensor(mul_17, add_3);  mul_17 = None
        mul_19: "f32[64, 128, 23923]" = torch.ops.aten.mul.Tensor(sigmoid_4, add_3);  sigmoid_4 = add_3 = None
        cat_4: "f32[64, 256, 23923]" = torch.ops.aten.cat.default([mul_19, mul_18], 1);  mul_19 = mul_18 = None
        sum_9: "f32[256]" = torch.ops.aten.sum.dim_IntList(cat_4, [0, 2])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(cat_4, relu_1, primals_8, [256], [1], [0], [1], False, [0], 1, [True, True, False]);  cat_4 = primals_8 = None
        getitem_24: "f32[64, 128, 23923]" = convolution_backward_8[0]
        getitem_25: "f32[256, 128, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
        le_4: "b8[64, 128, 23923]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_4: "f32[64, 128, 23923]" = torch.ops.aten.where.self(le_4, full_default, getitem_24);  le_4 = getitem_24 = None
        sum_10: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_4, glu, primals_6, [128], [4], [0], [1], False, [0], 1, [True, True, False]);  where_4 = glu = primals_6 = None
        getitem_27: "f32[64, 64, 95696]" = convolution_backward_9[0]
        getitem_28: "f32[128, 64, 8]" = convolution_backward_9[1];  convolution_backward_9 = None
        add_4: "f32[64, 64, 95696]" = torch.ops.aten.add.Tensor(tangents_2, getitem_27);  tangents_2 = getitem_27 = None
        slice_11: "f32[64, 64, 95696]" = torch.ops.aten.slice.Tensor(convolution_1, 1, 0, 64)
        slice_12: "f32[64, 64, 95696]" = torch.ops.aten.slice.Tensor(convolution_1, 1, 64, 128);  convolution_1 = None
        sigmoid_5: "f32[64, 64, 95696]" = torch.ops.aten.sigmoid.default(slice_12);  slice_12 = None
        sub_5: "f32[64, 64, 95696]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_5)
        mul_20: "f32[64, 64, 95696]" = torch.ops.aten.mul.Tensor(sub_5, sigmoid_5);  sub_5 = None
        mul_21: "f32[64, 64, 95696]" = torch.ops.aten.mul.Tensor(mul_20, slice_11);  mul_20 = slice_11 = None
        mul_22: "f32[64, 64, 95696]" = torch.ops.aten.mul.Tensor(mul_21, add_4);  mul_21 = None
        mul_23: "f32[64, 64, 95696]" = torch.ops.aten.mul.Tensor(sigmoid_5, add_4);  sigmoid_5 = add_4 = None
        cat_5: "f32[64, 128, 95696]" = torch.ops.aten.cat.default([mul_23, mul_22], 1);  mul_23 = mul_22 = None
        sum_11: "f32[128]" = torch.ops.aten.sum.dim_IntList(cat_5, [0, 2])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(cat_5, relu, primals_4, [128], [1], [0], [1], False, [0], 1, [True, True, False]);  cat_5 = primals_4 = None
        getitem_30: "f32[64, 64, 95696]" = convolution_backward_10[0]
        getitem_31: "f32[128, 64, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
        le_5: "b8[64, 64, 95696]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_5: "f32[64, 64, 95696]" = torch.ops.aten.where.self(le_5, full_default, getitem_30);  le_5 = full_default = getitem_30 = None
        sum_12: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_5, primals_3, primals_1, [64], [4], [0], [1], False, [0], 1, [False, True, False]);  where_5 = primals_3 = primals_1 = None
        getitem_34: "f32[64, 2, 8]" = convolution_backward_11[1];  convolution_backward_11 = None
        return (getitem_34, sum_12, None, getitem_31, sum_11, getitem_28, sum_10, getitem_25, sum_9, getitem_22, sum_8, getitem_19, sum_7, getitem_16, sum_6, getitem_13, sum_5, getitem_10, sum_4, getitem_7, sum_3, getitem_4, sum_2, getitem_1, sum_1)
