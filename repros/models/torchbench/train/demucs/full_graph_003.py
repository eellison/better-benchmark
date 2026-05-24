import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "f32[4096, 2048, 3]", primals_5: "f32[2048, 1024, 8]", primals_8: "f32[2048, 1024, 3]", primals_10: "f32[1024, 512, 8]", primals_13: "f32[1024, 512, 3]", primals_15: "f32[512, 256, 8]", primals_18: "f32[512, 256, 3]", primals_20: "f32[256, 128, 8]", primals_23: "f32[256, 128, 3]", primals_25: "f32[128, 64, 8]", primals_28: "f32[128, 64, 3]", primals_30: "f32[64, 8, 8]", add: "f32[64, 2048, 92]", convolution: "f32[64, 4096, 90]", glu: "f32[64, 2048, 90]", add_1: "f32[64, 1024, 364]", convolution_2: "f32[64, 2048, 362]", glu_1: "f32[64, 1024, 362]", add_2: "f32[64, 512, 1452]", convolution_4: "f32[64, 1024, 1450]", glu_2: "f32[64, 512, 1450]", add_3: "f32[64, 256, 5804]", convolution_6: "f32[64, 512, 5802]", glu_3: "f32[64, 256, 5802]", add_4: "f32[64, 128, 23212]", convolution_8: "f32[64, 256, 23210]", glu_4: "f32[64, 128, 23210]", add_5: "f32[64, 64, 92844]", convolution_10: "f32[64, 128, 92842]", glu_5: "f32[64, 64, 92842]", le: "b8[64, 64, 92844]", le_1: "b8[64, 128, 23212]", le_2: "b8[64, 256, 5804]", le_3: "b8[64, 512, 1452]", le_4: "b8[64, 1024, 364]", tangents_1: "f32[64, 4, 2, 371372]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:232 in torch_dynamo_resume_in_forward_at_220, code: x = x.view(x.size(0), self.sources, self.audio_channels, x.size(-1))
        view_1: "f32[64, 8, 371372]" = torch.ops.aten.reshape.default(tangents_1, [64, 8, 371372]);  tangents_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        sum_1: "f32[8]" = torch.ops.aten.sum.dim_IntList(view_1, [0, 2])
        convolution_backward = torch.ops.aten.convolution_backward.default(view_1, glu_5, primals_30, [8], [4], [0], [1], True, [0], 1, [True, True, False]);  view_1 = glu_5 = primals_30 = None
        getitem: "f32[64, 64, 92842]" = convolution_backward[0]
        getitem_1: "f32[64, 8, 8]" = convolution_backward[1];  convolution_backward = None
        slice_6: "f32[64, 64, 92842]" = torch.ops.aten.slice.Tensor(convolution_10, 1, 0, 64)
        slice_7: "f32[64, 64, 92842]" = torch.ops.aten.slice.Tensor(convolution_10, 1, 64, 128);  convolution_10 = None
        sigmoid: "f32[64, 64, 92842]" = torch.ops.aten.sigmoid.default(slice_7);  slice_7 = None
        sub: "f32[64, 64, 92842]" = torch.ops.aten.sub.Tensor(1.0, sigmoid)
        mul: "f32[64, 64, 92842]" = torch.ops.aten.mul.Tensor(sub, sigmoid);  sub = None
        mul_1: "f32[64, 64, 92842]" = torch.ops.aten.mul.Tensor(mul, slice_6);  mul = slice_6 = None
        mul_2: "f32[64, 64, 92842]" = torch.ops.aten.mul.Tensor(mul_1, getitem);  mul_1 = None
        mul_3: "f32[64, 64, 92842]" = torch.ops.aten.mul.Tensor(sigmoid, getitem);  sigmoid = getitem = None
        cat: "f32[64, 128, 92842]" = torch.ops.aten.cat.default([mul_3, mul_2], 1);  mul_3 = mul_2 = None
        sum_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(cat, [0, 2])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(cat, add_5, primals_28, [128], [1], [0], [1], False, [0], 1, [True, True, False]);  cat = add_5 = primals_28 = None
        getitem_3: "f32[64, 64, 92844]" = convolution_backward_1[0]
        getitem_4: "f32[128, 64, 3]" = convolution_backward_1[1];  convolution_backward_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default: "f32[64, 64, 95696]" = torch.ops.aten.full.default([64, 64, 95696], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "f32[64, 64, 95696]" = torch.ops.aten.slice_scatter.default(full_default, getitem_3, 2, 1426, -1426);  full_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[64, 64, 92844]" = torch.ops.aten.where.self(le, full_default_1, getitem_3);  le = getitem_3 = None
        sum_3: "f32[64]" = torch.ops.aten.sum.dim_IntList(where, [0, 2])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where, glu_4, primals_25, [64], [4], [0], [1], True, [0], 1, [True, True, False]);  where = glu_4 = primals_25 = None
        getitem_6: "f32[64, 128, 23210]" = convolution_backward_2[0]
        getitem_7: "f32[128, 64, 8]" = convolution_backward_2[1];  convolution_backward_2 = None
        slice_8: "f32[64, 128, 23210]" = torch.ops.aten.slice.Tensor(convolution_8, 1, 0, 128)
        slice_9: "f32[64, 128, 23210]" = torch.ops.aten.slice.Tensor(convolution_8, 1, 128, 256);  convolution_8 = None
        sigmoid_1: "f32[64, 128, 23210]" = torch.ops.aten.sigmoid.default(slice_9);  slice_9 = None
        sub_1: "f32[64, 128, 23210]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_1)
        mul_4: "f32[64, 128, 23210]" = torch.ops.aten.mul.Tensor(sub_1, sigmoid_1);  sub_1 = None
        mul_5: "f32[64, 128, 23210]" = torch.ops.aten.mul.Tensor(mul_4, slice_8);  mul_4 = slice_8 = None
        mul_6: "f32[64, 128, 23210]" = torch.ops.aten.mul.Tensor(mul_5, getitem_6);  mul_5 = None
        mul_7: "f32[64, 128, 23210]" = torch.ops.aten.mul.Tensor(sigmoid_1, getitem_6);  sigmoid_1 = getitem_6 = None
        cat_1: "f32[64, 256, 23210]" = torch.ops.aten.cat.default([mul_7, mul_6], 1);  mul_7 = mul_6 = None
        sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(cat_1, [0, 2])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(cat_1, add_4, primals_23, [256], [1], [0], [1], False, [0], 1, [True, True, False]);  cat_1 = add_4 = primals_23 = None
        getitem_9: "f32[64, 128, 23212]" = convolution_backward_3[0]
        getitem_10: "f32[256, 128, 3]" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_2: "f32[64, 128, 23923]" = torch.ops.aten.full.default([64, 128, 23923], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_1: "f32[64, 128, 23923]" = torch.ops.aten.slice_scatter.default(full_default_2, getitem_9, 2, 355, -356);  full_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        where_1: "f32[64, 128, 23212]" = torch.ops.aten.where.self(le_1, full_default_1, getitem_9);  le_1 = getitem_9 = None
        sum_5: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_1, glu_3, primals_20, [128], [4], [0], [1], True, [0], 1, [True, True, False]);  where_1 = glu_3 = primals_20 = None
        getitem_12: "f32[64, 256, 5802]" = convolution_backward_4[0]
        getitem_13: "f32[256, 128, 8]" = convolution_backward_4[1];  convolution_backward_4 = None
        slice_10: "f32[64, 256, 5802]" = torch.ops.aten.slice.Tensor(convolution_6, 1, 0, 256)
        slice_11: "f32[64, 256, 5802]" = torch.ops.aten.slice.Tensor(convolution_6, 1, 256, 512);  convolution_6 = None
        sigmoid_2: "f32[64, 256, 5802]" = torch.ops.aten.sigmoid.default(slice_11);  slice_11 = None
        sub_2: "f32[64, 256, 5802]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_2)
        mul_8: "f32[64, 256, 5802]" = torch.ops.aten.mul.Tensor(sub_2, sigmoid_2);  sub_2 = None
        mul_9: "f32[64, 256, 5802]" = torch.ops.aten.mul.Tensor(mul_8, slice_10);  mul_8 = slice_10 = None
        mul_10: "f32[64, 256, 5802]" = torch.ops.aten.mul.Tensor(mul_9, getitem_12);  mul_9 = None
        mul_11: "f32[64, 256, 5802]" = torch.ops.aten.mul.Tensor(sigmoid_2, getitem_12);  sigmoid_2 = getitem_12 = None
        cat_2: "f32[64, 512, 5802]" = torch.ops.aten.cat.default([mul_11, mul_10], 1);  mul_11 = mul_10 = None
        sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(cat_2, [0, 2])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(cat_2, add_3, primals_18, [512], [1], [0], [1], False, [0], 1, [True, True, False]);  cat_2 = add_3 = primals_18 = None
        getitem_15: "f32[64, 256, 5804]" = convolution_backward_5[0]
        getitem_16: "f32[512, 256, 3]" = convolution_backward_5[1];  convolution_backward_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_4: "f32[64, 256, 5979]" = torch.ops.aten.full.default([64, 256, 5979], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_2: "f32[64, 256, 5979]" = torch.ops.aten.slice_scatter.default(full_default_4, getitem_15, 2, 87, -88);  full_default_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        where_2: "f32[64, 256, 5804]" = torch.ops.aten.where.self(le_2, full_default_1, getitem_15);  le_2 = getitem_15 = None
        sum_7: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_2, glu_2, primals_15, [256], [4], [0], [1], True, [0], 1, [True, True, False]);  where_2 = glu_2 = primals_15 = None
        getitem_18: "f32[64, 512, 1450]" = convolution_backward_6[0]
        getitem_19: "f32[512, 256, 8]" = convolution_backward_6[1];  convolution_backward_6 = None
        slice_12: "f32[64, 512, 1450]" = torch.ops.aten.slice.Tensor(convolution_4, 1, 0, 512)
        slice_13: "f32[64, 512, 1450]" = torch.ops.aten.slice.Tensor(convolution_4, 1, 512, 1024);  convolution_4 = None
        sigmoid_3: "f32[64, 512, 1450]" = torch.ops.aten.sigmoid.default(slice_13);  slice_13 = None
        sub_3: "f32[64, 512, 1450]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_3)
        mul_12: "f32[64, 512, 1450]" = torch.ops.aten.mul.Tensor(sub_3, sigmoid_3);  sub_3 = None
        mul_13: "f32[64, 512, 1450]" = torch.ops.aten.mul.Tensor(mul_12, slice_12);  mul_12 = slice_12 = None
        mul_14: "f32[64, 512, 1450]" = torch.ops.aten.mul.Tensor(mul_13, getitem_18);  mul_13 = None
        mul_15: "f32[64, 512, 1450]" = torch.ops.aten.mul.Tensor(sigmoid_3, getitem_18);  sigmoid_3 = getitem_18 = None
        cat_3: "f32[64, 1024, 1450]" = torch.ops.aten.cat.default([mul_15, mul_14], 1);  mul_15 = mul_14 = None
        sum_8: "f32[1024]" = torch.ops.aten.sum.dim_IntList(cat_3, [0, 2])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(cat_3, add_2, primals_13, [1024], [1], [0], [1], False, [0], 1, [True, True, False]);  cat_3 = add_2 = primals_13 = None
        getitem_21: "f32[64, 512, 1452]" = convolution_backward_7[0]
        getitem_22: "f32[1024, 512, 3]" = convolution_backward_7[1];  convolution_backward_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_6: "f32[64, 512, 1493]" = torch.ops.aten.full.default([64, 512, 1493], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_3: "f32[64, 512, 1493]" = torch.ops.aten.slice_scatter.default(full_default_6, getitem_21, 2, 20, -21);  full_default_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        where_3: "f32[64, 512, 1452]" = torch.ops.aten.where.self(le_3, full_default_1, getitem_21);  le_3 = getitem_21 = None
        sum_9: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_3, glu_1, primals_10, [512], [4], [0], [1], True, [0], 1, [True, True, False]);  where_3 = glu_1 = primals_10 = None
        getitem_24: "f32[64, 1024, 362]" = convolution_backward_8[0]
        getitem_25: "f32[1024, 512, 8]" = convolution_backward_8[1];  convolution_backward_8 = None
        slice_14: "f32[64, 1024, 362]" = torch.ops.aten.slice.Tensor(convolution_2, 1, 0, 1024)
        slice_15: "f32[64, 1024, 362]" = torch.ops.aten.slice.Tensor(convolution_2, 1, 1024, 2048);  convolution_2 = None
        sigmoid_4: "f32[64, 1024, 362]" = torch.ops.aten.sigmoid.default(slice_15);  slice_15 = None
        sub_4: "f32[64, 1024, 362]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_4)
        mul_16: "f32[64, 1024, 362]" = torch.ops.aten.mul.Tensor(sub_4, sigmoid_4);  sub_4 = None
        mul_17: "f32[64, 1024, 362]" = torch.ops.aten.mul.Tensor(mul_16, slice_14);  mul_16 = slice_14 = None
        mul_18: "f32[64, 1024, 362]" = torch.ops.aten.mul.Tensor(mul_17, getitem_24);  mul_17 = None
        mul_19: "f32[64, 1024, 362]" = torch.ops.aten.mul.Tensor(sigmoid_4, getitem_24);  sigmoid_4 = getitem_24 = None
        cat_4: "f32[64, 2048, 362]" = torch.ops.aten.cat.default([mul_19, mul_18], 1);  mul_19 = mul_18 = None
        sum_10: "f32[2048]" = torch.ops.aten.sum.dim_IntList(cat_4, [0, 2])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(cat_4, add_1, primals_8, [2048], [1], [0], [1], False, [0], 1, [True, True, False]);  cat_4 = add_1 = primals_8 = None
        getitem_27: "f32[64, 1024, 364]" = convolution_backward_9[0]
        getitem_28: "f32[2048, 1024, 3]" = convolution_backward_9[1];  convolution_backward_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default_8: "f32[64, 1024, 372]" = torch.ops.aten.full.default([64, 1024, 372], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_4: "f32[64, 1024, 372]" = torch.ops.aten.slice_scatter.default(full_default_8, getitem_27, 2, 4, -4);  full_default_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        where_4: "f32[64, 1024, 364]" = torch.ops.aten.where.self(le_4, full_default_1, getitem_27);  le_4 = full_default_1 = getitem_27 = None
        sum_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(where_4, glu, primals_5, [1024], [4], [0], [1], True, [0], 1, [True, True, False]);  where_4 = glu = primals_5 = None
        getitem_30: "f32[64, 2048, 90]" = convolution_backward_10[0]
        getitem_31: "f32[2048, 1024, 8]" = convolution_backward_10[1];  convolution_backward_10 = None
        slice_16: "f32[64, 2048, 90]" = torch.ops.aten.slice.Tensor(convolution, 1, 0, 2048)
        slice_17: "f32[64, 2048, 90]" = torch.ops.aten.slice.Tensor(convolution, 1, 2048, 4096);  convolution = None
        sigmoid_5: "f32[64, 2048, 90]" = torch.ops.aten.sigmoid.default(slice_17);  slice_17 = None
        sub_5: "f32[64, 2048, 90]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_5)
        mul_20: "f32[64, 2048, 90]" = torch.ops.aten.mul.Tensor(sub_5, sigmoid_5);  sub_5 = None
        mul_21: "f32[64, 2048, 90]" = torch.ops.aten.mul.Tensor(mul_20, slice_16);  mul_20 = slice_16 = None
        mul_22: "f32[64, 2048, 90]" = torch.ops.aten.mul.Tensor(mul_21, getitem_30);  mul_21 = None
        mul_23: "f32[64, 2048, 90]" = torch.ops.aten.mul.Tensor(sigmoid_5, getitem_30);  sigmoid_5 = getitem_30 = None
        cat_5: "f32[64, 4096, 90]" = torch.ops.aten.cat.default([mul_23, mul_22], 1);  mul_23 = mul_22 = None
        sum_12: "f32[4096]" = torch.ops.aten.sum.dim_IntList(cat_5, [0, 2])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(cat_5, add, primals_3, [4096], [1], [0], [1], False, [0], 1, [True, True, False]);  cat_5 = add = primals_3 = None
        getitem_33: "f32[64, 2048, 92]" = convolution_backward_11[0]
        getitem_34: "f32[4096, 2048, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
        return (getitem_33, getitem_33, getitem_34, sum_12, getitem_31, sum_11, slice_scatter_4, getitem_28, sum_10, getitem_25, sum_9, slice_scatter_3, getitem_22, sum_8, getitem_19, sum_7, slice_scatter_2, getitem_16, sum_6, getitem_13, sum_5, slice_scatter_1, getitem_10, sum_4, getitem_7, sum_3, slice_scatter, getitem_4, sum_2, getitem_1, sum_1)
