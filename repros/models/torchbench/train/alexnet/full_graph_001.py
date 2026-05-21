class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 11, 11]", primals_3: "f32[1024, 3, 224, 224]", primals_4: "f32[192, 64, 5, 5]", primals_6: "f32[384, 192, 3, 3]", primals_8: "f32[256, 384, 3, 3]", primals_10: "f32[256, 256, 3, 3]", primals_12: "f32[4096, 9216]", primals_14: "f32[4096, 4096]", primals_16: "f32[1000, 4096]", getitem: "f32[1024, 64, 27, 27]", getitem_1: "i8[1024, 64, 27, 27]", getitem_2: "f32[1024, 192, 13, 13]", getitem_3: "i8[1024, 192, 13, 13]", relu_2: "f32[1024, 384, 13, 13]", relu_3: "f32[1024, 256, 13, 13]", getitem_4: "f32[1024, 256, 6, 6]", getitem_5: "i8[1024, 256, 6, 6]", gt: "b8[1024, 9216]", mul_1: "f32[1024, 9216]", gt_1: "b8[1024, 4096]", mul_3: "f32[1024, 4096]", relu_6: "f32[1024, 4096]", le_1: "b8[1024, 4096]", le_2: "b8[1024, 256, 13, 13]", le_5: "b8[1024, 192, 27, 27]", le_6: "b8[1024, 64, 55, 55]", tangents_1: "f32[1024, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        permute_2: "f32[4096, 1000]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_3: "f32[1000, 4096]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm: "f32[1024, 4096]" = torch.ops.aten.mm.default(tangents_1, permute_3);  permute_3 = None
        permute_4: "f32[1000, 1024]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 4096]" = torch.ops.aten.mm.default(permute_4, relu_6);  permute_4 = None
        sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        le: "b8[1024, 4096]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1024, 4096]" = torch.ops.aten.where.self(le, full_default, mm);  le = mm = None
        permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_7: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_2: "f32[1024, 4096]" = torch.ops.aten.mm.default(where, permute_7);  permute_7 = None
        permute_8: "f32[4096, 1024]" = torch.ops.aten.permute.default(where, [1, 0])
        mm_3: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_8, mul_3);  permute_8 = mul_3 = None
        sum_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        view_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_2, [4096]);  sum_2 = None
        convert_element_type: "f32[1024, 4096]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_4: "f32[1024, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
        mul_5: "f32[1024, 4096]" = torch.ops.aten.mul.Tensor(mm_2, mul_4);  mm_2 = mul_4 = None
        where_1: "f32[1024, 4096]" = torch.ops.aten.where.self(le_1, full_default, mul_5);  le_1 = mul_5 = None
        permute: "f32[9216, 4096]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_11: "f32[4096, 9216]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_4: "f32[1024, 9216]" = torch.ops.aten.mm.default(where_1, permute_11);  permute_11 = None
        permute_12: "f32[4096, 1024]" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_5: "f32[4096, 9216]" = torch.ops.aten.mm.default(permute_12, mul_1);  permute_12 = mul_1 = None
        sum_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        view_3: "f32[4096]" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
        convert_element_type_1: "f32[1024, 9216]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_6: "f32[1024, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 2.0);  convert_element_type_1 = None
        mul_7: "f32[1024, 9216]" = torch.ops.aten.mul.Tensor(mm_4, mul_6);  mm_4 = mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:50 in forward, code: x = torch.flatten(x, 1)
        view_4: "f32[1024, 256, 6, 6]" = torch.ops.aten.reshape.default(mul_7, [1024, 256, 6, 6]);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:49 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d_backward: "f32[1024, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d_backward.default(view_4, getitem_4);  view_4 = getitem_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        full_default_2: "f32[262144, 169]" = torch.ops.aten.full.default([262144, 169], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_5: "f32[262144, 36]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d_backward, [262144, 36]);  _adaptive_avg_pool2d_backward = None
        _low_memory_max_pool_offsets_to_indices_2: "i64[1024, 256, 6, 6]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_5, [3, 3], [13, 13], [2, 2], [0, 0], [1, 1]);  getitem_5 = None
        view_6: "i64[262144, 36]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_2, [262144, 36]);  _low_memory_max_pool_offsets_to_indices_2 = None
        scatter_add: "f32[262144, 169]" = torch.ops.aten.scatter_add.default(full_default_2, 1, view_6, view_5);  full_default_2 = view_6 = view_5 = None
        view_7: "f32[1024, 256, 13, 13]" = torch.ops.aten.reshape.default(scatter_add, [1024, 256, 13, 13]);  scatter_add = None
        where_2: "f32[1024, 256, 13, 13]" = torch.ops.aten.where.self(le_2, full_default, view_7);  le_2 = view_7 = None
        sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(where_2, relu_3, primals_10, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = primals_10 = None
        getitem_6: "f32[1024, 256, 13, 13]" = convolution_backward[0]
        getitem_7: "f32[256, 256, 3, 3]" = convolution_backward[1];  convolution_backward = None
        le_3: "b8[1024, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_3: "f32[1024, 256, 13, 13]" = torch.ops.aten.where.self(le_3, full_default, getitem_6);  le_3 = getitem_6 = None
        sum_5: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_3, relu_2, primals_8, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = primals_8 = None
        getitem_9: "f32[1024, 384, 13, 13]" = convolution_backward_1[0]
        getitem_10: "f32[256, 384, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
        le_4: "b8[1024, 384, 13, 13]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_4: "f32[1024, 384, 13, 13]" = torch.ops.aten.where.self(le_4, full_default, getitem_9);  le_4 = getitem_9 = None
        sum_6: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, getitem_2, primals_6, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = getitem_2 = primals_6 = None
        getitem_12: "f32[1024, 192, 13, 13]" = convolution_backward_2[0]
        getitem_13: "f32[384, 192, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
        full_default_6: "f32[196608, 729]" = torch.ops.aten.full.default([196608, 729], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_8: "f32[196608, 169]" = torch.ops.aten.reshape.default(getitem_12, [196608, 169]);  getitem_12 = None
        _low_memory_max_pool_offsets_to_indices_1: "i64[1024, 192, 13, 13]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [27, 27], [2, 2], [0, 0], [1, 1]);  getitem_3 = None
        view_9: "i64[196608, 169]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_1, [196608, 169]);  _low_memory_max_pool_offsets_to_indices_1 = None
        scatter_add_1: "f32[196608, 729]" = torch.ops.aten.scatter_add.default(full_default_6, 1, view_9, view_8);  full_default_6 = view_9 = view_8 = None
        view_10: "f32[1024, 192, 27, 27]" = torch.ops.aten.reshape.default(scatter_add_1, [1024, 192, 27, 27]);  scatter_add_1 = None
        where_5: "f32[1024, 192, 27, 27]" = torch.ops.aten.where.self(le_5, full_default, view_10);  le_5 = view_10 = None
        sum_7: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, getitem, primals_4, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = getitem = primals_4 = None
        getitem_15: "f32[1024, 64, 27, 27]" = convolution_backward_3[0]
        getitem_16: "f32[192, 64, 5, 5]" = convolution_backward_3[1];  convolution_backward_3 = None
        full_default_8: "f32[65536, 3025]" = torch.ops.aten.full.default([65536, 3025], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_11: "f32[65536, 729]" = torch.ops.aten.reshape.default(getitem_15, [65536, 729]);  getitem_15 = None
        _low_memory_max_pool_offsets_to_indices: "i64[1024, 64, 27, 27]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_1, [3, 3], [55, 55], [2, 2], [0, 0], [1, 1]);  getitem_1 = None
        view_12: "i64[65536, 729]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [65536, 729]);  _low_memory_max_pool_offsets_to_indices = None
        scatter_add_2: "f32[65536, 3025]" = torch.ops.aten.scatter_add.default(full_default_8, 1, view_12, view_11);  full_default_8 = view_12 = view_11 = None
        view_13: "f32[1024, 64, 55, 55]" = torch.ops.aten.reshape.default(scatter_add_2, [1024, 64, 55, 55]);  scatter_add_2 = None
        where_6: "f32[1024, 64, 55, 55]" = torch.ops.aten.where.self(le_6, full_default, view_13);  le_6 = full_default = view_13 = None
        sum_8: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_6, primals_3, primals_1, [64], [4, 4], [2, 2], [1, 1], False, [0, 0], 1, [False, True, False]);  where_6 = primals_3 = primals_1 = None
        getitem_19: "f32[64, 3, 11, 11]" = convolution_backward_4[1];  convolution_backward_4 = None
        return (getitem_19, sum_8, None, getitem_16, sum_7, getitem_13, sum_6, getitem_10, sum_5, getitem_7, sum_4, mm_5, view_3, mm_3, view_2, mm_1, view_1)
