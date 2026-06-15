class GraphModule(torch.nn.Module):
    def forward(self, convert_element_type_1: "bf16[64, 3, 11, 11][363, 1, 33, 3]cuda:0", convert_element_type_2: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", getitem: "bf16[128, 64, 27, 27][46656, 1, 1728, 64]cuda:0", getitem_1: "i8[128, 64, 27, 27][46656, 1, 1728, 64]cuda:0", convert_element_type_4: "bf16[192, 64, 5, 5][1600, 1, 320, 64]cuda:0", getitem_2: "bf16[128, 192, 13, 13][32448, 1, 2496, 192]cuda:0", getitem_3: "i8[128, 192, 13, 13][32448, 1, 2496, 192]cuda:0", convert_element_type_6: "bf16[384, 192, 3, 3][1728, 1, 576, 192]cuda:0", relu_2: "bf16[128, 384, 13, 13][64896, 1, 4992, 384]cuda:0", convert_element_type_8: "bf16[256, 384, 3, 3][3456, 1, 1152, 384]cuda:0", relu_3: "bf16[128, 256, 13, 13][43264, 1, 3328, 256]cuda:0", convert_element_type_10: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", getitem_4: "bf16[128, 256, 6, 6][9216, 36, 6, 1]cuda:0", getitem_5: "i8[128, 256, 6, 6][9216, 36, 6, 1]cuda:0", gt: "b8[128, 9216][9216, 1]cuda:0", mul_1: "bf16[128, 9216][9216, 1]cuda:0", gt_1: "b8[128, 4096][4096, 1]cuda:0", mul_3: "bf16[128, 4096][4096, 1]cuda:0", relu_6: "bf16[128, 4096][4096, 1]cuda:0", permute_3: "bf16[1000, 4096][4096, 1]cuda:0", permute_7: "bf16[4096, 4096][4096, 1]cuda:0", le_1: "b8[128, 4096][4096, 1]cuda:0", permute_11: "bf16[4096, 9216][9216, 1]cuda:0", le_2: "b8[128, 256, 13, 13][43264, 169, 13, 1]cuda:0", le_5: "b8[128, 192, 27, 27][139968, 1, 5184, 192]cuda:0", le_6: "b8[128, 64, 55, 55][193600, 1, 3520, 64]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        mm: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_3);  permute_3 = None
        permute_4: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_4, relu_6);  permute_4 = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_30: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_31: "f32[1000, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_32: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_30, torch.float32);  convert_element_type_30 = None
        le: "b8[128, 4096][4096, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, mm);  le = mm = None
        mm_2: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(where, permute_7);  permute_7 = None
        permute_8: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(where, [1, 0])
        mm_3: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_8, mul_3);  permute_8 = mul_3 = None
        sum_2: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(where, [0], True, dtype = torch.float32);  where = None
        view_2: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [4096]);  sum_2 = None
        convert_element_type_37: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_38: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_39: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_37, torch.float32);  convert_element_type_37 = None
        convert_element_type_40: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_4: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_40, 2.0);  convert_element_type_40 = None
        mul_5: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mm_2, mul_4);  mm_2 = mul_4 = None
        where_1: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.where.self(le_1, full_default, mul_5);  le_1 = mul_5 = None
        mm_4: "bf16[128, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(where_1, permute_11);  permute_11 = None
        permute_12: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_5: "bf16[4096, 9216][9216, 1]cuda:0" = torch.ops.aten.mm.default(permute_12, mul_1);  permute_12 = mul_1 = None
        sum_3: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_1, [0], True, dtype = torch.float32);  where_1 = None
        view_3: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
        convert_element_type_45: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_46: "f32[4096, 9216][9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_47: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_45, torch.float32);  convert_element_type_45 = None
        convert_element_type_48: "bf16[128, 9216][9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_6: "bf16[128, 9216][9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_48, 2.0);  convert_element_type_48 = None
        mul_7: "bf16[128, 9216][9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mm_4, mul_6);  mm_4 = mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:50 in forward, code: x = torch.flatten(x, 1)
        view_4: "bf16[128, 256, 6, 6][9216, 36, 6, 1]cuda:0" = torch.ops.aten.reshape.default(mul_7, [128, 256, 6, 6]);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:49 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d_backward: "bf16[128, 256, 6, 6][9216, 36, 6, 1]cuda:0" = torch.ops.aten._adaptive_avg_pool2d_backward.default(view_4, getitem_4);  view_4 = getitem_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        full_default_2: "f32[32768, 169][169, 1]cuda:0" = torch.ops.aten.full.default([32768, 169], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_5: "bf16[32768, 36][36, 1]cuda:0" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d_backward, [32768, 36]);  _adaptive_avg_pool2d_backward = None
        _low_memory_max_pool_offsets_to_indices_2: "i64[128, 256, 6, 6][9216, 36, 6, 1]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_5, [3, 3], [13, 13], [2, 2], [0, 0], [1, 1]);  getitem_5 = None
        view_6: "i64[32768, 36][36, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_2, [32768, 36]);  _low_memory_max_pool_offsets_to_indices_2 = None
        convert_element_type_49: "f32[32768, 36][36, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        scatter_add: "f32[32768, 169][169, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_2, 1, view_6, convert_element_type_49);  full_default_2 = view_6 = convert_element_type_49 = None
        view_7: "f32[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add, [128, 256, 13, 13]);  scatter_add = None
        convert_element_type_50: "bf16[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        where_2: "bf16[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.where.self(le_2, full_default, convert_element_type_50);  le_2 = convert_element_type_50 = None
        sum_4: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(where_2, relu_3, convert_element_type_10, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = convert_element_type_10 = None
        getitem_6: "bf16[128, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = convolution_backward[0]
        getitem_7: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_51: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
        convert_element_type_52: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_4, torch.float32);  sum_4 = None
        le_3: "b8[128, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_3: "bf16[128, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = torch.ops.aten.where.self(le_3, full_default, getitem_6);  le_3 = getitem_6 = None
        sum_5: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_3, relu_2, convert_element_type_8, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = convert_element_type_8 = None
        getitem_9: "bf16[128, 384, 13, 13][64896, 1, 4992, 384]cuda:0" = convolution_backward_1[0]
        getitem_10: "bf16[256, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_53: "f32[256, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
        convert_element_type_54: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_5, torch.float32);  sum_5 = None
        le_4: "b8[128, 384, 13, 13][64896, 1, 4992, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_4: "bf16[128, 384, 13, 13][64896, 1, 4992, 384]cuda:0" = torch.ops.aten.where.self(le_4, full_default, getitem_9);  le_4 = getitem_9 = None
        sum_6: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, getitem_2, convert_element_type_6, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = getitem_2 = convert_element_type_6 = None
        getitem_12: "bf16[128, 192, 13, 13][32448, 1, 2496, 192]cuda:0" = convolution_backward_2[0]
        getitem_13: "bf16[384, 192, 3, 3][1728, 1, 576, 192]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_55: "f32[384, 192, 3, 3][1728, 1, 576, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
        convert_element_type_56: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_6, torch.float32);  sum_6 = None
        full_default_6: "f32[24576, 729][729, 1]cuda:0" = torch.ops.aten.full.default([24576, 729], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_8: "bf16[24576, 169][169, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_12, [24576, 169]);  getitem_12 = None
        _low_memory_max_pool_offsets_to_indices_1: "i64[128, 192, 13, 13][32448, 1, 2496, 192]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [27, 27], [2, 2], [0, 0], [1, 1]);  getitem_3 = None
        view_9: "i64[24576, 169][169, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_1, [24576, 169]);  _low_memory_max_pool_offsets_to_indices_1 = None
        convert_element_type_57: "f32[24576, 169][169, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        scatter_add_1: "f32[24576, 729][729, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_6, 1, view_9, convert_element_type_57);  full_default_6 = view_9 = convert_element_type_57 = None
        view_10: "f32[128, 192, 27, 27][139968, 729, 27, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add_1, [128, 192, 27, 27]);  scatter_add_1 = None
        convert_element_type_58: "bf16[128, 192, 27, 27][139968, 729, 27, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        where_5: "bf16[128, 192, 27, 27][139968, 1, 5184, 192]cuda:0" = torch.ops.aten.where.self(le_5, full_default, convert_element_type_58);  le_5 = convert_element_type_58 = None
        sum_7: "bf16[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, getitem, convert_element_type_4, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = getitem = convert_element_type_4 = None
        getitem_15: "bf16[128, 64, 27, 27][46656, 1, 1728, 64]cuda:0" = convolution_backward_3[0]
        getitem_16: "bf16[192, 64, 5, 5][1600, 1, 320, 64]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_59: "f32[192, 64, 5, 5][1600, 1, 320, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_16, torch.float32);  getitem_16 = None
        convert_element_type_60: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None
        full_default_8: "f32[8192, 3025][3025, 1]cuda:0" = torch.ops.aten.full.default([8192, 3025], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_11: "bf16[8192, 729][729, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_15, [8192, 729]);  getitem_15 = None
        _low_memory_max_pool_offsets_to_indices: "i64[128, 64, 27, 27][46656, 1, 1728, 64]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_1, [3, 3], [55, 55], [2, 2], [0, 0], [1, 1]);  getitem_1 = None
        view_12: "i64[8192, 729][729, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [8192, 729]);  _low_memory_max_pool_offsets_to_indices = None
        convert_element_type_61: "f32[8192, 729][729, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        scatter_add_2: "f32[8192, 3025][3025, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_8, 1, view_12, convert_element_type_61);  full_default_8 = view_12 = convert_element_type_61 = None
        view_13: "f32[128, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add_2, [128, 64, 55, 55]);  scatter_add_2 = None
        convert_element_type_62: "bf16[128, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.bfloat16);  view_13 = None
        where_6: "bf16[128, 64, 55, 55][193600, 1, 3520, 64]cuda:0" = torch.ops.aten.where.self(le_6, full_default, convert_element_type_62);  le_6 = full_default = convert_element_type_62 = None
        sum_8: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_6, convert_element_type_2, convert_element_type_1, [64], [4, 4], [2, 2], [1, 1], False, [0, 0], 1, [False, True, False]);  where_6 = convert_element_type_2 = convert_element_type_1 = None
        getitem_19: "bf16[64, 3, 11, 11][363, 1, 33, 3]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_63: "f32[64, 3, 11, 11][363, 1, 33, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
        convert_element_type_64: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
        return (convert_element_type_63, convert_element_type_64, None, convert_element_type_59, convert_element_type_60, convert_element_type_55, convert_element_type_56, convert_element_type_53, convert_element_type_54, convert_element_type_51, convert_element_type_52, convert_element_type_46, convert_element_type_47, convert_element_type_38, convert_element_type_39, convert_element_type_31, convert_element_type_32)
