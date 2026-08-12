class GraphModule(torch.nn.Module):
    def forward(self, convert_element_type_1: "bf16[64, 3, 3, 3][27, 1, 9, 3]cuda:0", convert_element_type_2: "bf16[64, 3, 224, 224][150528, 1, 672, 3]cuda:0", relu: "bf16[64, 64, 224, 224][3211264, 1, 14336, 64]cuda:0", convert_element_type_4: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0", getitem: "bf16[64, 64, 112, 112][802816, 1, 7168, 64]cuda:0", getitem_1: "i8[64, 64, 112, 112][802816, 1, 7168, 64]cuda:0", convert_element_type_6: "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0", relu_2: "bf16[64, 128, 112, 112][1605632, 1, 14336, 128]cuda:0", convert_element_type_8: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", getitem_2: "bf16[64, 128, 56, 56][401408, 1, 7168, 128]cuda:0", getitem_3: "i8[64, 128, 56, 56][401408, 1, 7168, 128]cuda:0", convert_element_type_10: "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0", relu_4: "bf16[64, 256, 56, 56][802816, 1, 14336, 256]cuda:0", convert_element_type_12: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", relu_5: "bf16[64, 256, 56, 56][802816, 1, 14336, 256]cuda:0", convert_element_type_14: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", getitem_4: "bf16[64, 256, 28, 28][200704, 1, 7168, 256]cuda:0", getitem_5: "i8[64, 256, 28, 28][200704, 1, 7168, 256]cuda:0", convert_element_type_16: "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0", relu_7: "bf16[64, 512, 28, 28][401408, 1, 14336, 512]cuda:0", convert_element_type_18: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0", relu_8: "bf16[64, 512, 28, 28][401408, 1, 14336, 512]cuda:0", convert_element_type_20: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0", getitem_6: "bf16[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0", getitem_7: "i8[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0", convert_element_type_22: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0", relu_10: "bf16[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0", convert_element_type_24: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0", relu_11: "bf16[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0", convert_element_type_26: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0", getitem_8: "bf16[64, 512, 7, 7][25088, 49, 7, 1]cuda:0", getitem_9: "i8[64, 512, 7, 7][25088, 49, 7, 1]cuda:0", view: "bf16[64, 25088][25088, 1]cuda:0", gt: "b8[64, 4096][4096, 1]cuda:0", mul_1: "bf16[64, 4096][4096, 1]cuda:0", gt_1: "b8[64, 4096][4096, 1]cuda:0", mul_3: "bf16[64, 4096][4096, 1]cuda:0", permute_3: "bf16[1000, 4096][4096, 1]cuda:0", le: "b8[64, 4096][4096, 1]cuda:0", permute_7: "bf16[4096, 4096][4096, 1]cuda:0", le_1: "b8[64, 4096][4096, 1]cuda:0", permute_11: "bf16[4096, 25088][25088, 1]cuda:0", le_2: "b8[64, 512, 14, 14][100352, 196, 14, 1]cuda:0", le_5: "b8[64, 512, 28, 28][401408, 1, 14336, 512]cuda:0", le_8: "b8[64, 256, 56, 56][802816, 1, 14336, 256]cuda:0", le_11: "b8[64, 128, 112, 112][1605632, 1, 14336, 128]cuda:0", le_13: "b8[64, 64, 224, 224][3211264, 1, 14336, 64]cuda:0", tangents_1: "bf16[64, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        mm: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_3);  permute_3 = None
        permute_4: "bf16[1000, 64][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_4, mul_3);  permute_4 = mul_3 = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_46: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_47: "f32[1000, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_48: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_46, torch.float32);  convert_element_type_46 = None
        convert_element_type_49: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_4: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_49, 2.0);  convert_element_type_49 = None
        mul_5: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mm, mul_4);  mm = mul_4 = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, mul_5);  le = mul_5 = None
        mm_2: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(where, permute_7);  permute_7 = None
        permute_8: "bf16[4096, 64][1, 4096]cuda:0" = torch.ops.aten.permute.default(where, [1, 0])
        mm_3: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_8, mul_1);  permute_8 = mul_1 = None
        sum_2: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(where, [0], True, dtype = torch.float32);  where = None
        view_2: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [4096]);  sum_2 = None
        convert_element_type_54: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_55: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_56: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_54, torch.float32);  convert_element_type_54 = None
        convert_element_type_57: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_6: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_57, 2.0);  convert_element_type_57 = None
        mul_7: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mm_2, mul_6);  mm_2 = mul_6 = None
        where_1: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.where.self(le_1, full_default, mul_7);  le_1 = mul_7 = None
        mm_4: "bf16[64, 25088][25088, 1]cuda:0" = torch.ops.aten.mm.default(where_1, permute_11);  permute_11 = None
        permute_12: "bf16[4096, 64][1, 4096]cuda:0" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_5: "bf16[4096, 25088][25088, 1]cuda:0" = torch.ops.aten.mm.default(permute_12, view);  permute_12 = view = None
        sum_3: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_1, [0], True, dtype = torch.float32);  where_1 = None
        view_3: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
        convert_element_type_62: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_63: "f32[4096, 25088][25088, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_64: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_62, torch.float32);  convert_element_type_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:68 in forward, code: x = torch.flatten(x, 1)
        view_4: "bf16[64, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [64, 512, 7, 7]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:67 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d_backward: "bf16[64, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten._adaptive_avg_pool2d_backward.default(view_4, getitem_8);  view_4 = getitem_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        full_default_2: "f32[32768, 196][196, 1]cuda:0" = torch.ops.aten.full.default([32768, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_5: "bf16[32768, 49][49, 1]cuda:0" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d_backward, [32768, 49]);  _adaptive_avg_pool2d_backward = None
        _low_memory_max_pool_offsets_to_indices_4: "i64[64, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_9, [2, 2], [14, 14], [2, 2], [0, 0], [1, 1]);  getitem_9 = None
        view_6: "i64[32768, 49][49, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_4, [32768, 49]);  _low_memory_max_pool_offsets_to_indices_4 = None
        convert_element_type_65: "f32[32768, 49][49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        scatter_add: "f32[32768, 196][196, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_2, 1, view_6, convert_element_type_65);  full_default_2 = view_6 = convert_element_type_65 = None
        view_7: "f32[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add, [64, 512, 14, 14]);  scatter_add = None
        convert_element_type_66: "bf16[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        where_2: "bf16[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_2, full_default, convert_element_type_66);  le_2 = convert_element_type_66 = None
        sum_4: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(where_2, relu_11, convert_element_type_26, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = convert_element_type_26 = None
        getitem_10: "bf16[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = convolution_backward[0]
        getitem_11: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_67: "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
        convert_element_type_68: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_4, torch.float32);  sum_4 = None
        le_3: "b8[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_3: "bf16[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.where.self(le_3, full_default, getitem_10);  le_3 = getitem_10 = None
        sum_5: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_3, relu_10, convert_element_type_24, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = convert_element_type_24 = None
        getitem_13: "bf16[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = convolution_backward_1[0]
        getitem_14: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_69: "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_14, torch.float32);  getitem_14 = None
        convert_element_type_70: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_5, torch.float32);  sum_5 = None
        le_4: "b8[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_4: "bf16[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.where.self(le_4, full_default, getitem_13);  le_4 = getitem_13 = None
        sum_6: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, getitem_6, convert_element_type_22, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = getitem_6 = convert_element_type_22 = None
        getitem_16: "bf16[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = convolution_backward_2[0]
        getitem_17: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_71: "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
        convert_element_type_72: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_6, torch.float32);  sum_6 = None
        full_default_6: "f32[32768, 784][784, 1]cuda:0" = torch.ops.aten.full.default([32768, 784], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_8: "bf16[32768, 196][196, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_16, [32768, 196]);  getitem_16 = None
        _low_memory_max_pool_offsets_to_indices_3: "i64[64, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_7, [2, 2], [28, 28], [2, 2], [0, 0], [1, 1]);  getitem_7 = None
        view_9: "i64[32768, 196][196, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_3, [32768, 196]);  _low_memory_max_pool_offsets_to_indices_3 = None
        convert_element_type_73: "f32[32768, 196][196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        scatter_add_1: "f32[32768, 784][784, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_6, 1, view_9, convert_element_type_73);  full_default_6 = view_9 = convert_element_type_73 = None
        view_10: "f32[64, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add_1, [64, 512, 28, 28]);  scatter_add_1 = None
        convert_element_type_74: "bf16[64, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        where_5: "bf16[64, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.where.self(le_5, full_default, convert_element_type_74);  le_5 = convert_element_type_74 = None
        sum_7: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, relu_8, convert_element_type_20, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = convert_element_type_20 = None
        getitem_19: "bf16[64, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_3[0]
        getitem_20: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_75: "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_20, torch.float32);  getitem_20 = None
        convert_element_type_76: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None
        le_6: "b8[64, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_6: "bf16[64, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.where.self(le_6, full_default, getitem_19);  le_6 = getitem_19 = None
        sum_8: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_6, relu_7, convert_element_type_18, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = convert_element_type_18 = None
        getitem_22: "bf16[64, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_4[0]
        getitem_23: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_77: "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_23, torch.float32);  getitem_23 = None
        convert_element_type_78: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
        le_7: "b8[64, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_7: "bf16[64, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.where.self(le_7, full_default, getitem_22);  le_7 = getitem_22 = None
        sum_9: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_7, getitem_4, convert_element_type_16, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_7 = getitem_4 = convert_element_type_16 = None
        getitem_25: "bf16[64, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = convolution_backward_5[0]
        getitem_26: "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_79: "f32[512, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_26, torch.float32);  getitem_26 = None
        convert_element_type_80: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_9, torch.float32);  sum_9 = None
        full_default_10: "f32[16384, 3136][3136, 1]cuda:0" = torch.ops.aten.full.default([16384, 3136], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_11: "bf16[16384, 784][784, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_25, [16384, 784]);  getitem_25 = None
        _low_memory_max_pool_offsets_to_indices_2: "i64[64, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_5, [2, 2], [56, 56], [2, 2], [0, 0], [1, 1]);  getitem_5 = None
        view_12: "i64[16384, 784][784, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_2, [16384, 784]);  _low_memory_max_pool_offsets_to_indices_2 = None
        convert_element_type_81: "f32[16384, 784][784, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        scatter_add_2: "f32[16384, 3136][3136, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_10, 1, view_12, convert_element_type_81);  full_default_10 = view_12 = convert_element_type_81 = None
        view_13: "f32[64, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add_2, [64, 256, 56, 56]);  scatter_add_2 = None
        convert_element_type_82: "bf16[64, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.bfloat16);  view_13 = None
        where_8: "bf16[64, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.where.self(le_8, full_default, convert_element_type_82);  le_8 = convert_element_type_82 = None
        sum_10: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_8, relu_5, convert_element_type_14, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = convert_element_type_14 = None
        getitem_28: "bf16[64, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = convolution_backward_6[0]
        getitem_29: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_83: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_29, torch.float32);  getitem_29 = None
        convert_element_type_84: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_10, torch.float32);  sum_10 = None
        le_9: "b8[64, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_9: "bf16[64, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.where.self(le_9, full_default, getitem_28);  le_9 = getitem_28 = None
        sum_11: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_9, relu_4, convert_element_type_12, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = convert_element_type_12 = None
        getitem_31: "bf16[64, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = convolution_backward_7[0]
        getitem_32: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_85: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_32, torch.float32);  getitem_32 = None
        convert_element_type_86: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_11, torch.float32);  sum_11 = None
        le_10: "b8[64, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_10: "bf16[64, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.where.self(le_10, full_default, getitem_31);  le_10 = getitem_31 = None
        sum_12: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_10, getitem_2, convert_element_type_10, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = getitem_2 = convert_element_type_10 = None
        getitem_34: "bf16[64, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = convolution_backward_8[0]
        getitem_35: "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_87: "f32[256, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_35, torch.float32);  getitem_35 = None
        convert_element_type_88: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_12, torch.float32);  sum_12 = None
        full_default_14: "f32[8192, 12544][12544, 1]cuda:0" = torch.ops.aten.full.default([8192, 12544], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_14: "bf16[8192, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_34, [8192, 3136]);  getitem_34 = None
        _low_memory_max_pool_offsets_to_indices_1: "i64[64, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [2, 2], [112, 112], [2, 2], [0, 0], [1, 1]);  getitem_3 = None
        view_15: "i64[8192, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_1, [8192, 3136]);  _low_memory_max_pool_offsets_to_indices_1 = None
        convert_element_type_89: "f32[8192, 3136][3136, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        scatter_add_3: "f32[8192, 12544][12544, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_14, 1, view_15, convert_element_type_89);  full_default_14 = view_15 = convert_element_type_89 = None
        view_16: "f32[64, 128, 112, 112][1605632, 12544, 112, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add_3, [64, 128, 112, 112]);  scatter_add_3 = None
        convert_element_type_90: "bf16[64, 128, 112, 112][1605632, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_16, torch.bfloat16);  view_16 = None
        where_11: "bf16[64, 128, 112, 112][1605632, 1, 14336, 128]cuda:0" = torch.ops.aten.where.self(le_11, full_default, convert_element_type_90);  le_11 = convert_element_type_90 = None
        sum_13: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_11, relu_2, convert_element_type_8, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = convert_element_type_8 = None
        getitem_37: "bf16[64, 128, 112, 112][1605632, 1, 14336, 128]cuda:0" = convolution_backward_9[0]
        getitem_38: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_91: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_38, torch.float32);  getitem_38 = None
        convert_element_type_92: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_13, torch.float32);  sum_13 = None
        le_12: "b8[64, 128, 112, 112][1605632, 1, 14336, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_12: "bf16[64, 128, 112, 112][1605632, 1, 14336, 128]cuda:0" = torch.ops.aten.where.self(le_12, full_default, getitem_37);  le_12 = getitem_37 = None
        sum_14: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(where_12, getitem, convert_element_type_6, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_12 = getitem = convert_element_type_6 = None
        getitem_40: "bf16[64, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = convolution_backward_10[0]
        getitem_41: "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_93: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
        convert_element_type_94: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        full_default_17: "f32[4096, 50176][50176, 1]cuda:0" = torch.ops.aten.full.default([4096, 50176], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_17: "bf16[4096, 12544][12544, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_40, [4096, 12544]);  getitem_40 = None
        _low_memory_max_pool_offsets_to_indices: "i64[64, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_1, [2, 2], [224, 224], [2, 2], [0, 0], [1, 1]);  getitem_1 = None
        view_18: "i64[4096, 12544][12544, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [4096, 12544]);  _low_memory_max_pool_offsets_to_indices = None
        convert_element_type_95: "f32[4096, 12544][12544, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        scatter_add_4: "f32[4096, 50176][50176, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_17, 1, view_18, convert_element_type_95);  full_default_17 = view_18 = convert_element_type_95 = None
        view_19: "f32[64, 64, 224, 224][3211264, 50176, 224, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add_4, [64, 64, 224, 224]);  scatter_add_4 = None
        convert_element_type_96: "bf16[64, 64, 224, 224][3211264, 50176, 224, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.bfloat16);  view_19 = None
        where_13: "bf16[64, 64, 224, 224][3211264, 1, 14336, 64]cuda:0" = torch.ops.aten.where.self(le_13, full_default, convert_element_type_96);  le_13 = convert_element_type_96 = None
        sum_15: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_13, relu, convert_element_type_4, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_13 = convert_element_type_4 = None
        getitem_43: "bf16[64, 64, 224, 224][3211264, 1, 14336, 64]cuda:0" = convolution_backward_11[0]
        getitem_44: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_97: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_44, torch.float32);  getitem_44 = None
        convert_element_type_98: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_15, torch.float32);  sum_15 = None
        le_14: "b8[64, 64, 224, 224][3211264, 1, 14336, 64]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_14: "bf16[64, 64, 224, 224][3211264, 1, 14336, 64]cuda:0" = torch.ops.aten.where.self(le_14, full_default, getitem_43);  le_14 = full_default = getitem_43 = None
        sum_16: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_14, convert_element_type_2, convert_element_type_1, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  where_14 = convert_element_type_2 = convert_element_type_1 = None
        getitem_47: "bf16[64, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_99: "f32[64, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
        convert_element_type_100: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_16, torch.float32);  sum_16 = None
        return (convert_element_type_99, convert_element_type_100, None, convert_element_type_97, convert_element_type_98, convert_element_type_93, convert_element_type_94, convert_element_type_91, convert_element_type_92, convert_element_type_87, convert_element_type_88, convert_element_type_85, convert_element_type_86, convert_element_type_83, convert_element_type_84, convert_element_type_79, convert_element_type_80, convert_element_type_77, convert_element_type_78, convert_element_type_75, convert_element_type_76, convert_element_type_71, convert_element_type_72, convert_element_type_69, convert_element_type_70, convert_element_type_67, convert_element_type_68, convert_element_type_63, convert_element_type_64, convert_element_type_55, convert_element_type_56, convert_element_type_47, convert_element_type_48)
