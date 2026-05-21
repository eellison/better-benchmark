class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_3: "f32[512, 3, 224, 224]", primals_4: "f32[16, 64, 1, 1]", primals_6: "f32[64, 16, 1, 1]", primals_8: "f32[64, 16, 3, 3]", primals_10: "f32[16, 128, 1, 1]", primals_12: "f32[64, 16, 1, 1]", primals_14: "f32[64, 16, 3, 3]", primals_16: "f32[32, 128, 1, 1]", primals_18: "f32[128, 32, 1, 1]", primals_20: "f32[128, 32, 3, 3]", primals_22: "f32[32, 256, 1, 1]", primals_24: "f32[128, 32, 1, 1]", primals_26: "f32[128, 32, 3, 3]", primals_28: "f32[48, 256, 1, 1]", primals_30: "f32[192, 48, 1, 1]", primals_32: "f32[192, 48, 3, 3]", primals_34: "f32[48, 384, 1, 1]", primals_36: "f32[192, 48, 1, 1]", primals_38: "f32[192, 48, 3, 3]", primals_40: "f32[64, 384, 1, 1]", primals_42: "f32[256, 64, 1, 1]", primals_44: "f32[256, 64, 3, 3]", primals_46: "f32[64, 512, 1, 1]", primals_48: "f32[256, 64, 1, 1]", primals_50: "f32[256, 64, 3, 3]", primals_52: "f32[1000, 512, 1, 1]", getitem: "f32[512, 64, 55, 55]", getitem_1: "i8[512, 64, 55, 55]", relu_1: "f32[512, 16, 55, 55]", cat: "f32[512, 128, 55, 55]", relu_4: "f32[512, 16, 55, 55]", getitem_2: "f32[512, 128, 27, 27]", getitem_3: "i8[512, 128, 27, 27]", relu_7: "f32[512, 32, 27, 27]", cat_2: "f32[512, 256, 27, 27]", relu_10: "f32[512, 32, 27, 27]", getitem_4: "f32[512, 256, 13, 13]", getitem_5: "i8[512, 256, 13, 13]", relu_13: "f32[512, 48, 13, 13]", cat_4: "f32[512, 384, 13, 13]", relu_16: "f32[512, 48, 13, 13]", cat_5: "f32[512, 384, 13, 13]", relu_19: "f32[512, 64, 13, 13]", cat_6: "f32[512, 512, 13, 13]", relu_22: "f32[512, 64, 13, 13]", gt: "b8[512, 512, 13, 13]", mul_1: "f32[512, 512, 13, 13]", le: "b8[512, 1000, 13, 13]", le_1: "b8[512, 256, 13, 13]", le_2: "b8[512, 256, 13, 13]", le_4: "b8[512, 256, 13, 13]", le_5: "b8[512, 256, 13, 13]", le_7: "b8[512, 192, 13, 13]", le_8: "b8[512, 192, 13, 13]", le_10: "b8[512, 192, 13, 13]", le_11: "b8[512, 192, 13, 13]", le_13: "b8[512, 128, 27, 27]", le_14: "b8[512, 128, 27, 27]", le_16: "b8[512, 128, 27, 27]", le_17: "b8[512, 128, 27, 27]", le_19: "b8[512, 64, 55, 55]", le_20: "b8[512, 64, 55, 55]", le_22: "b8[512, 64, 55, 55]", le_23: "b8[512, 64, 55, 55]", le_25: "b8[512, 64, 111, 111]", tangents_1: "f32[512, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:97 in forward, code: return torch.flatten(x, 1)
        view_1: "f32[512, 1000, 1, 1]" = torch.ops.aten.reshape.default(tangents_1, [512, 1000, 1, 1]);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        expand: "f32[512, 1000, 13, 13]" = torch.ops.aten.expand.default(view_1, [512, 1000, 13, 13]);  view_1 = None
        div: "f32[512, 1000, 13, 13]" = torch.ops.aten.div.Scalar(expand, 169);  expand = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[512, 1000, 13, 13]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
        sum_1: "f32[1000]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(where, mul_1, primals_52, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = mul_1 = primals_52 = None
        getitem_6: "f32[512, 512, 13, 13]" = convolution_backward[0]
        getitem_7: "f32[1000, 512, 1, 1]" = convolution_backward[1];  convolution_backward = None
        convert_element_type: "f32[512, 512, 13, 13]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_2: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
        mul_3: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(getitem_6, mul_2);  getitem_6 = mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_1: "f32[512, 256, 13, 13]" = torch.ops.aten.slice.Tensor(mul_3, 1, 0, 256)
        slice_2: "f32[512, 256, 13, 13]" = torch.ops.aten.slice.Tensor(mul_3, 1, 256, 512);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_1: "f32[512, 256, 13, 13]" = torch.ops.aten.where.self(le_1, full_default, slice_2);  le_1 = slice_2 = None
        sum_2: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_1, relu_22, primals_50, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = primals_50 = None
        getitem_9: "f32[512, 64, 13, 13]" = convolution_backward_1[0]
        getitem_10: "f32[256, 64, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
        where_2: "f32[512, 256, 13, 13]" = torch.ops.aten.where.self(le_2, full_default, slice_1);  le_2 = slice_1 = None
        sum_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_2, relu_22, primals_48, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = primals_48 = None
        getitem_12: "f32[512, 64, 13, 13]" = convolution_backward_2[0]
        getitem_13: "f32[256, 64, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
        add: "f32[512, 64, 13, 13]" = torch.ops.aten.add.Tensor(getitem_9, getitem_12);  getitem_9 = getitem_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_3: "b8[512, 64, 13, 13]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_3: "f32[512, 64, 13, 13]" = torch.ops.aten.where.self(le_3, full_default, add);  le_3 = add = None
        sum_4: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_3, cat_6, primals_46, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = cat_6 = primals_46 = None
        getitem_15: "f32[512, 512, 13, 13]" = convolution_backward_3[0]
        getitem_16: "f32[64, 512, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_3: "f32[512, 256, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_15, 1, 0, 256)
        slice_4: "f32[512, 256, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_15, 1, 256, 512);  getitem_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_4: "f32[512, 256, 13, 13]" = torch.ops.aten.where.self(le_4, full_default, slice_4);  le_4 = slice_4 = None
        sum_5: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_4, relu_19, primals_44, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = primals_44 = None
        getitem_18: "f32[512, 64, 13, 13]" = convolution_backward_4[0]
        getitem_19: "f32[256, 64, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
        where_5: "f32[512, 256, 13, 13]" = torch.ops.aten.where.self(le_5, full_default, slice_3);  le_5 = slice_3 = None
        sum_6: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_5, relu_19, primals_42, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = primals_42 = None
        getitem_21: "f32[512, 64, 13, 13]" = convolution_backward_5[0]
        getitem_22: "f32[256, 64, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
        add_1: "f32[512, 64, 13, 13]" = torch.ops.aten.add.Tensor(getitem_18, getitem_21);  getitem_18 = getitem_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_6: "b8[512, 64, 13, 13]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_6: "f32[512, 64, 13, 13]" = torch.ops.aten.where.self(le_6, full_default, add_1);  le_6 = add_1 = None
        sum_7: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_6, cat_5, primals_40, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = cat_5 = primals_40 = None
        getitem_24: "f32[512, 384, 13, 13]" = convolution_backward_6[0]
        getitem_25: "f32[64, 384, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_5: "f32[512, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 0, 192)
        slice_6: "f32[512, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 192, 384);  getitem_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_7: "f32[512, 192, 13, 13]" = torch.ops.aten.where.self(le_7, full_default, slice_6);  le_7 = slice_6 = None
        sum_8: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_7, relu_16, primals_38, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_7 = primals_38 = None
        getitem_27: "f32[512, 48, 13, 13]" = convolution_backward_7[0]
        getitem_28: "f32[192, 48, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
        where_8: "f32[512, 192, 13, 13]" = torch.ops.aten.where.self(le_8, full_default, slice_5);  le_8 = slice_5 = None
        sum_9: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_8, relu_16, primals_36, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = primals_36 = None
        getitem_30: "f32[512, 48, 13, 13]" = convolution_backward_8[0]
        getitem_31: "f32[192, 48, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
        add_2: "f32[512, 48, 13, 13]" = torch.ops.aten.add.Tensor(getitem_27, getitem_30);  getitem_27 = getitem_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_9: "b8[512, 48, 13, 13]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_9: "f32[512, 48, 13, 13]" = torch.ops.aten.where.self(le_9, full_default, add_2);  le_9 = add_2 = None
        sum_10: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_9, cat_4, primals_34, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = cat_4 = primals_34 = None
        getitem_33: "f32[512, 384, 13, 13]" = convolution_backward_9[0]
        getitem_34: "f32[48, 384, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_7: "f32[512, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_33, 1, 0, 192)
        slice_8: "f32[512, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_33, 1, 192, 384);  getitem_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_10: "f32[512, 192, 13, 13]" = torch.ops.aten.where.self(le_10, full_default, slice_8);  le_10 = slice_8 = None
        sum_11: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(where_10, relu_13, primals_32, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = primals_32 = None
        getitem_36: "f32[512, 48, 13, 13]" = convolution_backward_10[0]
        getitem_37: "f32[192, 48, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
        where_11: "f32[512, 192, 13, 13]" = torch.ops.aten.where.self(le_11, full_default, slice_7);  le_11 = slice_7 = None
        sum_12: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_11, relu_13, primals_30, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = primals_30 = None
        getitem_39: "f32[512, 48, 13, 13]" = convolution_backward_11[0]
        getitem_40: "f32[192, 48, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
        add_3: "f32[512, 48, 13, 13]" = torch.ops.aten.add.Tensor(getitem_36, getitem_39);  getitem_36 = getitem_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_12: "b8[512, 48, 13, 13]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_12: "f32[512, 48, 13, 13]" = torch.ops.aten.where.self(le_12, full_default, add_3);  le_12 = add_3 = None
        sum_13: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_12, getitem_4, primals_28, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_12 = getitem_4 = primals_28 = None
        getitem_42: "f32[512, 256, 13, 13]" = convolution_backward_12[0]
        getitem_43: "f32[48, 256, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        full_default_13: "f32[131072, 729]" = torch.ops.aten.full.default([131072, 729], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_2: "f32[131072, 169]" = torch.ops.aten.reshape.default(getitem_42, [131072, 169]);  getitem_42 = None
        _low_memory_max_pool_offsets_to_indices_2: "i64[512, 256, 13, 13]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_5, [3, 3], [27, 27], [2, 2], [0, 0], [1, 1]);  getitem_5 = None
        view_3: "i64[131072, 169]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_2, [131072, 169]);  _low_memory_max_pool_offsets_to_indices_2 = None
        scatter_add: "f32[131072, 729]" = torch.ops.aten.scatter_add.default(full_default_13, 1, view_3, view_2);  full_default_13 = view_3 = view_2 = None
        view_4: "f32[512, 256, 27, 27]" = torch.ops.aten.reshape.default(scatter_add, [512, 256, 27, 27]);  scatter_add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_9: "f32[512, 128, 27, 27]" = torch.ops.aten.slice.Tensor(view_4, 1, 0, 128)
        slice_10: "f32[512, 128, 27, 27]" = torch.ops.aten.slice.Tensor(view_4, 1, 128, 256);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_13: "f32[512, 128, 27, 27]" = torch.ops.aten.where.self(le_13, full_default, slice_10);  le_13 = slice_10 = None
        sum_14: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_13, relu_10, primals_26, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_13 = primals_26 = None
        getitem_45: "f32[512, 32, 27, 27]" = convolution_backward_13[0]
        getitem_46: "f32[128, 32, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
        where_14: "f32[512, 128, 27, 27]" = torch.ops.aten.where.self(le_14, full_default, slice_9);  le_14 = slice_9 = None
        sum_15: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_14, relu_10, primals_24, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_14 = primals_24 = None
        getitem_48: "f32[512, 32, 27, 27]" = convolution_backward_14[0]
        getitem_49: "f32[128, 32, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
        add_4: "f32[512, 32, 27, 27]" = torch.ops.aten.add.Tensor(getitem_45, getitem_48);  getitem_45 = getitem_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_15: "b8[512, 32, 27, 27]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_15: "f32[512, 32, 27, 27]" = torch.ops.aten.where.self(le_15, full_default, add_4);  le_15 = add_4 = None
        sum_16: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(where_15, cat_2, primals_22, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_15 = cat_2 = primals_22 = None
        getitem_51: "f32[512, 256, 27, 27]" = convolution_backward_15[0]
        getitem_52: "f32[32, 256, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_11: "f32[512, 128, 27, 27]" = torch.ops.aten.slice.Tensor(getitem_51, 1, 0, 128)
        slice_12: "f32[512, 128, 27, 27]" = torch.ops.aten.slice.Tensor(getitem_51, 1, 128, 256);  getitem_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_16: "f32[512, 128, 27, 27]" = torch.ops.aten.where.self(le_16, full_default, slice_12);  le_16 = slice_12 = None
        sum_17: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(where_16, relu_7, primals_20, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_16 = primals_20 = None
        getitem_54: "f32[512, 32, 27, 27]" = convolution_backward_16[0]
        getitem_55: "f32[128, 32, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
        where_17: "f32[512, 128, 27, 27]" = torch.ops.aten.where.self(le_17, full_default, slice_11);  le_17 = slice_11 = None
        sum_18: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_17, relu_7, primals_18, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_17 = primals_18 = None
        getitem_57: "f32[512, 32, 27, 27]" = convolution_backward_17[0]
        getitem_58: "f32[128, 32, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
        add_5: "f32[512, 32, 27, 27]" = torch.ops.aten.add.Tensor(getitem_54, getitem_57);  getitem_54 = getitem_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_18: "b8[512, 32, 27, 27]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_18: "f32[512, 32, 27, 27]" = torch.ops.aten.where.self(le_18, full_default, add_5);  le_18 = add_5 = None
        sum_19: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_18, getitem_2, primals_16, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_18 = getitem_2 = primals_16 = None
        getitem_60: "f32[512, 128, 27, 27]" = convolution_backward_18[0]
        getitem_61: "f32[32, 128, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        full_default_20: "f32[65536, 3025]" = torch.ops.aten.full.default([65536, 3025], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_5: "f32[65536, 729]" = torch.ops.aten.reshape.default(getitem_60, [65536, 729]);  getitem_60 = None
        _low_memory_max_pool_offsets_to_indices_1: "i64[512, 128, 27, 27]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [55, 55], [2, 2], [0, 0], [1, 1]);  getitem_3 = None
        view_6: "i64[65536, 729]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_1, [65536, 729]);  _low_memory_max_pool_offsets_to_indices_1 = None
        scatter_add_1: "f32[65536, 3025]" = torch.ops.aten.scatter_add.default(full_default_20, 1, view_6, view_5);  full_default_20 = view_6 = view_5 = None
        view_7: "f32[512, 128, 55, 55]" = torch.ops.aten.reshape.default(scatter_add_1, [512, 128, 55, 55]);  scatter_add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_13: "f32[512, 64, 55, 55]" = torch.ops.aten.slice.Tensor(view_7, 1, 0, 64)
        slice_14: "f32[512, 64, 55, 55]" = torch.ops.aten.slice.Tensor(view_7, 1, 64, 128);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_19: "f32[512, 64, 55, 55]" = torch.ops.aten.where.self(le_19, full_default, slice_14);  le_19 = slice_14 = None
        sum_20: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(where_19, relu_4, primals_14, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_19 = primals_14 = None
        getitem_63: "f32[512, 16, 55, 55]" = convolution_backward_19[0]
        getitem_64: "f32[64, 16, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
        where_20: "f32[512, 64, 55, 55]" = torch.ops.aten.where.self(le_20, full_default, slice_13);  le_20 = slice_13 = None
        sum_21: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(where_20, relu_4, primals_12, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_20 = primals_12 = None
        getitem_66: "f32[512, 16, 55, 55]" = convolution_backward_20[0]
        getitem_67: "f32[64, 16, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
        add_6: "f32[512, 16, 55, 55]" = torch.ops.aten.add.Tensor(getitem_63, getitem_66);  getitem_63 = getitem_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_21: "b8[512, 16, 55, 55]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_21: "f32[512, 16, 55, 55]" = torch.ops.aten.where.self(le_21, full_default, add_6);  le_21 = add_6 = None
        sum_22: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(where_21, cat, primals_10, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_21 = cat = primals_10 = None
        getitem_69: "f32[512, 128, 55, 55]" = convolution_backward_21[0]
        getitem_70: "f32[16, 128, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_15: "f32[512, 64, 55, 55]" = torch.ops.aten.slice.Tensor(getitem_69, 1, 0, 64)
        slice_16: "f32[512, 64, 55, 55]" = torch.ops.aten.slice.Tensor(getitem_69, 1, 64, 128);  getitem_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_22: "f32[512, 64, 55, 55]" = torch.ops.aten.where.self(le_22, full_default, slice_16);  le_22 = slice_16 = None
        sum_23: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(where_22, relu_1, primals_8, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_22 = primals_8 = None
        getitem_72: "f32[512, 16, 55, 55]" = convolution_backward_22[0]
        getitem_73: "f32[64, 16, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
        where_23: "f32[512, 64, 55, 55]" = torch.ops.aten.where.self(le_23, full_default, slice_15);  le_23 = slice_15 = None
        sum_24: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_23, relu_1, primals_6, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_23 = primals_6 = None
        getitem_75: "f32[512, 16, 55, 55]" = convolution_backward_23[0]
        getitem_76: "f32[64, 16, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
        add_7: "f32[512, 16, 55, 55]" = torch.ops.aten.add.Tensor(getitem_72, getitem_75);  getitem_72 = getitem_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_24: "b8[512, 16, 55, 55]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_24: "f32[512, 16, 55, 55]" = torch.ops.aten.where.self(le_24, full_default, add_7);  le_24 = add_7 = None
        sum_25: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(where_24, getitem, primals_4, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_24 = getitem = primals_4 = None
        getitem_78: "f32[512, 64, 55, 55]" = convolution_backward_24[0]
        getitem_79: "f32[16, 64, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        full_default_27: "f32[32768, 12321]" = torch.ops.aten.full.default([32768, 12321], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_8: "f32[32768, 3025]" = torch.ops.aten.reshape.default(getitem_78, [32768, 3025]);  getitem_78 = None
        _low_memory_max_pool_offsets_to_indices: "i64[512, 64, 55, 55]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_1, [3, 3], [111, 111], [2, 2], [0, 0], [1, 1]);  getitem_1 = None
        view_9: "i64[32768, 3025]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [32768, 3025]);  _low_memory_max_pool_offsets_to_indices = None
        scatter_add_2: "f32[32768, 12321]" = torch.ops.aten.scatter_add.default(full_default_27, 1, view_9, view_8);  full_default_27 = view_9 = view_8 = None
        view_10: "f32[512, 64, 111, 111]" = torch.ops.aten.reshape.default(scatter_add_2, [512, 64, 111, 111]);  scatter_add_2 = None
        where_25: "f32[512, 64, 111, 111]" = torch.ops.aten.where.self(le_25, full_default, view_10);  le_25 = full_default = view_10 = None
        sum_26: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(where_25, primals_3, primals_1, [64], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  where_25 = primals_3 = primals_1 = None
        getitem_82: "f32[64, 3, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
        return (getitem_82, sum_26, None, getitem_79, sum_25, getitem_76, sum_24, getitem_73, sum_23, getitem_70, sum_22, getitem_67, sum_21, getitem_64, sum_20, getitem_61, sum_19, getitem_58, sum_18, getitem_55, sum_17, getitem_52, sum_16, getitem_49, sum_15, getitem_46, sum_14, getitem_43, sum_13, getitem_40, sum_12, getitem_37, sum_11, getitem_34, sum_10, getitem_31, sum_9, getitem_28, sum_8, getitem_25, sum_7, getitem_22, sum_6, getitem_19, sum_5, getitem_16, sum_4, getitem_13, sum_3, getitem_10, sum_2, getitem_7, sum_1)
