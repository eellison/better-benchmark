class GraphModule(torch.nn.Module):
    def forward(self, convert_element_type_1: "bf16[64, 3, 3, 3][27, 1, 9, 3]cuda:0", convert_element_type_2: "bf16[32, 3, 224, 224][150528, 1, 672, 3]cuda:0", getitem: "bf16[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0", getitem_1: "i8[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0", convert_element_type_4: "bf16[16, 64, 1, 1][64, 1, 64, 64]cuda:0", relu_1: "bf16[32, 16, 55, 55][48400, 1, 880, 16]cuda:0", convert_element_type_6: "bf16[64, 16, 1, 1][16, 1, 16, 16]cuda:0", convert_element_type_8: "bf16[64, 16, 3, 3][144, 1, 48, 16]cuda:0", cat: "bf16[32, 128, 55, 55][387200, 1, 7040, 128]cuda:0", convert_element_type_10: "bf16[16, 128, 1, 1][128, 1, 128, 128]cuda:0", relu_4: "bf16[32, 16, 55, 55][48400, 1, 880, 16]cuda:0", convert_element_type_12: "bf16[64, 16, 1, 1][16, 1, 16, 16]cuda:0", convert_element_type_14: "bf16[64, 16, 3, 3][144, 1, 48, 16]cuda:0", getitem_2: "bf16[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0", getitem_3: "i8[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0", convert_element_type_16: "bf16[32, 128, 1, 1][128, 1, 128, 128]cuda:0", relu_7: "bf16[32, 32, 27, 27][23328, 1, 864, 32]cuda:0", convert_element_type_18: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0", convert_element_type_20: "bf16[128, 32, 3, 3][288, 1, 96, 32]cuda:0", cat_2: "bf16[32, 256, 27, 27][186624, 1, 6912, 256]cuda:0", convert_element_type_22: "bf16[32, 256, 1, 1][256, 1, 256, 256]cuda:0", relu_10: "bf16[32, 32, 27, 27][23328, 1, 864, 32]cuda:0", convert_element_type_24: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0", convert_element_type_26: "bf16[128, 32, 3, 3][288, 1, 96, 32]cuda:0", getitem_4: "bf16[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0", getitem_5: "i8[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0", convert_element_type_28: "bf16[48, 256, 1, 1][256, 1, 256, 256]cuda:0", relu_13: "bf16[32, 48, 13, 13][8112, 1, 624, 48]cuda:0", convert_element_type_30: "bf16[192, 48, 1, 1][48, 1, 48, 48]cuda:0", convert_element_type_32: "bf16[192, 48, 3, 3][432, 1, 144, 48]cuda:0", cat_4: "bf16[32, 384, 13, 13][64896, 1, 4992, 384]cuda:0", convert_element_type_34: "bf16[48, 384, 1, 1][384, 1, 384, 384]cuda:0", relu_16: "bf16[32, 48, 13, 13][8112, 1, 624, 48]cuda:0", convert_element_type_36: "bf16[192, 48, 1, 1][48, 1, 48, 48]cuda:0", convert_element_type_38: "bf16[192, 48, 3, 3][432, 1, 144, 48]cuda:0", cat_5: "bf16[32, 384, 13, 13][64896, 1, 4992, 384]cuda:0", convert_element_type_40: "bf16[64, 384, 1, 1][384, 1, 384, 384]cuda:0", relu_19: "bf16[32, 64, 13, 13][10816, 1, 832, 64]cuda:0", convert_element_type_42: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", convert_element_type_44: "bf16[256, 64, 3, 3][576, 1, 192, 64]cuda:0", cat_6: "bf16[32, 512, 13, 13][86528, 1, 6656, 512]cuda:0", convert_element_type_46: "bf16[64, 512, 1, 1][512, 1, 512, 512]cuda:0", relu_22: "bf16[32, 64, 13, 13][10816, 1, 832, 64]cuda:0", convert_element_type_48: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", convert_element_type_50: "bf16[256, 64, 3, 3][576, 1, 192, 64]cuda:0", gt: "b8[32, 512, 13, 13][86528, 1, 6656, 512]cuda:0", mul_1: "bf16[32, 512, 13, 13][86528, 1, 6656, 512]cuda:0", convert_element_type_52: "bf16[1000, 512, 1, 1][512, 1, 512, 512]cuda:0", le: "b8[32, 1000, 13, 13][169088, 169, 13, 1]cuda:0", le_1: "b8[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0", le_2: "b8[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0", le_4: "b8[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0", le_5: "b8[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0", le_7: "b8[32, 192, 13, 13][32448, 1, 2496, 192]cuda:0", le_8: "b8[32, 192, 13, 13][32448, 1, 2496, 192]cuda:0", le_10: "b8[32, 192, 13, 13][32448, 1, 2496, 192]cuda:0", le_11: "b8[32, 192, 13, 13][32448, 1, 2496, 192]cuda:0", le_13: "b8[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0", le_14: "b8[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0", le_16: "b8[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0", le_17: "b8[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0", le_19: "b8[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0", le_20: "b8[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0", le_22: "b8[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0", le_23: "b8[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0", le_25: "b8[32, 64, 111, 111][788544, 1, 7104, 64]cuda:0", tangents_1: "bf16[32, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:97 in forward, code: return torch.flatten(x, 1)
        view_1: "bf16[32, 1000, 1, 1][1000, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(tangents_1, [32, 1000, 1, 1]);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        expand: "bf16[32, 1000, 13, 13][1000, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(view_1, [32, 1000, 13, 13]);  view_1 = None
        div: "bf16[32, 1000, 13, 13][169000, 169, 13, 1]cuda:0" = torch.ops.aten.div.Scalar(expand, 169);  expand = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[32, 1000, 13, 13][169000, 169, 13, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
        sum_1: "bf16[1000][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(where, mul_1, convert_element_type_52, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = mul_1 = convert_element_type_52 = None
        getitem_6: "bf16[32, 512, 13, 13][86528, 1, 6656, 512]cuda:0" = convolution_backward[0]
        getitem_7: "bf16[1000, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_53: "f32[1000, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
        convert_element_type_54: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        convert_element_type_55: "bf16[32, 512, 13, 13][86528, 1, 6656, 512]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_2: "bf16[32, 512, 13, 13][86528, 1, 6656, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, 2.0);  convert_element_type_55 = None
        mul_3: "bf16[32, 512, 13, 13][86528, 1, 6656, 512]cuda:0" = torch.ops.aten.mul.Tensor(getitem_6, mul_2);  getitem_6 = mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_1: "bf16[32, 256, 13, 13][86528, 1, 6656, 512]cuda:0" = torch.ops.aten.slice.Tensor(mul_3, 1, 0, 256)
        slice_2: "bf16[32, 256, 13, 13][86528, 1, 6656, 512]cuda:0" = torch.ops.aten.slice.Tensor(mul_3, 1, 256, 512);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_1: "bf16[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = torch.ops.aten.where.self(le_1, full_default, slice_2);  le_1 = slice_2 = None
        sum_2: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_1, relu_22, convert_element_type_50, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = convert_element_type_50 = None
        getitem_9: "bf16[32, 64, 13, 13][10816, 1, 832, 64]cuda:0" = convolution_backward_1[0]
        getitem_10: "bf16[256, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_56: "f32[256, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
        convert_element_type_57: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        where_2: "bf16[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = torch.ops.aten.where.self(le_2, full_default, slice_1);  le_2 = slice_1 = None
        sum_3: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_2, relu_22, convert_element_type_48, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = convert_element_type_48 = None
        getitem_12: "bf16[32, 64, 13, 13][10816, 1, 832, 64]cuda:0" = convolution_backward_2[0]
        getitem_13: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        add: "bf16[32, 64, 13, 13][10816, 1, 832, 64]cuda:0" = torch.ops.aten.add.Tensor(getitem_9, getitem_12);  getitem_9 = getitem_12 = None
        convert_element_type_58: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
        convert_element_type_59: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_3: "b8[32, 64, 13, 13][10816, 1, 832, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_3: "bf16[32, 64, 13, 13][10816, 1, 832, 64]cuda:0" = torch.ops.aten.where.self(le_3, full_default, add);  le_3 = add = None
        sum_4: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_3, cat_6, convert_element_type_46, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = cat_6 = convert_element_type_46 = None
        getitem_15: "bf16[32, 512, 13, 13][86528, 1, 6656, 512]cuda:0" = convolution_backward_3[0]
        getitem_16: "bf16[64, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_60: "f32[64, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_16, torch.float32);  getitem_16 = None
        convert_element_type_61: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_4, torch.float32);  sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_3: "bf16[32, 256, 13, 13][86528, 1, 6656, 512]cuda:0" = torch.ops.aten.slice.Tensor(getitem_15, 1, 0, 256)
        slice_4: "bf16[32, 256, 13, 13][86528, 1, 6656, 512]cuda:0" = torch.ops.aten.slice.Tensor(getitem_15, 1, 256, 512);  getitem_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_4: "bf16[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = torch.ops.aten.where.self(le_4, full_default, slice_4);  le_4 = slice_4 = None
        sum_5: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_4, relu_19, convert_element_type_44, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = convert_element_type_44 = None
        getitem_18: "bf16[32, 64, 13, 13][10816, 1, 832, 64]cuda:0" = convolution_backward_4[0]
        getitem_19: "bf16[256, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_62: "f32[256, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
        convert_element_type_63: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_5, torch.float32);  sum_5 = None
        where_5: "bf16[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = torch.ops.aten.where.self(le_5, full_default, slice_3);  le_5 = slice_3 = None
        sum_6: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_5, relu_19, convert_element_type_42, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = convert_element_type_42 = None
        getitem_21: "bf16[32, 64, 13, 13][10816, 1, 832, 64]cuda:0" = convolution_backward_5[0]
        getitem_22: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        add_1: "bf16[32, 64, 13, 13][10816, 1, 832, 64]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, getitem_21);  getitem_18 = getitem_21 = None
        convert_element_type_64: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_22, torch.float32);  getitem_22 = None
        convert_element_type_65: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_6, torch.float32);  sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_6: "b8[32, 64, 13, 13][10816, 1, 832, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_6: "bf16[32, 64, 13, 13][10816, 1, 832, 64]cuda:0" = torch.ops.aten.where.self(le_6, full_default, add_1);  le_6 = add_1 = None
        sum_7: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_6, cat_5, convert_element_type_40, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = cat_5 = convert_element_type_40 = None
        getitem_24: "bf16[32, 384, 13, 13][64896, 1, 4992, 384]cuda:0" = convolution_backward_6[0]
        getitem_25: "bf16[64, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_66: "f32[64, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
        convert_element_type_67: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_5: "bf16[32, 192, 13, 13][64896, 1, 4992, 384]cuda:0" = torch.ops.aten.slice.Tensor(getitem_24, 1, 0, 192)
        slice_6: "bf16[32, 192, 13, 13][64896, 1, 4992, 384]cuda:0" = torch.ops.aten.slice.Tensor(getitem_24, 1, 192, 384);  getitem_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_7: "bf16[32, 192, 13, 13][32448, 1, 2496, 192]cuda:0" = torch.ops.aten.where.self(le_7, full_default, slice_6);  le_7 = slice_6 = None
        sum_8: "bf16[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_7, relu_16, convert_element_type_38, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_7 = convert_element_type_38 = None
        getitem_27: "bf16[32, 48, 13, 13][8112, 1, 624, 48]cuda:0" = convolution_backward_7[0]
        getitem_28: "bf16[192, 48, 3, 3][432, 1, 144, 48]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_68: "f32[192, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_28, torch.float32);  getitem_28 = None
        convert_element_type_69: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
        where_8: "bf16[32, 192, 13, 13][32448, 1, 2496, 192]cuda:0" = torch.ops.aten.where.self(le_8, full_default, slice_5);  le_8 = slice_5 = None
        sum_9: "bf16[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_8, relu_16, convert_element_type_36, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = convert_element_type_36 = None
        getitem_30: "bf16[32, 48, 13, 13][8112, 1, 624, 48]cuda:0" = convolution_backward_8[0]
        getitem_31: "bf16[192, 48, 1, 1][48, 1, 48, 48]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        add_2: "bf16[32, 48, 13, 13][8112, 1, 624, 48]cuda:0" = torch.ops.aten.add.Tensor(getitem_27, getitem_30);  getitem_27 = getitem_30 = None
        convert_element_type_70: "f32[192, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
        convert_element_type_71: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_9, torch.float32);  sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_9: "b8[32, 48, 13, 13][8112, 1, 624, 48]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_9: "bf16[32, 48, 13, 13][8112, 1, 624, 48]cuda:0" = torch.ops.aten.where.self(le_9, full_default, add_2);  le_9 = add_2 = None
        sum_10: "bf16[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_9, cat_4, convert_element_type_34, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = cat_4 = convert_element_type_34 = None
        getitem_33: "bf16[32, 384, 13, 13][64896, 1, 4992, 384]cuda:0" = convolution_backward_9[0]
        getitem_34: "bf16[48, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_72: "f32[48, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_34, torch.float32);  getitem_34 = None
        convert_element_type_73: "f32[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_10, torch.float32);  sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_7: "bf16[32, 192, 13, 13][64896, 1, 4992, 384]cuda:0" = torch.ops.aten.slice.Tensor(getitem_33, 1, 0, 192)
        slice_8: "bf16[32, 192, 13, 13][64896, 1, 4992, 384]cuda:0" = torch.ops.aten.slice.Tensor(getitem_33, 1, 192, 384);  getitem_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_10: "bf16[32, 192, 13, 13][32448, 1, 2496, 192]cuda:0" = torch.ops.aten.where.self(le_10, full_default, slice_8);  le_10 = slice_8 = None
        sum_11: "bf16[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(where_10, relu_13, convert_element_type_32, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = convert_element_type_32 = None
        getitem_36: "bf16[32, 48, 13, 13][8112, 1, 624, 48]cuda:0" = convolution_backward_10[0]
        getitem_37: "bf16[192, 48, 3, 3][432, 1, 144, 48]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_74: "f32[192, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
        convert_element_type_75: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_11, torch.float32);  sum_11 = None
        where_11: "bf16[32, 192, 13, 13][32448, 1, 2496, 192]cuda:0" = torch.ops.aten.where.self(le_11, full_default, slice_7);  le_11 = slice_7 = None
        sum_12: "bf16[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_11, relu_13, convert_element_type_30, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = convert_element_type_30 = None
        getitem_39: "bf16[32, 48, 13, 13][8112, 1, 624, 48]cuda:0" = convolution_backward_11[0]
        getitem_40: "bf16[192, 48, 1, 1][48, 1, 48, 48]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        add_3: "bf16[32, 48, 13, 13][8112, 1, 624, 48]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, getitem_39);  getitem_36 = getitem_39 = None
        convert_element_type_76: "f32[192, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_40, torch.float32);  getitem_40 = None
        convert_element_type_77: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_12, torch.float32);  sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_12: "b8[32, 48, 13, 13][8112, 1, 624, 48]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_12: "bf16[32, 48, 13, 13][8112, 1, 624, 48]cuda:0" = torch.ops.aten.where.self(le_12, full_default, add_3);  le_12 = add_3 = None
        sum_13: "bf16[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_12, getitem_4, convert_element_type_28, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_12 = getitem_4 = convert_element_type_28 = None
        getitem_42: "bf16[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = convolution_backward_12[0]
        getitem_43: "bf16[48, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_78: "f32[48, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
        convert_element_type_79: "f32[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_13, torch.float32);  sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        full_default_13: "f32[8192, 729][729, 1]cuda:0" = torch.ops.aten.full.default([8192, 729], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_2: "bf16[8192, 169][169, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_42, [8192, 169]);  getitem_42 = None
        _low_memory_max_pool_offsets_to_indices_2: "i64[32, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_5, [3, 3], [27, 27], [2, 2], [0, 0], [1, 1]);  getitem_5 = None
        view_3: "i64[8192, 169][169, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_2, [8192, 169]);  _low_memory_max_pool_offsets_to_indices_2 = None
        convert_element_type_80: "f32[8192, 169][169, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        scatter_add: "f32[8192, 729][729, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_13, 1, view_3, convert_element_type_80);  full_default_13 = view_3 = convert_element_type_80 = None
        view_4: "f32[32, 256, 27, 27][186624, 729, 27, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add, [32, 256, 27, 27]);  scatter_add = None
        convert_element_type_81: "bf16[32, 256, 27, 27][186624, 729, 27, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_9: "bf16[32, 128, 27, 27][186624, 729, 27, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_81, 1, 0, 128)
        slice_10: "bf16[32, 128, 27, 27][186624, 729, 27, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_81, 1, 128, 256);  convert_element_type_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_13: "bf16[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0" = torch.ops.aten.where.self(le_13, full_default, slice_10);  le_13 = slice_10 = None
        sum_14: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_13, relu_10, convert_element_type_26, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_13 = convert_element_type_26 = None
        getitem_45: "bf16[32, 32, 27, 27][23328, 1, 864, 32]cuda:0" = convolution_backward_13[0]
        getitem_46: "bf16[128, 32, 3, 3][288, 1, 96, 32]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_82: "f32[128, 32, 3, 3][288, 1, 96, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_46, torch.float32);  getitem_46 = None
        convert_element_type_83: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        where_14: "bf16[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0" = torch.ops.aten.where.self(le_14, full_default, slice_9);  le_14 = slice_9 = None
        sum_15: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_14, relu_10, convert_element_type_24, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_14 = convert_element_type_24 = None
        getitem_48: "bf16[32, 32, 27, 27][23328, 1, 864, 32]cuda:0" = convolution_backward_14[0]
        getitem_49: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        add_4: "bf16[32, 32, 27, 27][23328, 1, 864, 32]cuda:0" = torch.ops.aten.add.Tensor(getitem_45, getitem_48);  getitem_45 = getitem_48 = None
        convert_element_type_84: "f32[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
        convert_element_type_85: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_15, torch.float32);  sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_15: "b8[32, 32, 27, 27][23328, 1, 864, 32]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_15: "bf16[32, 32, 27, 27][23328, 1, 864, 32]cuda:0" = torch.ops.aten.where.self(le_15, full_default, add_4);  le_15 = add_4 = None
        sum_16: "bf16[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(where_15, cat_2, convert_element_type_22, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_15 = cat_2 = convert_element_type_22 = None
        getitem_51: "bf16[32, 256, 27, 27][186624, 1, 6912, 256]cuda:0" = convolution_backward_15[0]
        getitem_52: "bf16[32, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_86: "f32[32, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_52, torch.float32);  getitem_52 = None
        convert_element_type_87: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_16, torch.float32);  sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_11: "bf16[32, 128, 27, 27][186624, 1, 6912, 256]cuda:0" = torch.ops.aten.slice.Tensor(getitem_51, 1, 0, 128)
        slice_12: "bf16[32, 128, 27, 27][186624, 1, 6912, 256]cuda:0" = torch.ops.aten.slice.Tensor(getitem_51, 1, 128, 256);  getitem_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_16: "bf16[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0" = torch.ops.aten.where.self(le_16, full_default, slice_12);  le_16 = slice_12 = None
        sum_17: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(where_16, relu_7, convert_element_type_20, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_16 = convert_element_type_20 = None
        getitem_54: "bf16[32, 32, 27, 27][23328, 1, 864, 32]cuda:0" = convolution_backward_16[0]
        getitem_55: "bf16[128, 32, 3, 3][288, 1, 96, 32]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_88: "f32[128, 32, 3, 3][288, 1, 96, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
        convert_element_type_89: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
        where_17: "bf16[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0" = torch.ops.aten.where.self(le_17, full_default, slice_11);  le_17 = slice_11 = None
        sum_18: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_17, relu_7, convert_element_type_18, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_17 = convert_element_type_18 = None
        getitem_57: "bf16[32, 32, 27, 27][23328, 1, 864, 32]cuda:0" = convolution_backward_17[0]
        getitem_58: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        add_5: "bf16[32, 32, 27, 27][23328, 1, 864, 32]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, getitem_57);  getitem_54 = getitem_57 = None
        convert_element_type_90: "f32[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_58, torch.float32);  getitem_58 = None
        convert_element_type_91: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_18, torch.float32);  sum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_18: "b8[32, 32, 27, 27][23328, 1, 864, 32]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_18: "bf16[32, 32, 27, 27][23328, 1, 864, 32]cuda:0" = torch.ops.aten.where.self(le_18, full_default, add_5);  le_18 = add_5 = None
        sum_19: "bf16[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_18, getitem_2, convert_element_type_16, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_18 = getitem_2 = convert_element_type_16 = None
        getitem_60: "bf16[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0" = convolution_backward_18[0]
        getitem_61: "bf16[32, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_92: "f32[32, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
        convert_element_type_93: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_19, torch.float32);  sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        full_default_20: "f32[4096, 3025][3025, 1]cuda:0" = torch.ops.aten.full.default([4096, 3025], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_5: "bf16[4096, 729][729, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_60, [4096, 729]);  getitem_60 = None
        _low_memory_max_pool_offsets_to_indices_1: "i64[32, 128, 27, 27][93312, 1, 3456, 128]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [55, 55], [2, 2], [0, 0], [1, 1]);  getitem_3 = None
        view_6: "i64[4096, 729][729, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_1, [4096, 729]);  _low_memory_max_pool_offsets_to_indices_1 = None
        convert_element_type_94: "f32[4096, 729][729, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        scatter_add_1: "f32[4096, 3025][3025, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_20, 1, view_6, convert_element_type_94);  full_default_20 = view_6 = convert_element_type_94 = None
        view_7: "f32[32, 128, 55, 55][387200, 3025, 55, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add_1, [32, 128, 55, 55]);  scatter_add_1 = None
        convert_element_type_95: "bf16[32, 128, 55, 55][387200, 3025, 55, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_13: "bf16[32, 64, 55, 55][387200, 3025, 55, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_95, 1, 0, 64)
        slice_14: "bf16[32, 64, 55, 55][387200, 3025, 55, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_95, 1, 64, 128);  convert_element_type_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_19: "bf16[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0" = torch.ops.aten.where.self(le_19, full_default, slice_14);  le_19 = slice_14 = None
        sum_20: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(where_19, relu_4, convert_element_type_14, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_19 = convert_element_type_14 = None
        getitem_63: "bf16[32, 16, 55, 55][48400, 1, 880, 16]cuda:0" = convolution_backward_19[0]
        getitem_64: "bf16[64, 16, 3, 3][144, 1, 48, 16]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_96: "f32[64, 16, 3, 3][144, 1, 48, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_64, torch.float32);  getitem_64 = None
        convert_element_type_97: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None
        where_20: "bf16[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0" = torch.ops.aten.where.self(le_20, full_default, slice_13);  le_20 = slice_13 = None
        sum_21: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(where_20, relu_4, convert_element_type_12, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_20 = convert_element_type_12 = None
        getitem_66: "bf16[32, 16, 55, 55][48400, 1, 880, 16]cuda:0" = convolution_backward_20[0]
        getitem_67: "bf16[64, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        add_6: "bf16[32, 16, 55, 55][48400, 1, 880, 16]cuda:0" = torch.ops.aten.add.Tensor(getitem_63, getitem_66);  getitem_63 = getitem_66 = None
        convert_element_type_98: "f32[64, 16, 1, 1][16, 1, 16, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
        convert_element_type_99: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_21, torch.float32);  sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_21: "b8[32, 16, 55, 55][48400, 1, 880, 16]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_21: "bf16[32, 16, 55, 55][48400, 1, 880, 16]cuda:0" = torch.ops.aten.where.self(le_21, full_default, add_6);  le_21 = add_6 = None
        sum_22: "bf16[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(where_21, cat, convert_element_type_10, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_21 = cat = convert_element_type_10 = None
        getitem_69: "bf16[32, 128, 55, 55][387200, 1, 7040, 128]cuda:0" = convolution_backward_21[0]
        getitem_70: "bf16[16, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_100: "f32[16, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_70, torch.float32);  getitem_70 = None
        convert_element_type_101: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_22, torch.float32);  sum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_15: "bf16[32, 64, 55, 55][387200, 1, 7040, 128]cuda:0" = torch.ops.aten.slice.Tensor(getitem_69, 1, 0, 64)
        slice_16: "bf16[32, 64, 55, 55][387200, 1, 7040, 128]cuda:0" = torch.ops.aten.slice.Tensor(getitem_69, 1, 64, 128);  getitem_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_22: "bf16[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0" = torch.ops.aten.where.self(le_22, full_default, slice_16);  le_22 = slice_16 = None
        sum_23: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(where_22, relu_1, convert_element_type_8, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_22 = convert_element_type_8 = None
        getitem_72: "bf16[32, 16, 55, 55][48400, 1, 880, 16]cuda:0" = convolution_backward_22[0]
        getitem_73: "bf16[64, 16, 3, 3][144, 1, 48, 16]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_102: "f32[64, 16, 3, 3][144, 1, 48, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
        convert_element_type_103: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_23, torch.float32);  sum_23 = None
        where_23: "bf16[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0" = torch.ops.aten.where.self(le_23, full_default, slice_15);  le_23 = slice_15 = None
        sum_24: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_23, relu_1, convert_element_type_6, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_23 = convert_element_type_6 = None
        getitem_75: "bf16[32, 16, 55, 55][48400, 1, 880, 16]cuda:0" = convolution_backward_23[0]
        getitem_76: "bf16[64, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        add_7: "bf16[32, 16, 55, 55][48400, 1, 880, 16]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, getitem_75);  getitem_72 = getitem_75 = None
        convert_element_type_104: "f32[64, 16, 1, 1][16, 1, 16, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_76, torch.float32);  getitem_76 = None
        convert_element_type_105: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_24, torch.float32);  sum_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_24: "b8[32, 16, 55, 55][48400, 1, 880, 16]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_24: "bf16[32, 16, 55, 55][48400, 1, 880, 16]cuda:0" = torch.ops.aten.where.self(le_24, full_default, add_7);  le_24 = add_7 = None
        sum_25: "bf16[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(where_24, getitem, convert_element_type_4, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_24 = getitem = convert_element_type_4 = None
        getitem_78: "bf16[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0" = convolution_backward_24[0]
        getitem_79: "bf16[16, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        convert_element_type_106: "f32[16, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None
        convert_element_type_107: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_25, torch.float32);  sum_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        full_default_27: "f32[2048, 12321][12321, 1]cuda:0" = torch.ops.aten.full.default([2048, 12321], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_8: "bf16[2048, 3025][3025, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_78, [2048, 3025]);  getitem_78 = None
        _low_memory_max_pool_offsets_to_indices: "i64[32, 64, 55, 55][193600, 1, 3520, 64]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_1, [3, 3], [111, 111], [2, 2], [0, 0], [1, 1]);  getitem_1 = None
        view_9: "i64[2048, 3025][3025, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [2048, 3025]);  _low_memory_max_pool_offsets_to_indices = None
        convert_element_type_108: "f32[2048, 3025][3025, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        scatter_add_2: "f32[2048, 12321][12321, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_27, 1, view_9, convert_element_type_108);  full_default_27 = view_9 = convert_element_type_108 = None
        view_10: "f32[32, 64, 111, 111][788544, 12321, 111, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add_2, [32, 64, 111, 111]);  scatter_add_2 = None
        convert_element_type_109: "bf16[32, 64, 111, 111][788544, 12321, 111, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        where_25: "bf16[32, 64, 111, 111][788544, 1, 7104, 64]cuda:0" = torch.ops.aten.where.self(le_25, full_default, convert_element_type_109);  le_25 = full_default = convert_element_type_109 = None
        sum_26: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(where_25, convert_element_type_2, convert_element_type_1, [64], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  where_25 = convert_element_type_2 = convert_element_type_1 = None
        getitem_82: "bf16[64, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_110: "f32[64, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_82, torch.float32);  getitem_82 = None
        convert_element_type_111: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        return (convert_element_type_110, convert_element_type_111, None, convert_element_type_106, convert_element_type_107, convert_element_type_104, convert_element_type_105, convert_element_type_102, convert_element_type_103, convert_element_type_100, convert_element_type_101, convert_element_type_98, convert_element_type_99, convert_element_type_96, convert_element_type_97, convert_element_type_92, convert_element_type_93, convert_element_type_90, convert_element_type_91, convert_element_type_88, convert_element_type_89, convert_element_type_86, convert_element_type_87, convert_element_type_84, convert_element_type_85, convert_element_type_82, convert_element_type_83, convert_element_type_78, convert_element_type_79, convert_element_type_76, convert_element_type_77, convert_element_type_74, convert_element_type_75, convert_element_type_72, convert_element_type_73, convert_element_type_70, convert_element_type_71, convert_element_type_68, convert_element_type_69, convert_element_type_66, convert_element_type_67, convert_element_type_64, convert_element_type_65, convert_element_type_62, convert_element_type_63, convert_element_type_60, convert_element_type_61, convert_element_type_58, convert_element_type_59, convert_element_type_56, convert_element_type_57, convert_element_type_53, convert_element_type_54)
