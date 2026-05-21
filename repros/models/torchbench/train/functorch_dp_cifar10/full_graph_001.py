class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64, 3, 32, 32]", primals_3: "f32[64]", primals_4: "f32[64]", primals_5: "f32[64, 64, 3, 3]", primals_6: "f32[64]", primals_8: "f32[64, 64, 3, 3]", primals_9: "f32[64]", primals_11: "f32[64, 64, 3, 3]", primals_12: "f32[64]", primals_14: "f32[64, 64, 3, 3]", primals_15: "f32[64]", primals_17: "f32[128, 64, 3, 3]", primals_18: "f32[128]", primals_20: "f32[128, 128, 3, 3]", primals_21: "f32[128]", primals_23: "f32[128, 64, 1, 1]", primals_24: "f32[128]", primals_26: "f32[128, 128, 3, 3]", primals_27: "f32[128]", primals_29: "f32[128, 128, 3, 3]", primals_30: "f32[128]", primals_32: "f32[256, 128, 3, 3]", primals_33: "f32[256]", primals_35: "f32[256, 256, 3, 3]", primals_36: "f32[256]", primals_38: "f32[256, 128, 1, 1]", primals_39: "f32[256]", primals_41: "f32[256, 256, 3, 3]", primals_42: "f32[256]", primals_44: "f32[256, 256, 3, 3]", primals_45: "f32[256]", primals_47: "f32[512, 256, 3, 3]", primals_48: "f32[512]", primals_50: "f32[512, 512, 3, 3]", primals_51: "f32[512]", primals_53: "f32[512, 256, 1, 1]", primals_54: "f32[512]", primals_56: "f32[512, 512, 3, 3]", primals_57: "f32[512]", primals_59: "f32[512, 512, 3, 3]", primals_60: "f32[512]", primals_62: "f32[1000, 512]", convolution: "f32[64, 64, 16, 16]", getitem_1: "f32[64, 32, 1, 1]", rsqrt: "f32[64, 32, 1, 1]", getitem_2: "f32[64, 64, 8, 8]", getitem_3: "i8[64, 64, 8, 8]", convolution_1: "f32[64, 64, 8, 8]", squeeze_2: "f32[64, 32]", squeeze_3: "f32[64, 32]", relu_1: "f32[64, 64, 8, 8]", convolution_2: "f32[64, 64, 8, 8]", squeeze_4: "f32[64, 32]", squeeze_5: "f32[64, 32]", relu_2: "f32[64, 64, 8, 8]", convolution_3: "f32[64, 64, 8, 8]", squeeze_6: "f32[64, 32]", squeeze_7: "f32[64, 32]", relu_3: "f32[64, 64, 8, 8]", convolution_4: "f32[64, 64, 8, 8]", squeeze_8: "f32[64, 32]", squeeze_9: "f32[64, 32]", relu_4: "f32[64, 64, 8, 8]", convolution_5: "f32[64, 128, 4, 4]", squeeze_10: "f32[64, 32]", squeeze_11: "f32[64, 32]", relu_5: "f32[64, 128, 4, 4]", convolution_6: "f32[64, 128, 4, 4]", squeeze_12: "f32[64, 32]", squeeze_13: "f32[64, 32]", convolution_7: "f32[64, 128, 4, 4]", squeeze_14: "f32[64, 32]", squeeze_15: "f32[64, 32]", relu_6: "f32[64, 128, 4, 4]", convolution_8: "f32[64, 128, 4, 4]", squeeze_16: "f32[64, 32]", squeeze_17: "f32[64, 32]", relu_7: "f32[64, 128, 4, 4]", convolution_9: "f32[64, 128, 4, 4]", squeeze_18: "f32[64, 32]", squeeze_19: "f32[64, 32]", relu_8: "f32[64, 128, 4, 4]", convolution_10: "f32[64, 256, 2, 2]", squeeze_20: "f32[64, 32]", squeeze_21: "f32[64, 32]", relu_9: "f32[64, 256, 2, 2]", convolution_11: "f32[64, 256, 2, 2]", squeeze_22: "f32[64, 32]", squeeze_23: "f32[64, 32]", convolution_12: "f32[64, 256, 2, 2]", squeeze_24: "f32[64, 32]", squeeze_25: "f32[64, 32]", relu_10: "f32[64, 256, 2, 2]", convolution_13: "f32[64, 256, 2, 2]", squeeze_26: "f32[64, 32]", squeeze_27: "f32[64, 32]", relu_11: "f32[64, 256, 2, 2]", convolution_14: "f32[64, 256, 2, 2]", squeeze_28: "f32[64, 32]", squeeze_29: "f32[64, 32]", relu_12: "f32[64, 256, 2, 2]", convolution_15: "f32[64, 512, 1, 1]", squeeze_30: "f32[64, 32]", squeeze_31: "f32[64, 32]", relu_13: "f32[64, 512, 1, 1]", convolution_16: "f32[64, 512, 1, 1]", squeeze_32: "f32[64, 32]", squeeze_33: "f32[64, 32]", convolution_17: "f32[64, 512, 1, 1]", squeeze_34: "f32[64, 32]", squeeze_35: "f32[64, 32]", relu_14: "f32[64, 512, 1, 1]", convolution_18: "f32[64, 512, 1, 1]", squeeze_36: "f32[64, 32]", squeeze_37: "f32[64, 32]", relu_15: "f32[64, 512, 1, 1]", convolution_19: "f32[64, 512, 1, 1]", squeeze_38: "f32[64, 32]", squeeze_39: "f32[64, 32]", view_40: "f32[64, 512]", le: "b8[64, 512, 1, 1]", tangents_1: "f32[64, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        permute: "f32[512, 1000]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_1: "f32[1000, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[64, 512]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "f32[1000, 64]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 512]" = torch.ops.aten.mm.default(permute_2, view_40);  permute_2 = view_40 = None
        sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_41: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        view_42: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(mm, [64, 512, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        expand: "f32[64, 512, 1, 1]" = torch.ops.aten.expand.default(view_42, [64, 512, 1, 1]);  view_42 = None
        div: "f32[64, 512, 1, 1]" = torch.ops.aten.div.Scalar(expand, 1);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[64, 512, 1, 1]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_40: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(where, convolution_19)
        view_43: "f32[64, 512, 1]" = torch.ops.aten.reshape.default(mul_40, [64, 512, 1]);  mul_40 = None
        sum_2: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(view_43, [2]);  view_43 = None
        view_44: "f32[64, 512, 1]" = torch.ops.aten.reshape.default(where, [64, 512, 1])
        sum_3: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(view_44, [2]);  view_44 = None
        unsqueeze_114: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_60, 0)
        mul_41: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_2, unsqueeze_114)
        view_45: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_41, [64, 32, 16]);  mul_41 = None
        sum_4: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_45, [2]);  view_45 = None
        mul_42: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_3, unsqueeze_114);  unsqueeze_114 = None
        view_46: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_42, [64, 32, 16]);  mul_42 = None
        sum_5: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_46, [2]);  view_46 = None
        unsqueeze_122: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_39, -1)
        view_47: "f32[1, 32, 16]" = torch.ops.aten.reshape.default(primals_60, [1, 32, 16]);  primals_60 = None
        mul_43: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(unsqueeze_122, view_47);  view_47 = None
        mul_44: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_38)
        sub_20: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_44, sum_4);  mul_44 = sum_4 = None
        mul_45: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_20, squeeze_39);  sub_20 = None
        mul_46: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_45, squeeze_39);  mul_45 = None
        mul_47: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_46, squeeze_39);  mul_46 = None
        mul_48: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_47, 0.0625);  mul_47 = None
        neg: "f32[64, 32]" = torch.ops.aten.neg.default(mul_48)
        mul_49: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg, squeeze_38);  neg = None
        mul_50: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_39);  sum_5 = squeeze_39 = None
        mul_51: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_50, 0.0625);  mul_50 = None
        sub_21: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_49, mul_51);  mul_49 = mul_51 = None
        unsqueeze_123: "f32[64, 32, 16, 1]" = torch.ops.aten.unsqueeze.default(mul_43, -1);  mul_43 = None
        unsqueeze_124: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
        unsqueeze_125: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        unsqueeze_126: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_21, -1);  sub_21 = None
        unsqueeze_127: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        view_48: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(where, [64, 32, 16, 1])
        mul_52: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_48, unsqueeze_123);  view_48 = unsqueeze_123 = None
        view_38: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_19, [64, 32, 16, 1]);  convolution_19 = None
        mul_53: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_38, unsqueeze_125);  view_38 = unsqueeze_125 = None
        add_48: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
        add_49: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(add_48, unsqueeze_127);  add_48 = unsqueeze_127 = None
        view_50: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(add_49, [64, 512, 1, 1]);  add_49 = None
        view_51: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(sum_2, [64, 32, 16]);  sum_2 = None
        view_52: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(sum_3, [64, 32, 16])
        unsqueeze_128: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_38, -1);  squeeze_38 = None
        mul_54: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(view_52, unsqueeze_128);  view_52 = unsqueeze_128 = None
        sub_22: "f32[64, 32, 16]" = torch.ops.aten.sub.Tensor(view_51, mul_54);  view_51 = mul_54 = None
        mul_55: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_122);  sub_22 = unsqueeze_122 = None
        sum_6: "f32[32, 16]" = torch.ops.aten.sum.dim_IntList(mul_55, [0]);  mul_55 = None
        view_53: "f32[512]" = torch.ops.aten.reshape.default(sum_6, [512]);  sum_6 = None
        sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(sum_3, [0]);  sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward = torch.ops.aten.convolution_backward.default(view_50, relu_15, primals_59, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_50 = primals_59 = None
        getitem_42: "f32[64, 512, 1, 1]" = convolution_backward[0]
        getitem_43: "f32[512, 512, 3, 3]" = convolution_backward[1];  convolution_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_1: "b8[64, 512, 1, 1]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_1: "f32[64, 512, 1, 1]" = torch.ops.aten.where.self(le_1, full_default, getitem_42);  le_1 = getitem_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_56: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(where_1, convolution_18)
        view_54: "f32[64, 512, 1]" = torch.ops.aten.reshape.default(mul_56, [64, 512, 1]);  mul_56 = None
        sum_8: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(view_54, [2]);  view_54 = None
        view_55: "f32[64, 512, 1]" = torch.ops.aten.reshape.default(where_1, [64, 512, 1])
        sum_9: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(view_55, [2]);  view_55 = None
        unsqueeze_108: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_57, 0)
        mul_57: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_8, unsqueeze_108)
        view_56: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_57, [64, 32, 16]);  mul_57 = None
        sum_10: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_56, [2]);  view_56 = None
        mul_58: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_9, unsqueeze_108);  unsqueeze_108 = None
        view_57: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_58, [64, 32, 16]);  mul_58 = None
        sum_11: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_57, [2]);  view_57 = None
        unsqueeze_132: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_37, -1)
        view_58: "f32[1, 32, 16]" = torch.ops.aten.reshape.default(primals_57, [1, 32, 16]);  primals_57 = None
        mul_59: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(unsqueeze_132, view_58);  view_58 = None
        mul_60: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_36)
        sub_23: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_60, sum_10);  mul_60 = sum_10 = None
        mul_61: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_23, squeeze_37);  sub_23 = None
        mul_62: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_61, squeeze_37);  mul_61 = None
        mul_63: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_62, squeeze_37);  mul_62 = None
        mul_64: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_63, 0.0625);  mul_63 = None
        neg_1: "f32[64, 32]" = torch.ops.aten.neg.default(mul_64)
        mul_65: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_1, squeeze_36);  neg_1 = None
        mul_66: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_37);  sum_11 = squeeze_37 = None
        mul_67: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_66, 0.0625);  mul_66 = None
        sub_24: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_65, mul_67);  mul_65 = mul_67 = None
        unsqueeze_133: "f32[64, 32, 16, 1]" = torch.ops.aten.unsqueeze.default(mul_59, -1);  mul_59 = None
        unsqueeze_134: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_64, -1);  mul_64 = None
        unsqueeze_135: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        unsqueeze_136: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_24, -1);  sub_24 = None
        unsqueeze_137: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        view_59: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(where_1, [64, 32, 16, 1]);  where_1 = None
        mul_68: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_59, unsqueeze_133);  view_59 = unsqueeze_133 = None
        view_36: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_18, [64, 32, 16, 1]);  convolution_18 = None
        mul_69: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_36, unsqueeze_135);  view_36 = unsqueeze_135 = None
        add_50: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
        add_51: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(add_50, unsqueeze_137);  add_50 = unsqueeze_137 = None
        view_61: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(add_51, [64, 512, 1, 1]);  add_51 = None
        view_62: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(sum_8, [64, 32, 16]);  sum_8 = None
        view_63: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(sum_9, [64, 32, 16])
        unsqueeze_138: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_36, -1);  squeeze_36 = None
        mul_70: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(view_63, unsqueeze_138);  view_63 = unsqueeze_138 = None
        sub_25: "f32[64, 32, 16]" = torch.ops.aten.sub.Tensor(view_62, mul_70);  view_62 = mul_70 = None
        mul_71: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_132);  sub_25 = unsqueeze_132 = None
        sum_12: "f32[32, 16]" = torch.ops.aten.sum.dim_IntList(mul_71, [0]);  mul_71 = None
        view_64: "f32[512]" = torch.ops.aten.reshape.default(sum_12, [512]);  sum_12 = None
        sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(sum_9, [0]);  sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(view_61, relu_14, primals_56, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_61 = primals_56 = None
        getitem_45: "f32[64, 512, 1, 1]" = convolution_backward_1[0]
        getitem_46: "f32[512, 512, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
        add_52: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(where, getitem_45);  where = getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_2: "b8[64, 512, 1, 1]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_2: "f32[64, 512, 1, 1]" = torch.ops.aten.where.self(le_2, full_default, add_52);  le_2 = add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        mul_72: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(where_2, convolution_17)
        view_65: "f32[64, 512, 1]" = torch.ops.aten.reshape.default(mul_72, [64, 512, 1]);  mul_72 = None
        sum_14: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(view_65, [2]);  view_65 = None
        view_66: "f32[64, 512, 1]" = torch.ops.aten.reshape.default(where_2, [64, 512, 1])
        sum_15: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(view_66, [2]);  view_66 = None
        unsqueeze_102: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_54, 0)
        mul_73: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_14, unsqueeze_102)
        view_67: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_73, [64, 32, 16]);  mul_73 = None
        sum_16: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_67, [2]);  view_67 = None
        mul_74: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_15, unsqueeze_102);  unsqueeze_102 = None
        view_68: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_74, [64, 32, 16]);  mul_74 = None
        sum_17: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_68, [2]);  view_68 = None
        unsqueeze_142: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_35, -1)
        view_69: "f32[1, 32, 16]" = torch.ops.aten.reshape.default(primals_54, [1, 32, 16]);  primals_54 = None
        mul_75: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(unsqueeze_142, view_69);  view_69 = None
        mul_76: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_34)
        sub_26: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_76, sum_16);  mul_76 = sum_16 = None
        mul_77: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_26, squeeze_35);  sub_26 = None
        mul_78: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_77, squeeze_35);  mul_77 = None
        mul_79: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_78, squeeze_35);  mul_78 = None
        mul_80: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_79, 0.0625);  mul_79 = None
        neg_2: "f32[64, 32]" = torch.ops.aten.neg.default(mul_80)
        mul_81: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_2, squeeze_34);  neg_2 = None
        mul_82: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_35);  sum_17 = squeeze_35 = None
        mul_83: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_82, 0.0625);  mul_82 = None
        sub_27: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_81, mul_83);  mul_81 = mul_83 = None
        unsqueeze_143: "f32[64, 32, 16, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
        unsqueeze_144: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_80, -1);  mul_80 = None
        unsqueeze_145: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        unsqueeze_146: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_27, -1);  sub_27 = None
        unsqueeze_147: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        view_70: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(where_2, [64, 32, 16, 1])
        mul_84: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_70, unsqueeze_143);  unsqueeze_143 = None
        view_34: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_17, [64, 32, 16, 1]);  convolution_17 = None
        mul_85: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_34, unsqueeze_145);  view_34 = unsqueeze_145 = None
        add_53: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
        add_54: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(add_53, unsqueeze_147);  add_53 = unsqueeze_147 = None
        view_72: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(add_54, [64, 512, 1, 1]);  add_54 = None
        view_73: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(sum_14, [64, 32, 16]);  sum_14 = None
        view_74: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(sum_15, [64, 32, 16])
        unsqueeze_148: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_34, -1);  squeeze_34 = None
        mul_86: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(view_74, unsqueeze_148);  unsqueeze_148 = None
        sub_28: "f32[64, 32, 16]" = torch.ops.aten.sub.Tensor(view_73, mul_86);  view_73 = mul_86 = None
        mul_87: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_142);  sub_28 = unsqueeze_142 = None
        sum_18: "f32[32, 16]" = torch.ops.aten.sum.dim_IntList(mul_87, [0]);  mul_87 = None
        view_75: "f32[512]" = torch.ops.aten.reshape.default(sum_18, [512]);  sum_18 = None
        sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(sum_15, [0])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(view_72, relu_12, primals_53, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_72 = primals_53 = None
        getitem_48: "f32[64, 256, 2, 2]" = convolution_backward_2[0]
        getitem_49: "f32[512, 256, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_88: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(where_2, convolution_16);  where_2 = None
        view_76: "f32[64, 512, 1]" = torch.ops.aten.reshape.default(mul_88, [64, 512, 1]);  mul_88 = None
        sum_20: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(view_76, [2]);  view_76 = None
        unsqueeze_96: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_51, 0)
        mul_89: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_20, unsqueeze_96)
        view_78: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_89, [64, 32, 16]);  mul_89 = None
        sum_22: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_78, [2]);  view_78 = None
        mul_90: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_15, unsqueeze_96);  unsqueeze_96 = None
        view_79: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_90, [64, 32, 16]);  mul_90 = None
        sum_23: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_79, [2]);  view_79 = None
        unsqueeze_152: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_33, -1)
        view_80: "f32[1, 32, 16]" = torch.ops.aten.reshape.default(primals_51, [1, 32, 16]);  primals_51 = None
        mul_91: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(unsqueeze_152, view_80);  view_80 = None
        mul_92: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_32)
        sub_29: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_92, sum_22);  mul_92 = sum_22 = None
        mul_93: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_29, squeeze_33);  sub_29 = None
        mul_94: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_93, squeeze_33);  mul_93 = None
        mul_95: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_94, squeeze_33);  mul_94 = None
        mul_96: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_95, 0.0625);  mul_95 = None
        neg_3: "f32[64, 32]" = torch.ops.aten.neg.default(mul_96)
        mul_97: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_3, squeeze_32);  neg_3 = None
        mul_98: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_33);  sum_23 = squeeze_33 = None
        mul_99: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_98, 0.0625);  mul_98 = None
        sub_30: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_97, mul_99);  mul_97 = mul_99 = None
        unsqueeze_153: "f32[64, 32, 16, 1]" = torch.ops.aten.unsqueeze.default(mul_91, -1);  mul_91 = None
        unsqueeze_154: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
        unsqueeze_155: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        unsqueeze_156: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_30, -1);  sub_30 = None
        unsqueeze_157: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_100: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_70, unsqueeze_153);  view_70 = unsqueeze_153 = None
        view_32: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_16, [64, 32, 16, 1]);  convolution_16 = None
        mul_101: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_32, unsqueeze_155);  view_32 = unsqueeze_155 = None
        add_55: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(mul_100, mul_101);  mul_100 = mul_101 = None
        add_56: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(add_55, unsqueeze_157);  add_55 = unsqueeze_157 = None
        view_83: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(add_56, [64, 512, 1, 1]);  add_56 = None
        view_84: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(sum_20, [64, 32, 16]);  sum_20 = None
        unsqueeze_158: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_32, -1);  squeeze_32 = None
        mul_102: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(view_74, unsqueeze_158);  view_74 = unsqueeze_158 = None
        sub_31: "f32[64, 32, 16]" = torch.ops.aten.sub.Tensor(view_84, mul_102);  view_84 = mul_102 = None
        mul_103: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_152);  sub_31 = unsqueeze_152 = None
        sum_24: "f32[32, 16]" = torch.ops.aten.sum.dim_IntList(mul_103, [0]);  mul_103 = None
        view_86: "f32[512]" = torch.ops.aten.reshape.default(sum_24, [512]);  sum_24 = None
        sum_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(sum_15, [0]);  sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_83, relu_13, primals_50, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_83 = primals_50 = None
        getitem_51: "f32[64, 512, 1, 1]" = convolution_backward_3[0]
        getitem_52: "f32[512, 512, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_3: "b8[64, 512, 1, 1]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_3: "f32[64, 512, 1, 1]" = torch.ops.aten.where.self(le_3, full_default, getitem_51);  le_3 = getitem_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_104: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(where_3, convolution_15)
        view_87: "f32[64, 512, 1]" = torch.ops.aten.reshape.default(mul_104, [64, 512, 1]);  mul_104 = None
        sum_26: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(view_87, [2]);  view_87 = None
        view_88: "f32[64, 512, 1]" = torch.ops.aten.reshape.default(where_3, [64, 512, 1])
        sum_27: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(view_88, [2]);  view_88 = None
        unsqueeze_90: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_48, 0)
        mul_105: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_26, unsqueeze_90)
        view_89: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_105, [64, 32, 16]);  mul_105 = None
        sum_28: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_89, [2]);  view_89 = None
        mul_106: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_27, unsqueeze_90);  unsqueeze_90 = None
        view_90: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_106, [64, 32, 16]);  mul_106 = None
        sum_29: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_90, [2]);  view_90 = None
        unsqueeze_162: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_31, -1)
        view_91: "f32[1, 32, 16]" = torch.ops.aten.reshape.default(primals_48, [1, 32, 16]);  primals_48 = None
        mul_107: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(unsqueeze_162, view_91);  view_91 = None
        mul_108: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_30)
        sub_32: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_108, sum_28);  mul_108 = sum_28 = None
        mul_109: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_32, squeeze_31);  sub_32 = None
        mul_110: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_109, squeeze_31);  mul_109 = None
        mul_111: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_110, squeeze_31);  mul_110 = None
        mul_112: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_111, 0.0625);  mul_111 = None
        neg_4: "f32[64, 32]" = torch.ops.aten.neg.default(mul_112)
        mul_113: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_4, squeeze_30);  neg_4 = None
        mul_114: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_31);  sum_29 = squeeze_31 = None
        mul_115: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_114, 0.0625);  mul_114 = None
        sub_33: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_113, mul_115);  mul_113 = mul_115 = None
        unsqueeze_163: "f32[64, 32, 16, 1]" = torch.ops.aten.unsqueeze.default(mul_107, -1);  mul_107 = None
        unsqueeze_164: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_112, -1);  mul_112 = None
        unsqueeze_165: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        unsqueeze_166: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_33, -1);  sub_33 = None
        unsqueeze_167: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        view_92: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(where_3, [64, 32, 16, 1]);  where_3 = None
        mul_116: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_92, unsqueeze_163);  view_92 = unsqueeze_163 = None
        view_30: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_15, [64, 32, 16, 1]);  convolution_15 = None
        mul_117: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_30, unsqueeze_165);  view_30 = unsqueeze_165 = None
        add_57: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        add_58: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(add_57, unsqueeze_167);  add_57 = unsqueeze_167 = None
        view_94: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(add_58, [64, 512, 1, 1]);  add_58 = None
        view_95: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(sum_26, [64, 32, 16]);  sum_26 = None
        view_96: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(sum_27, [64, 32, 16])
        unsqueeze_168: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_30, -1);  squeeze_30 = None
        mul_118: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(view_96, unsqueeze_168);  view_96 = unsqueeze_168 = None
        sub_34: "f32[64, 32, 16]" = torch.ops.aten.sub.Tensor(view_95, mul_118);  view_95 = mul_118 = None
        mul_119: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_162);  sub_34 = unsqueeze_162 = None
        sum_30: "f32[32, 16]" = torch.ops.aten.sum.dim_IntList(mul_119, [0]);  mul_119 = None
        view_97: "f32[512]" = torch.ops.aten.reshape.default(sum_30, [512]);  sum_30 = None
        sum_31: "f32[512]" = torch.ops.aten.sum.dim_IntList(sum_27, [0]);  sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(view_94, relu_12, primals_47, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_94 = primals_47 = None
        getitem_54: "f32[64, 256, 2, 2]" = convolution_backward_4[0]
        getitem_55: "f32[512, 256, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
        add_59: "f32[64, 256, 2, 2]" = torch.ops.aten.add.Tensor(getitem_48, getitem_54);  getitem_48 = getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_4: "b8[64, 256, 2, 2]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_4: "f32[64, 256, 2, 2]" = torch.ops.aten.where.self(le_4, full_default, add_59);  le_4 = add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_120: "f32[64, 256, 2, 2]" = torch.ops.aten.mul.Tensor(where_4, convolution_14)
        view_98: "f32[64, 256, 4]" = torch.ops.aten.reshape.default(mul_120, [64, 256, 4]);  mul_120 = None
        sum_32: "f32[64, 256]" = torch.ops.aten.sum.dim_IntList(view_98, [2]);  view_98 = None
        view_99: "f32[64, 256, 4]" = torch.ops.aten.reshape.default(where_4, [64, 256, 4])
        sum_33: "f32[64, 256]" = torch.ops.aten.sum.dim_IntList(view_99, [2]);  view_99 = None
        unsqueeze_84: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_45, 0)
        mul_121: "f32[64, 256]" = torch.ops.aten.mul.Tensor(sum_32, unsqueeze_84)
        view_100: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(mul_121, [64, 32, 8]);  mul_121 = None
        sum_34: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_100, [2]);  view_100 = None
        mul_122: "f32[64, 256]" = torch.ops.aten.mul.Tensor(sum_33, unsqueeze_84);  unsqueeze_84 = None
        view_101: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(mul_122, [64, 32, 8]);  mul_122 = None
        sum_35: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_101, [2]);  view_101 = None
        unsqueeze_172: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_29, -1)
        view_102: "f32[1, 32, 8]" = torch.ops.aten.reshape.default(primals_45, [1, 32, 8]);  primals_45 = None
        mul_123: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(unsqueeze_172, view_102);  view_102 = None
        mul_124: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_28)
        sub_35: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_124, sum_34);  mul_124 = sum_34 = None
        mul_125: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_35, squeeze_29);  sub_35 = None
        mul_126: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_125, squeeze_29);  mul_125 = None
        mul_127: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_126, squeeze_29);  mul_126 = None
        mul_128: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_127, 0.03125);  mul_127 = None
        neg_5: "f32[64, 32]" = torch.ops.aten.neg.default(mul_128)
        mul_129: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_5, squeeze_28);  neg_5 = None
        mul_130: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_29);  sum_35 = squeeze_29 = None
        mul_131: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_130, 0.03125);  mul_130 = None
        sub_36: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_129, mul_131);  mul_129 = mul_131 = None
        unsqueeze_173: "f32[64, 32, 8, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_174: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_128, -1);  mul_128 = None
        unsqueeze_175: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        unsqueeze_176: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_36, -1);  sub_36 = None
        unsqueeze_177: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        view_103: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(where_4, [64, 32, 8, 4])
        mul_132: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(view_103, unsqueeze_173);  view_103 = unsqueeze_173 = None
        view_28: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(convolution_14, [64, 32, 8, 4]);  convolution_14 = None
        mul_133: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(view_28, unsqueeze_175);  view_28 = unsqueeze_175 = None
        add_60: "f32[64, 32, 8, 4]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
        add_61: "f32[64, 32, 8, 4]" = torch.ops.aten.add.Tensor(add_60, unsqueeze_177);  add_60 = unsqueeze_177 = None
        view_105: "f32[64, 256, 2, 2]" = torch.ops.aten.reshape.default(add_61, [64, 256, 2, 2]);  add_61 = None
        view_106: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(sum_32, [64, 32, 8]);  sum_32 = None
        view_107: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(sum_33, [64, 32, 8])
        unsqueeze_178: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_28, -1);  squeeze_28 = None
        mul_134: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(view_107, unsqueeze_178);  view_107 = unsqueeze_178 = None
        sub_37: "f32[64, 32, 8]" = torch.ops.aten.sub.Tensor(view_106, mul_134);  view_106 = mul_134 = None
        mul_135: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_172);  sub_37 = unsqueeze_172 = None
        sum_36: "f32[32, 8]" = torch.ops.aten.sum.dim_IntList(mul_135, [0]);  mul_135 = None
        view_108: "f32[256]" = torch.ops.aten.reshape.default(sum_36, [256]);  sum_36 = None
        sum_37: "f32[256]" = torch.ops.aten.sum.dim_IntList(sum_33, [0]);  sum_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(view_105, relu_11, primals_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_105 = primals_44 = None
        getitem_57: "f32[64, 256, 2, 2]" = convolution_backward_5[0]
        getitem_58: "f32[256, 256, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_5: "b8[64, 256, 2, 2]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_5: "f32[64, 256, 2, 2]" = torch.ops.aten.where.self(le_5, full_default, getitem_57);  le_5 = getitem_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_136: "f32[64, 256, 2, 2]" = torch.ops.aten.mul.Tensor(where_5, convolution_13)
        view_109: "f32[64, 256, 4]" = torch.ops.aten.reshape.default(mul_136, [64, 256, 4]);  mul_136 = None
        sum_38: "f32[64, 256]" = torch.ops.aten.sum.dim_IntList(view_109, [2]);  view_109 = None
        view_110: "f32[64, 256, 4]" = torch.ops.aten.reshape.default(where_5, [64, 256, 4])
        sum_39: "f32[64, 256]" = torch.ops.aten.sum.dim_IntList(view_110, [2]);  view_110 = None
        unsqueeze_78: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_42, 0)
        mul_137: "f32[64, 256]" = torch.ops.aten.mul.Tensor(sum_38, unsqueeze_78)
        view_111: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(mul_137, [64, 32, 8]);  mul_137 = None
        sum_40: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_111, [2]);  view_111 = None
        mul_138: "f32[64, 256]" = torch.ops.aten.mul.Tensor(sum_39, unsqueeze_78);  unsqueeze_78 = None
        view_112: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(mul_138, [64, 32, 8]);  mul_138 = None
        sum_41: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_112, [2]);  view_112 = None
        unsqueeze_182: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_27, -1)
        view_113: "f32[1, 32, 8]" = torch.ops.aten.reshape.default(primals_42, [1, 32, 8]);  primals_42 = None
        mul_139: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(unsqueeze_182, view_113);  view_113 = None
        mul_140: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_26)
        sub_38: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_140, sum_40);  mul_140 = sum_40 = None
        mul_141: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_38, squeeze_27);  sub_38 = None
        mul_142: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_141, squeeze_27);  mul_141 = None
        mul_143: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_142, squeeze_27);  mul_142 = None
        mul_144: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_143, 0.03125);  mul_143 = None
        neg_6: "f32[64, 32]" = torch.ops.aten.neg.default(mul_144)
        mul_145: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_6, squeeze_26);  neg_6 = None
        mul_146: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_27);  sum_41 = squeeze_27 = None
        mul_147: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_146, 0.03125);  mul_146 = None
        sub_39: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_145, mul_147);  mul_145 = mul_147 = None
        unsqueeze_183: "f32[64, 32, 8, 1]" = torch.ops.aten.unsqueeze.default(mul_139, -1);  mul_139 = None
        unsqueeze_184: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
        unsqueeze_185: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        unsqueeze_186: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_39, -1);  sub_39 = None
        unsqueeze_187: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        view_114: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(where_5, [64, 32, 8, 4]);  where_5 = None
        mul_148: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(view_114, unsqueeze_183);  view_114 = unsqueeze_183 = None
        view_26: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(convolution_13, [64, 32, 8, 4]);  convolution_13 = None
        mul_149: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(view_26, unsqueeze_185);  view_26 = unsqueeze_185 = None
        add_62: "f32[64, 32, 8, 4]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        add_63: "f32[64, 32, 8, 4]" = torch.ops.aten.add.Tensor(add_62, unsqueeze_187);  add_62 = unsqueeze_187 = None
        view_116: "f32[64, 256, 2, 2]" = torch.ops.aten.reshape.default(add_63, [64, 256, 2, 2]);  add_63 = None
        view_117: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(sum_38, [64, 32, 8]);  sum_38 = None
        view_118: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(sum_39, [64, 32, 8])
        unsqueeze_188: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_26, -1);  squeeze_26 = None
        mul_150: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(view_118, unsqueeze_188);  view_118 = unsqueeze_188 = None
        sub_40: "f32[64, 32, 8]" = torch.ops.aten.sub.Tensor(view_117, mul_150);  view_117 = mul_150 = None
        mul_151: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_182);  sub_40 = unsqueeze_182 = None
        sum_42: "f32[32, 8]" = torch.ops.aten.sum.dim_IntList(mul_151, [0]);  mul_151 = None
        view_119: "f32[256]" = torch.ops.aten.reshape.default(sum_42, [256]);  sum_42 = None
        sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(sum_39, [0]);  sum_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(view_116, relu_10, primals_41, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_116 = primals_41 = None
        getitem_60: "f32[64, 256, 2, 2]" = convolution_backward_6[0]
        getitem_61: "f32[256, 256, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
        add_64: "f32[64, 256, 2, 2]" = torch.ops.aten.add.Tensor(where_4, getitem_60);  where_4 = getitem_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_6: "b8[64, 256, 2, 2]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_6: "f32[64, 256, 2, 2]" = torch.ops.aten.where.self(le_6, full_default, add_64);  le_6 = add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        mul_152: "f32[64, 256, 2, 2]" = torch.ops.aten.mul.Tensor(where_6, convolution_12)
        view_120: "f32[64, 256, 4]" = torch.ops.aten.reshape.default(mul_152, [64, 256, 4]);  mul_152 = None
        sum_44: "f32[64, 256]" = torch.ops.aten.sum.dim_IntList(view_120, [2]);  view_120 = None
        view_121: "f32[64, 256, 4]" = torch.ops.aten.reshape.default(where_6, [64, 256, 4])
        sum_45: "f32[64, 256]" = torch.ops.aten.sum.dim_IntList(view_121, [2]);  view_121 = None
        unsqueeze_72: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_39, 0)
        mul_153: "f32[64, 256]" = torch.ops.aten.mul.Tensor(sum_44, unsqueeze_72)
        view_122: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(mul_153, [64, 32, 8]);  mul_153 = None
        sum_46: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_122, [2]);  view_122 = None
        mul_154: "f32[64, 256]" = torch.ops.aten.mul.Tensor(sum_45, unsqueeze_72);  unsqueeze_72 = None
        view_123: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(mul_154, [64, 32, 8]);  mul_154 = None
        sum_47: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_123, [2]);  view_123 = None
        unsqueeze_192: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_25, -1)
        view_124: "f32[1, 32, 8]" = torch.ops.aten.reshape.default(primals_39, [1, 32, 8]);  primals_39 = None
        mul_155: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(unsqueeze_192, view_124);  view_124 = None
        mul_156: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_24)
        sub_41: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_156, sum_46);  mul_156 = sum_46 = None
        mul_157: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_41, squeeze_25);  sub_41 = None
        mul_158: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_157, squeeze_25);  mul_157 = None
        mul_159: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_158, squeeze_25);  mul_158 = None
        mul_160: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_159, 0.03125);  mul_159 = None
        neg_7: "f32[64, 32]" = torch.ops.aten.neg.default(mul_160)
        mul_161: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_7, squeeze_24);  neg_7 = None
        mul_162: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_25);  sum_47 = squeeze_25 = None
        mul_163: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_162, 0.03125);  mul_162 = None
        sub_42: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_161, mul_163);  mul_161 = mul_163 = None
        unsqueeze_193: "f32[64, 32, 8, 1]" = torch.ops.aten.unsqueeze.default(mul_155, -1);  mul_155 = None
        unsqueeze_194: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_160, -1);  mul_160 = None
        unsqueeze_195: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        unsqueeze_196: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_42, -1);  sub_42 = None
        unsqueeze_197: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        view_125: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(where_6, [64, 32, 8, 4])
        mul_164: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(view_125, unsqueeze_193);  unsqueeze_193 = None
        view_24: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(convolution_12, [64, 32, 8, 4]);  convolution_12 = None
        mul_165: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(view_24, unsqueeze_195);  view_24 = unsqueeze_195 = None
        add_65: "f32[64, 32, 8, 4]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
        add_66: "f32[64, 32, 8, 4]" = torch.ops.aten.add.Tensor(add_65, unsqueeze_197);  add_65 = unsqueeze_197 = None
        view_127: "f32[64, 256, 2, 2]" = torch.ops.aten.reshape.default(add_66, [64, 256, 2, 2]);  add_66 = None
        view_128: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(sum_44, [64, 32, 8]);  sum_44 = None
        view_129: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(sum_45, [64, 32, 8])
        unsqueeze_198: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_24, -1);  squeeze_24 = None
        mul_166: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(view_129, unsqueeze_198);  unsqueeze_198 = None
        sub_43: "f32[64, 32, 8]" = torch.ops.aten.sub.Tensor(view_128, mul_166);  view_128 = mul_166 = None
        mul_167: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_192);  sub_43 = unsqueeze_192 = None
        sum_48: "f32[32, 8]" = torch.ops.aten.sum.dim_IntList(mul_167, [0]);  mul_167 = None
        view_130: "f32[256]" = torch.ops.aten.reshape.default(sum_48, [256]);  sum_48 = None
        sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(sum_45, [0])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(view_127, relu_8, primals_38, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_127 = primals_38 = None
        getitem_63: "f32[64, 128, 4, 4]" = convolution_backward_7[0]
        getitem_64: "f32[256, 128, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_168: "f32[64, 256, 2, 2]" = torch.ops.aten.mul.Tensor(where_6, convolution_11);  where_6 = None
        view_131: "f32[64, 256, 4]" = torch.ops.aten.reshape.default(mul_168, [64, 256, 4]);  mul_168 = None
        sum_50: "f32[64, 256]" = torch.ops.aten.sum.dim_IntList(view_131, [2]);  view_131 = None
        unsqueeze_66: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_36, 0)
        mul_169: "f32[64, 256]" = torch.ops.aten.mul.Tensor(sum_50, unsqueeze_66)
        view_133: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(mul_169, [64, 32, 8]);  mul_169 = None
        sum_52: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_133, [2]);  view_133 = None
        mul_170: "f32[64, 256]" = torch.ops.aten.mul.Tensor(sum_45, unsqueeze_66);  unsqueeze_66 = None
        view_134: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(mul_170, [64, 32, 8]);  mul_170 = None
        sum_53: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_134, [2]);  view_134 = None
        unsqueeze_202: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_23, -1)
        view_135: "f32[1, 32, 8]" = torch.ops.aten.reshape.default(primals_36, [1, 32, 8]);  primals_36 = None
        mul_171: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(unsqueeze_202, view_135);  view_135 = None
        mul_172: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_22)
        sub_44: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_172, sum_52);  mul_172 = sum_52 = None
        mul_173: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_44, squeeze_23);  sub_44 = None
        mul_174: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_173, squeeze_23);  mul_173 = None
        mul_175: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_174, squeeze_23);  mul_174 = None
        mul_176: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_175, 0.03125);  mul_175 = None
        neg_8: "f32[64, 32]" = torch.ops.aten.neg.default(mul_176)
        mul_177: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_8, squeeze_22);  neg_8 = None
        mul_178: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_23);  sum_53 = squeeze_23 = None
        mul_179: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_178, 0.03125);  mul_178 = None
        sub_45: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_177, mul_179);  mul_177 = mul_179 = None
        unsqueeze_203: "f32[64, 32, 8, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_204: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_176, -1);  mul_176 = None
        unsqueeze_205: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        unsqueeze_206: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_45, -1);  sub_45 = None
        unsqueeze_207: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        mul_180: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(view_125, unsqueeze_203);  view_125 = unsqueeze_203 = None
        view_22: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(convolution_11, [64, 32, 8, 4]);  convolution_11 = None
        mul_181: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(view_22, unsqueeze_205);  view_22 = unsqueeze_205 = None
        add_67: "f32[64, 32, 8, 4]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
        add_68: "f32[64, 32, 8, 4]" = torch.ops.aten.add.Tensor(add_67, unsqueeze_207);  add_67 = unsqueeze_207 = None
        view_138: "f32[64, 256, 2, 2]" = torch.ops.aten.reshape.default(add_68, [64, 256, 2, 2]);  add_68 = None
        view_139: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(sum_50, [64, 32, 8]);  sum_50 = None
        unsqueeze_208: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_22, -1);  squeeze_22 = None
        mul_182: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(view_129, unsqueeze_208);  view_129 = unsqueeze_208 = None
        sub_46: "f32[64, 32, 8]" = torch.ops.aten.sub.Tensor(view_139, mul_182);  view_139 = mul_182 = None
        mul_183: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_202);  sub_46 = unsqueeze_202 = None
        sum_54: "f32[32, 8]" = torch.ops.aten.sum.dim_IntList(mul_183, [0]);  mul_183 = None
        view_141: "f32[256]" = torch.ops.aten.reshape.default(sum_54, [256]);  sum_54 = None
        sum_55: "f32[256]" = torch.ops.aten.sum.dim_IntList(sum_45, [0]);  sum_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(view_138, relu_9, primals_35, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_138 = primals_35 = None
        getitem_66: "f32[64, 256, 2, 2]" = convolution_backward_8[0]
        getitem_67: "f32[256, 256, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_7: "b8[64, 256, 2, 2]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_7: "f32[64, 256, 2, 2]" = torch.ops.aten.where.self(le_7, full_default, getitem_66);  le_7 = getitem_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_184: "f32[64, 256, 2, 2]" = torch.ops.aten.mul.Tensor(where_7, convolution_10)
        view_142: "f32[64, 256, 4]" = torch.ops.aten.reshape.default(mul_184, [64, 256, 4]);  mul_184 = None
        sum_56: "f32[64, 256]" = torch.ops.aten.sum.dim_IntList(view_142, [2]);  view_142 = None
        view_143: "f32[64, 256, 4]" = torch.ops.aten.reshape.default(where_7, [64, 256, 4])
        sum_57: "f32[64, 256]" = torch.ops.aten.sum.dim_IntList(view_143, [2]);  view_143 = None
        unsqueeze_60: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_33, 0)
        mul_185: "f32[64, 256]" = torch.ops.aten.mul.Tensor(sum_56, unsqueeze_60)
        view_144: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(mul_185, [64, 32, 8]);  mul_185 = None
        sum_58: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_144, [2]);  view_144 = None
        mul_186: "f32[64, 256]" = torch.ops.aten.mul.Tensor(sum_57, unsqueeze_60);  unsqueeze_60 = None
        view_145: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(mul_186, [64, 32, 8]);  mul_186 = None
        sum_59: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_145, [2]);  view_145 = None
        unsqueeze_212: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_21, -1)
        view_146: "f32[1, 32, 8]" = torch.ops.aten.reshape.default(primals_33, [1, 32, 8]);  primals_33 = None
        mul_187: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(unsqueeze_212, view_146);  view_146 = None
        mul_188: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_20)
        sub_47: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_188, sum_58);  mul_188 = sum_58 = None
        mul_189: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_47, squeeze_21);  sub_47 = None
        mul_190: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_189, squeeze_21);  mul_189 = None
        mul_191: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_190, squeeze_21);  mul_190 = None
        mul_192: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_191, 0.03125);  mul_191 = None
        neg_9: "f32[64, 32]" = torch.ops.aten.neg.default(mul_192)
        mul_193: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_9, squeeze_20);  neg_9 = None
        mul_194: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_21);  sum_59 = squeeze_21 = None
        mul_195: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_194, 0.03125);  mul_194 = None
        sub_48: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_193, mul_195);  mul_193 = mul_195 = None
        unsqueeze_213: "f32[64, 32, 8, 1]" = torch.ops.aten.unsqueeze.default(mul_187, -1);  mul_187 = None
        unsqueeze_214: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
        unsqueeze_215: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        unsqueeze_216: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_48, -1);  sub_48 = None
        unsqueeze_217: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        view_147: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(where_7, [64, 32, 8, 4]);  where_7 = None
        mul_196: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(view_147, unsqueeze_213);  view_147 = unsqueeze_213 = None
        view_20: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(convolution_10, [64, 32, 8, 4]);  convolution_10 = None
        mul_197: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(view_20, unsqueeze_215);  view_20 = unsqueeze_215 = None
        add_69: "f32[64, 32, 8, 4]" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
        add_70: "f32[64, 32, 8, 4]" = torch.ops.aten.add.Tensor(add_69, unsqueeze_217);  add_69 = unsqueeze_217 = None
        view_149: "f32[64, 256, 2, 2]" = torch.ops.aten.reshape.default(add_70, [64, 256, 2, 2]);  add_70 = None
        view_150: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(sum_56, [64, 32, 8]);  sum_56 = None
        view_151: "f32[64, 32, 8]" = torch.ops.aten.reshape.default(sum_57, [64, 32, 8])
        unsqueeze_218: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_20, -1);  squeeze_20 = None
        mul_198: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(view_151, unsqueeze_218);  view_151 = unsqueeze_218 = None
        sub_49: "f32[64, 32, 8]" = torch.ops.aten.sub.Tensor(view_150, mul_198);  view_150 = mul_198 = None
        mul_199: "f32[64, 32, 8]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_212);  sub_49 = unsqueeze_212 = None
        sum_60: "f32[32, 8]" = torch.ops.aten.sum.dim_IntList(mul_199, [0]);  mul_199 = None
        view_152: "f32[256]" = torch.ops.aten.reshape.default(sum_60, [256]);  sum_60 = None
        sum_61: "f32[256]" = torch.ops.aten.sum.dim_IntList(sum_57, [0]);  sum_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(view_149, relu_8, primals_32, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_149 = primals_32 = None
        getitem_69: "f32[64, 128, 4, 4]" = convolution_backward_9[0]
        getitem_70: "f32[256, 128, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
        add_71: "f32[64, 128, 4, 4]" = torch.ops.aten.add.Tensor(getitem_63, getitem_69);  getitem_63 = getitem_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_8: "b8[64, 128, 4, 4]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_8: "f32[64, 128, 4, 4]" = torch.ops.aten.where.self(le_8, full_default, add_71);  le_8 = add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_200: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(where_8, convolution_9)
        view_153: "f32[64, 128, 16]" = torch.ops.aten.reshape.default(mul_200, [64, 128, 16]);  mul_200 = None
        sum_62: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_153, [2]);  view_153 = None
        view_154: "f32[64, 128, 16]" = torch.ops.aten.reshape.default(where_8, [64, 128, 16])
        sum_63: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_154, [2]);  view_154 = None
        unsqueeze_54: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_30, 0)
        mul_201: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_62, unsqueeze_54)
        view_155: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_201, [64, 32, 4]);  mul_201 = None
        sum_64: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_155, [2]);  view_155 = None
        mul_202: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_63, unsqueeze_54);  unsqueeze_54 = None
        view_156: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_202, [64, 32, 4]);  mul_202 = None
        sum_65: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_156, [2]);  view_156 = None
        unsqueeze_222: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_19, -1)
        view_157: "f32[1, 32, 4]" = torch.ops.aten.reshape.default(primals_30, [1, 32, 4]);  primals_30 = None
        mul_203: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_222, view_157);  view_157 = None
        mul_204: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_18)
        sub_50: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_204, sum_64);  mul_204 = sum_64 = None
        mul_205: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_50, squeeze_19);  sub_50 = None
        mul_206: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_205, squeeze_19);  mul_205 = None
        mul_207: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_206, squeeze_19);  mul_206 = None
        mul_208: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_207, 0.015625);  mul_207 = None
        neg_10: "f32[64, 32]" = torch.ops.aten.neg.default(mul_208)
        mul_209: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_10, squeeze_18);  neg_10 = None
        mul_210: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_19);  sum_65 = squeeze_19 = None
        mul_211: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_210, 0.015625);  mul_210 = None
        sub_51: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_209, mul_211);  mul_209 = mul_211 = None
        unsqueeze_223: "f32[64, 32, 4, 1]" = torch.ops.aten.unsqueeze.default(mul_203, -1);  mul_203 = None
        unsqueeze_224: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_208, -1);  mul_208 = None
        unsqueeze_225: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        unsqueeze_226: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_51, -1);  sub_51 = None
        unsqueeze_227: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        view_158: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(where_8, [64, 32, 4, 16])
        mul_212: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_158, unsqueeze_223);  view_158 = unsqueeze_223 = None
        view_18: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(convolution_9, [64, 32, 4, 16]);  convolution_9 = None
        mul_213: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_18, unsqueeze_225);  view_18 = unsqueeze_225 = None
        add_72: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
        add_73: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(add_72, unsqueeze_227);  add_72 = unsqueeze_227 = None
        view_160: "f32[64, 128, 4, 4]" = torch.ops.aten.reshape.default(add_73, [64, 128, 4, 4]);  add_73 = None
        view_161: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(sum_62, [64, 32, 4]);  sum_62 = None
        view_162: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(sum_63, [64, 32, 4])
        unsqueeze_228: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_18, -1);  squeeze_18 = None
        mul_214: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(view_162, unsqueeze_228);  view_162 = unsqueeze_228 = None
        sub_52: "f32[64, 32, 4]" = torch.ops.aten.sub.Tensor(view_161, mul_214);  view_161 = mul_214 = None
        mul_215: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_222);  sub_52 = unsqueeze_222 = None
        sum_66: "f32[32, 4]" = torch.ops.aten.sum.dim_IntList(mul_215, [0]);  mul_215 = None
        view_163: "f32[128]" = torch.ops.aten.reshape.default(sum_66, [128]);  sum_66 = None
        sum_67: "f32[128]" = torch.ops.aten.sum.dim_IntList(sum_63, [0]);  sum_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(view_160, relu_7, primals_29, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_160 = primals_29 = None
        getitem_72: "f32[64, 128, 4, 4]" = convolution_backward_10[0]
        getitem_73: "f32[128, 128, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_9: "b8[64, 128, 4, 4]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_9: "f32[64, 128, 4, 4]" = torch.ops.aten.where.self(le_9, full_default, getitem_72);  le_9 = getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_216: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(where_9, convolution_8)
        view_164: "f32[64, 128, 16]" = torch.ops.aten.reshape.default(mul_216, [64, 128, 16]);  mul_216 = None
        sum_68: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_164, [2]);  view_164 = None
        view_165: "f32[64, 128, 16]" = torch.ops.aten.reshape.default(where_9, [64, 128, 16])
        sum_69: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_165, [2]);  view_165 = None
        unsqueeze_48: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_27, 0)
        mul_217: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_68, unsqueeze_48)
        view_166: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_217, [64, 32, 4]);  mul_217 = None
        sum_70: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_166, [2]);  view_166 = None
        mul_218: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_69, unsqueeze_48);  unsqueeze_48 = None
        view_167: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_218, [64, 32, 4]);  mul_218 = None
        sum_71: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_167, [2]);  view_167 = None
        unsqueeze_232: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_17, -1)
        view_168: "f32[1, 32, 4]" = torch.ops.aten.reshape.default(primals_27, [1, 32, 4]);  primals_27 = None
        mul_219: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_232, view_168);  view_168 = None
        mul_220: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_16)
        sub_53: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_220, sum_70);  mul_220 = sum_70 = None
        mul_221: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_53, squeeze_17);  sub_53 = None
        mul_222: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_221, squeeze_17);  mul_221 = None
        mul_223: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_222, squeeze_17);  mul_222 = None
        mul_224: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_223, 0.015625);  mul_223 = None
        neg_11: "f32[64, 32]" = torch.ops.aten.neg.default(mul_224)
        mul_225: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_11, squeeze_16);  neg_11 = None
        mul_226: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_17);  sum_71 = squeeze_17 = None
        mul_227: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_226, 0.015625);  mul_226 = None
        sub_54: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_225, mul_227);  mul_225 = mul_227 = None
        unsqueeze_233: "f32[64, 32, 4, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_234: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_224, -1);  mul_224 = None
        unsqueeze_235: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        unsqueeze_236: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_54, -1);  sub_54 = None
        unsqueeze_237: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        view_169: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(where_9, [64, 32, 4, 16]);  where_9 = None
        mul_228: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_169, unsqueeze_233);  view_169 = unsqueeze_233 = None
        view_16: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(convolution_8, [64, 32, 4, 16]);  convolution_8 = None
        mul_229: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_16, unsqueeze_235);  view_16 = unsqueeze_235 = None
        add_74: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        add_75: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(add_74, unsqueeze_237);  add_74 = unsqueeze_237 = None
        view_171: "f32[64, 128, 4, 4]" = torch.ops.aten.reshape.default(add_75, [64, 128, 4, 4]);  add_75 = None
        view_172: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(sum_68, [64, 32, 4]);  sum_68 = None
        view_173: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(sum_69, [64, 32, 4])
        unsqueeze_238: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_16, -1);  squeeze_16 = None
        mul_230: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(view_173, unsqueeze_238);  view_173 = unsqueeze_238 = None
        sub_55: "f32[64, 32, 4]" = torch.ops.aten.sub.Tensor(view_172, mul_230);  view_172 = mul_230 = None
        mul_231: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_232);  sub_55 = unsqueeze_232 = None
        sum_72: "f32[32, 4]" = torch.ops.aten.sum.dim_IntList(mul_231, [0]);  mul_231 = None
        view_174: "f32[128]" = torch.ops.aten.reshape.default(sum_72, [128]);  sum_72 = None
        sum_73: "f32[128]" = torch.ops.aten.sum.dim_IntList(sum_69, [0]);  sum_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(view_171, relu_6, primals_26, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_171 = primals_26 = None
        getitem_75: "f32[64, 128, 4, 4]" = convolution_backward_11[0]
        getitem_76: "f32[128, 128, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
        add_76: "f32[64, 128, 4, 4]" = torch.ops.aten.add.Tensor(where_8, getitem_75);  where_8 = getitem_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_10: "b8[64, 128, 4, 4]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_10: "f32[64, 128, 4, 4]" = torch.ops.aten.where.self(le_10, full_default, add_76);  le_10 = add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        mul_232: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(where_10, convolution_7)
        view_175: "f32[64, 128, 16]" = torch.ops.aten.reshape.default(mul_232, [64, 128, 16]);  mul_232 = None
        sum_74: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_175, [2]);  view_175 = None
        view_176: "f32[64, 128, 16]" = torch.ops.aten.reshape.default(where_10, [64, 128, 16])
        sum_75: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_176, [2]);  view_176 = None
        unsqueeze_42: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_24, 0)
        mul_233: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_74, unsqueeze_42)
        view_177: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_233, [64, 32, 4]);  mul_233 = None
        sum_76: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_177, [2]);  view_177 = None
        mul_234: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_75, unsqueeze_42);  unsqueeze_42 = None
        view_178: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_234, [64, 32, 4]);  mul_234 = None
        sum_77: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_178, [2]);  view_178 = None
        unsqueeze_242: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_15, -1)
        view_179: "f32[1, 32, 4]" = torch.ops.aten.reshape.default(primals_24, [1, 32, 4]);  primals_24 = None
        mul_235: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_242, view_179);  view_179 = None
        mul_236: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_14)
        sub_56: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_236, sum_76);  mul_236 = sum_76 = None
        mul_237: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_56, squeeze_15);  sub_56 = None
        mul_238: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_237, squeeze_15);  mul_237 = None
        mul_239: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_238, squeeze_15);  mul_238 = None
        mul_240: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_239, 0.015625);  mul_239 = None
        neg_12: "f32[64, 32]" = torch.ops.aten.neg.default(mul_240)
        mul_241: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_12, squeeze_14);  neg_12 = None
        mul_242: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_15);  sum_77 = squeeze_15 = None
        mul_243: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_242, 0.015625);  mul_242 = None
        sub_57: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_241, mul_243);  mul_241 = mul_243 = None
        unsqueeze_243: "f32[64, 32, 4, 1]" = torch.ops.aten.unsqueeze.default(mul_235, -1);  mul_235 = None
        unsqueeze_244: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_245: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        unsqueeze_246: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_57, -1);  sub_57 = None
        unsqueeze_247: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        view_180: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(where_10, [64, 32, 4, 16])
        mul_244: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_180, unsqueeze_243);  unsqueeze_243 = None
        view_14: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(convolution_7, [64, 32, 4, 16]);  convolution_7 = None
        mul_245: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_14, unsqueeze_245);  view_14 = unsqueeze_245 = None
        add_77: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
        add_78: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(add_77, unsqueeze_247);  add_77 = unsqueeze_247 = None
        view_182: "f32[64, 128, 4, 4]" = torch.ops.aten.reshape.default(add_78, [64, 128, 4, 4]);  add_78 = None
        view_183: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(sum_74, [64, 32, 4]);  sum_74 = None
        view_184: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(sum_75, [64, 32, 4])
        unsqueeze_248: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_14, -1);  squeeze_14 = None
        mul_246: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(view_184, unsqueeze_248);  unsqueeze_248 = None
        sub_58: "f32[64, 32, 4]" = torch.ops.aten.sub.Tensor(view_183, mul_246);  view_183 = mul_246 = None
        mul_247: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_242);  sub_58 = unsqueeze_242 = None
        sum_78: "f32[32, 4]" = torch.ops.aten.sum.dim_IntList(mul_247, [0]);  mul_247 = None
        view_185: "f32[128]" = torch.ops.aten.reshape.default(sum_78, [128]);  sum_78 = None
        sum_79: "f32[128]" = torch.ops.aten.sum.dim_IntList(sum_75, [0])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(view_182, relu_4, primals_23, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_182 = primals_23 = None
        getitem_78: "f32[64, 64, 8, 8]" = convolution_backward_12[0]
        getitem_79: "f32[128, 64, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_248: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(where_10, convolution_6);  where_10 = None
        view_186: "f32[64, 128, 16]" = torch.ops.aten.reshape.default(mul_248, [64, 128, 16]);  mul_248 = None
        sum_80: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_186, [2]);  view_186 = None
        unsqueeze_36: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_21, 0)
        mul_249: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_80, unsqueeze_36)
        view_188: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_249, [64, 32, 4]);  mul_249 = None
        sum_82: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_188, [2]);  view_188 = None
        mul_250: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_75, unsqueeze_36);  unsqueeze_36 = None
        view_189: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_250, [64, 32, 4]);  mul_250 = None
        sum_83: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_189, [2]);  view_189 = None
        unsqueeze_252: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_13, -1)
        view_190: "f32[1, 32, 4]" = torch.ops.aten.reshape.default(primals_21, [1, 32, 4]);  primals_21 = None
        mul_251: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_252, view_190);  view_190 = None
        mul_252: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_12)
        sub_59: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_252, sum_82);  mul_252 = sum_82 = None
        mul_253: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_59, squeeze_13);  sub_59 = None
        mul_254: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_253, squeeze_13);  mul_253 = None
        mul_255: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_254, squeeze_13);  mul_254 = None
        mul_256: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_255, 0.015625);  mul_255 = None
        neg_13: "f32[64, 32]" = torch.ops.aten.neg.default(mul_256)
        mul_257: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_13, squeeze_12);  neg_13 = None
        mul_258: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_13);  sum_83 = squeeze_13 = None
        mul_259: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_258, 0.015625);  mul_258 = None
        sub_60: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_257, mul_259);  mul_257 = mul_259 = None
        unsqueeze_253: "f32[64, 32, 4, 1]" = torch.ops.aten.unsqueeze.default(mul_251, -1);  mul_251 = None
        unsqueeze_254: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_256, -1);  mul_256 = None
        unsqueeze_255: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        unsqueeze_256: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_60, -1);  sub_60 = None
        unsqueeze_257: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        mul_260: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_180, unsqueeze_253);  view_180 = unsqueeze_253 = None
        view_12: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(convolution_6, [64, 32, 4, 16]);  convolution_6 = None
        mul_261: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_12, unsqueeze_255);  view_12 = unsqueeze_255 = None
        add_79: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        add_80: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(add_79, unsqueeze_257);  add_79 = unsqueeze_257 = None
        view_193: "f32[64, 128, 4, 4]" = torch.ops.aten.reshape.default(add_80, [64, 128, 4, 4]);  add_80 = None
        view_194: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(sum_80, [64, 32, 4]);  sum_80 = None
        unsqueeze_258: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_12, -1);  squeeze_12 = None
        mul_262: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(view_184, unsqueeze_258);  view_184 = unsqueeze_258 = None
        sub_61: "f32[64, 32, 4]" = torch.ops.aten.sub.Tensor(view_194, mul_262);  view_194 = mul_262 = None
        mul_263: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_252);  sub_61 = unsqueeze_252 = None
        sum_84: "f32[32, 4]" = torch.ops.aten.sum.dim_IntList(mul_263, [0]);  mul_263 = None
        view_196: "f32[128]" = torch.ops.aten.reshape.default(sum_84, [128]);  sum_84 = None
        sum_85: "f32[128]" = torch.ops.aten.sum.dim_IntList(sum_75, [0]);  sum_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(view_193, relu_5, primals_20, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_193 = primals_20 = None
        getitem_81: "f32[64, 128, 4, 4]" = convolution_backward_13[0]
        getitem_82: "f32[128, 128, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_11: "b8[64, 128, 4, 4]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_11: "f32[64, 128, 4, 4]" = torch.ops.aten.where.self(le_11, full_default, getitem_81);  le_11 = getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_264: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(where_11, convolution_5)
        view_197: "f32[64, 128, 16]" = torch.ops.aten.reshape.default(mul_264, [64, 128, 16]);  mul_264 = None
        sum_86: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_197, [2]);  view_197 = None
        view_198: "f32[64, 128, 16]" = torch.ops.aten.reshape.default(where_11, [64, 128, 16])
        sum_87: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_198, [2]);  view_198 = None
        unsqueeze_30: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_18, 0)
        mul_265: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_86, unsqueeze_30)
        view_199: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_265, [64, 32, 4]);  mul_265 = None
        sum_88: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_199, [2]);  view_199 = None
        mul_266: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_87, unsqueeze_30);  unsqueeze_30 = None
        view_200: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_266, [64, 32, 4]);  mul_266 = None
        sum_89: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_200, [2]);  view_200 = None
        unsqueeze_262: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_11, -1)
        view_201: "f32[1, 32, 4]" = torch.ops.aten.reshape.default(primals_18, [1, 32, 4]);  primals_18 = None
        mul_267: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_262, view_201);  view_201 = None
        mul_268: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_10)
        sub_62: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_268, sum_88);  mul_268 = sum_88 = None
        mul_269: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_62, squeeze_11);  sub_62 = None
        mul_270: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_269, squeeze_11);  mul_269 = None
        mul_271: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_270, squeeze_11);  mul_270 = None
        mul_272: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_271, 0.015625);  mul_271 = None
        neg_14: "f32[64, 32]" = torch.ops.aten.neg.default(mul_272)
        mul_273: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_14, squeeze_10);  neg_14 = None
        mul_274: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_11);  sum_89 = squeeze_11 = None
        mul_275: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_274, 0.015625);  mul_274 = None
        sub_63: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_273, mul_275);  mul_273 = mul_275 = None
        unsqueeze_263: "f32[64, 32, 4, 1]" = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
        unsqueeze_264: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_272, -1);  mul_272 = None
        unsqueeze_265: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        unsqueeze_266: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_63, -1);  sub_63 = None
        unsqueeze_267: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        view_202: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(where_11, [64, 32, 4, 16]);  where_11 = None
        mul_276: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_202, unsqueeze_263);  view_202 = unsqueeze_263 = None
        view_10: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(convolution_5, [64, 32, 4, 16]);  convolution_5 = None
        mul_277: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_10, unsqueeze_265);  view_10 = unsqueeze_265 = None
        add_81: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
        add_82: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(add_81, unsqueeze_267);  add_81 = unsqueeze_267 = None
        view_204: "f32[64, 128, 4, 4]" = torch.ops.aten.reshape.default(add_82, [64, 128, 4, 4]);  add_82 = None
        view_205: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(sum_86, [64, 32, 4]);  sum_86 = None
        view_206: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(sum_87, [64, 32, 4])
        unsqueeze_268: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_10, -1);  squeeze_10 = None
        mul_278: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(view_206, unsqueeze_268);  view_206 = unsqueeze_268 = None
        sub_64: "f32[64, 32, 4]" = torch.ops.aten.sub.Tensor(view_205, mul_278);  view_205 = mul_278 = None
        mul_279: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_262);  sub_64 = unsqueeze_262 = None
        sum_90: "f32[32, 4]" = torch.ops.aten.sum.dim_IntList(mul_279, [0]);  mul_279 = None
        view_207: "f32[128]" = torch.ops.aten.reshape.default(sum_90, [128]);  sum_90 = None
        sum_91: "f32[128]" = torch.ops.aten.sum.dim_IntList(sum_87, [0]);  sum_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(view_204, relu_4, primals_17, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_204 = primals_17 = None
        getitem_84: "f32[64, 64, 8, 8]" = convolution_backward_14[0]
        getitem_85: "f32[128, 64, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
        add_83: "f32[64, 64, 8, 8]" = torch.ops.aten.add.Tensor(getitem_78, getitem_84);  getitem_78 = getitem_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_12: "b8[64, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_12: "f32[64, 64, 8, 8]" = torch.ops.aten.where.self(le_12, full_default, add_83);  le_12 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_280: "f32[64, 64, 8, 8]" = torch.ops.aten.mul.Tensor(where_12, convolution_4)
        view_208: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(mul_280, [64, 64, 64]);  mul_280 = None
        sum_92: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(view_208, [2]);  view_208 = None
        view_209: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(where_12, [64, 64, 64])
        sum_93: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(view_209, [2]);  view_209 = None
        unsqueeze_24: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_15, 0)
        mul_281: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_92, unsqueeze_24)
        view_210: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_281, [64, 32, 2]);  mul_281 = None
        sum_94: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_210, [2]);  view_210 = None
        mul_282: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_93, unsqueeze_24);  unsqueeze_24 = None
        view_211: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_282, [64, 32, 2]);  mul_282 = None
        sum_95: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_211, [2]);  view_211 = None
        unsqueeze_272: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_9, -1)
        view_212: "f32[1, 32, 2]" = torch.ops.aten.reshape.default(primals_15, [1, 32, 2]);  primals_15 = None
        mul_283: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(unsqueeze_272, view_212);  view_212 = None
        mul_284: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_8)
        sub_65: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_284, sum_94);  mul_284 = sum_94 = None
        mul_285: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_65, squeeze_9);  sub_65 = None
        mul_286: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_285, squeeze_9);  mul_285 = None
        mul_287: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_286, squeeze_9);  mul_286 = None
        mul_288: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_287, 0.0078125);  mul_287 = None
        neg_15: "f32[64, 32]" = torch.ops.aten.neg.default(mul_288)
        mul_289: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_15, squeeze_8);  neg_15 = None
        mul_290: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_9);  sum_95 = squeeze_9 = None
        mul_291: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_290, 0.0078125);  mul_290 = None
        sub_66: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_289, mul_291);  mul_289 = mul_291 = None
        unsqueeze_273: "f32[64, 32, 2, 1]" = torch.ops.aten.unsqueeze.default(mul_283, -1);  mul_283 = None
        unsqueeze_274: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_275: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        unsqueeze_276: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_66, -1);  sub_66 = None
        unsqueeze_277: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        view_213: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(where_12, [64, 32, 2, 64])
        mul_292: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(view_213, unsqueeze_273);  view_213 = unsqueeze_273 = None
        view_8: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(convolution_4, [64, 32, 2, 64]);  convolution_4 = None
        mul_293: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(view_8, unsqueeze_275);  view_8 = unsqueeze_275 = None
        add_84: "f32[64, 32, 2, 64]" = torch.ops.aten.add.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
        add_85: "f32[64, 32, 2, 64]" = torch.ops.aten.add.Tensor(add_84, unsqueeze_277);  add_84 = unsqueeze_277 = None
        view_215: "f32[64, 64, 8, 8]" = torch.ops.aten.reshape.default(add_85, [64, 64, 8, 8]);  add_85 = None
        view_216: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(sum_92, [64, 32, 2]);  sum_92 = None
        view_217: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(sum_93, [64, 32, 2])
        unsqueeze_278: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_8, -1);  squeeze_8 = None
        mul_294: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(view_217, unsqueeze_278);  view_217 = unsqueeze_278 = None
        sub_67: "f32[64, 32, 2]" = torch.ops.aten.sub.Tensor(view_216, mul_294);  view_216 = mul_294 = None
        mul_295: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_272);  sub_67 = unsqueeze_272 = None
        sum_96: "f32[32, 2]" = torch.ops.aten.sum.dim_IntList(mul_295, [0]);  mul_295 = None
        view_218: "f32[64]" = torch.ops.aten.reshape.default(sum_96, [64]);  sum_96 = None
        sum_97: "f32[64]" = torch.ops.aten.sum.dim_IntList(sum_93, [0]);  sum_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(view_215, relu_3, primals_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_215 = primals_14 = None
        getitem_87: "f32[64, 64, 8, 8]" = convolution_backward_15[0]
        getitem_88: "f32[64, 64, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_13: "b8[64, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_13: "f32[64, 64, 8, 8]" = torch.ops.aten.where.self(le_13, full_default, getitem_87);  le_13 = getitem_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_296: "f32[64, 64, 8, 8]" = torch.ops.aten.mul.Tensor(where_13, convolution_3)
        view_219: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(mul_296, [64, 64, 64]);  mul_296 = None
        sum_98: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(view_219, [2]);  view_219 = None
        view_220: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(where_13, [64, 64, 64])
        sum_99: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(view_220, [2]);  view_220 = None
        unsqueeze_18: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_12, 0)
        mul_297: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_98, unsqueeze_18)
        view_221: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_297, [64, 32, 2]);  mul_297 = None
        sum_100: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_221, [2]);  view_221 = None
        mul_298: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_99, unsqueeze_18);  unsqueeze_18 = None
        view_222: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_298, [64, 32, 2]);  mul_298 = None
        sum_101: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_222, [2]);  view_222 = None
        unsqueeze_282: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_7, -1)
        view_223: "f32[1, 32, 2]" = torch.ops.aten.reshape.default(primals_12, [1, 32, 2]);  primals_12 = None
        mul_299: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(unsqueeze_282, view_223);  view_223 = None
        mul_300: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_6)
        sub_68: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_300, sum_100);  mul_300 = sum_100 = None
        mul_301: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_68, squeeze_7);  sub_68 = None
        mul_302: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_301, squeeze_7);  mul_301 = None
        mul_303: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_302, squeeze_7);  mul_302 = None
        mul_304: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_303, 0.0078125);  mul_303 = None
        neg_16: "f32[64, 32]" = torch.ops.aten.neg.default(mul_304)
        mul_305: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_16, squeeze_6);  neg_16 = None
        mul_306: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_7);  sum_101 = squeeze_7 = None
        mul_307: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_306, 0.0078125);  mul_306 = None
        sub_69: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_305, mul_307);  mul_305 = mul_307 = None
        unsqueeze_283: "f32[64, 32, 2, 1]" = torch.ops.aten.unsqueeze.default(mul_299, -1);  mul_299 = None
        unsqueeze_284: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_304, -1);  mul_304 = None
        unsqueeze_285: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        unsqueeze_286: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_69, -1);  sub_69 = None
        unsqueeze_287: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        view_224: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(where_13, [64, 32, 2, 64]);  where_13 = None
        mul_308: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(view_224, unsqueeze_283);  view_224 = unsqueeze_283 = None
        view_6: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(convolution_3, [64, 32, 2, 64]);  convolution_3 = None
        mul_309: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(view_6, unsqueeze_285);  view_6 = unsqueeze_285 = None
        add_86: "f32[64, 32, 2, 64]" = torch.ops.aten.add.Tensor(mul_308, mul_309);  mul_308 = mul_309 = None
        add_87: "f32[64, 32, 2, 64]" = torch.ops.aten.add.Tensor(add_86, unsqueeze_287);  add_86 = unsqueeze_287 = None
        view_226: "f32[64, 64, 8, 8]" = torch.ops.aten.reshape.default(add_87, [64, 64, 8, 8]);  add_87 = None
        view_227: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(sum_98, [64, 32, 2]);  sum_98 = None
        view_228: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(sum_99, [64, 32, 2])
        unsqueeze_288: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_6, -1);  squeeze_6 = None
        mul_310: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(view_228, unsqueeze_288);  view_228 = unsqueeze_288 = None
        sub_70: "f32[64, 32, 2]" = torch.ops.aten.sub.Tensor(view_227, mul_310);  view_227 = mul_310 = None
        mul_311: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_282);  sub_70 = unsqueeze_282 = None
        sum_102: "f32[32, 2]" = torch.ops.aten.sum.dim_IntList(mul_311, [0]);  mul_311 = None
        view_229: "f32[64]" = torch.ops.aten.reshape.default(sum_102, [64]);  sum_102 = None
        sum_103: "f32[64]" = torch.ops.aten.sum.dim_IntList(sum_99, [0]);  sum_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(view_226, relu_2, primals_11, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_226 = primals_11 = None
        getitem_90: "f32[64, 64, 8, 8]" = convolution_backward_16[0]
        getitem_91: "f32[64, 64, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
        add_88: "f32[64, 64, 8, 8]" = torch.ops.aten.add.Tensor(where_12, getitem_90);  where_12 = getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_14: "b8[64, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_14: "f32[64, 64, 8, 8]" = torch.ops.aten.where.self(le_14, full_default, add_88);  le_14 = add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_312: "f32[64, 64, 8, 8]" = torch.ops.aten.mul.Tensor(where_14, convolution_2)
        view_230: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(mul_312, [64, 64, 64]);  mul_312 = None
        sum_104: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(view_230, [2]);  view_230 = None
        view_231: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(where_14, [64, 64, 64])
        sum_105: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(view_231, [2]);  view_231 = None
        unsqueeze_12: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_9, 0)
        mul_313: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_104, unsqueeze_12)
        view_232: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_313, [64, 32, 2]);  mul_313 = None
        sum_106: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_232, [2]);  view_232 = None
        mul_314: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_105, unsqueeze_12);  unsqueeze_12 = None
        view_233: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_314, [64, 32, 2]);  mul_314 = None
        sum_107: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_233, [2]);  view_233 = None
        unsqueeze_292: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_5, -1)
        view_234: "f32[1, 32, 2]" = torch.ops.aten.reshape.default(primals_9, [1, 32, 2]);  primals_9 = None
        mul_315: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(unsqueeze_292, view_234);  view_234 = None
        mul_316: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_4)
        sub_71: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_316, sum_106);  mul_316 = sum_106 = None
        mul_317: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_71, squeeze_5);  sub_71 = None
        mul_318: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_317, squeeze_5);  mul_317 = None
        mul_319: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_318, squeeze_5);  mul_318 = None
        mul_320: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_319, 0.0078125);  mul_319 = None
        neg_17: "f32[64, 32]" = torch.ops.aten.neg.default(mul_320)
        mul_321: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_17, squeeze_4);  neg_17 = None
        mul_322: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_5);  sum_107 = squeeze_5 = None
        mul_323: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_322, 0.0078125);  mul_322 = None
        sub_72: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_321, mul_323);  mul_321 = mul_323 = None
        unsqueeze_293: "f32[64, 32, 2, 1]" = torch.ops.aten.unsqueeze.default(mul_315, -1);  mul_315 = None
        unsqueeze_294: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_320, -1);  mul_320 = None
        unsqueeze_295: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        unsqueeze_296: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_72, -1);  sub_72 = None
        unsqueeze_297: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        view_235: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(where_14, [64, 32, 2, 64])
        mul_324: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(view_235, unsqueeze_293);  view_235 = unsqueeze_293 = None
        view_4: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(convolution_2, [64, 32, 2, 64]);  convolution_2 = None
        mul_325: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(view_4, unsqueeze_295);  view_4 = unsqueeze_295 = None
        add_89: "f32[64, 32, 2, 64]" = torch.ops.aten.add.Tensor(mul_324, mul_325);  mul_324 = mul_325 = None
        add_90: "f32[64, 32, 2, 64]" = torch.ops.aten.add.Tensor(add_89, unsqueeze_297);  add_89 = unsqueeze_297 = None
        view_237: "f32[64, 64, 8, 8]" = torch.ops.aten.reshape.default(add_90, [64, 64, 8, 8]);  add_90 = None
        view_238: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(sum_104, [64, 32, 2]);  sum_104 = None
        view_239: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(sum_105, [64, 32, 2])
        unsqueeze_298: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_4, -1);  squeeze_4 = None
        mul_326: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(view_239, unsqueeze_298);  view_239 = unsqueeze_298 = None
        sub_73: "f32[64, 32, 2]" = torch.ops.aten.sub.Tensor(view_238, mul_326);  view_238 = mul_326 = None
        mul_327: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_292);  sub_73 = unsqueeze_292 = None
        sum_108: "f32[32, 2]" = torch.ops.aten.sum.dim_IntList(mul_327, [0]);  mul_327 = None
        view_240: "f32[64]" = torch.ops.aten.reshape.default(sum_108, [64]);  sum_108 = None
        sum_109: "f32[64]" = torch.ops.aten.sum.dim_IntList(sum_105, [0]);  sum_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_237, relu_1, primals_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_237 = primals_8 = None
        getitem_93: "f32[64, 64, 8, 8]" = convolution_backward_17[0]
        getitem_94: "f32[64, 64, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_15: "b8[64, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_15: "f32[64, 64, 8, 8]" = torch.ops.aten.where.self(le_15, full_default, getitem_93);  le_15 = getitem_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_328: "f32[64, 64, 8, 8]" = torch.ops.aten.mul.Tensor(where_15, convolution_1)
        view_241: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(mul_328, [64, 64, 64]);  mul_328 = None
        sum_110: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(view_241, [2]);  view_241 = None
        view_242: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(where_15, [64, 64, 64])
        sum_111: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(view_242, [2]);  view_242 = None
        unsqueeze_6: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_6, 0)
        mul_329: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_110, unsqueeze_6)
        view_243: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_329, [64, 32, 2]);  mul_329 = None
        sum_112: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_243, [2]);  view_243 = None
        mul_330: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_111, unsqueeze_6);  unsqueeze_6 = None
        view_244: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_330, [64, 32, 2]);  mul_330 = None
        sum_113: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_244, [2]);  view_244 = None
        unsqueeze_302: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_3, -1)
        view_245: "f32[1, 32, 2]" = torch.ops.aten.reshape.default(primals_6, [1, 32, 2]);  primals_6 = None
        mul_331: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(unsqueeze_302, view_245);  view_245 = None
        mul_332: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_2)
        sub_74: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_332, sum_112);  mul_332 = sum_112 = None
        mul_333: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_74, squeeze_3);  sub_74 = None
        mul_334: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_333, squeeze_3);  mul_333 = None
        mul_335: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_334, squeeze_3);  mul_334 = None
        mul_336: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_335, 0.0078125);  mul_335 = None
        neg_18: "f32[64, 32]" = torch.ops.aten.neg.default(mul_336)
        mul_337: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_18, squeeze_2);  neg_18 = None
        mul_338: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_3);  sum_113 = squeeze_3 = None
        mul_339: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_338, 0.0078125);  mul_338 = None
        sub_75: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_337, mul_339);  mul_337 = mul_339 = None
        unsqueeze_303: "f32[64, 32, 2, 1]" = torch.ops.aten.unsqueeze.default(mul_331, -1);  mul_331 = None
        unsqueeze_304: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_305: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_75, -1);  sub_75 = None
        unsqueeze_307: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        view_246: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(where_15, [64, 32, 2, 64]);  where_15 = None
        mul_340: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(view_246, unsqueeze_303);  view_246 = unsqueeze_303 = None
        view_2: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(convolution_1, [64, 32, 2, 64]);  convolution_1 = None
        mul_341: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(view_2, unsqueeze_305);  view_2 = unsqueeze_305 = None
        add_91: "f32[64, 32, 2, 64]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        add_92: "f32[64, 32, 2, 64]" = torch.ops.aten.add.Tensor(add_91, unsqueeze_307);  add_91 = unsqueeze_307 = None
        view_248: "f32[64, 64, 8, 8]" = torch.ops.aten.reshape.default(add_92, [64, 64, 8, 8]);  add_92 = None
        view_249: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(sum_110, [64, 32, 2]);  sum_110 = None
        view_250: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(sum_111, [64, 32, 2])
        unsqueeze_308: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_2, -1);  squeeze_2 = None
        mul_342: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(view_250, unsqueeze_308);  view_250 = unsqueeze_308 = None
        sub_76: "f32[64, 32, 2]" = torch.ops.aten.sub.Tensor(view_249, mul_342);  view_249 = mul_342 = None
        mul_343: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_302);  sub_76 = unsqueeze_302 = None
        sum_114: "f32[32, 2]" = torch.ops.aten.sum.dim_IntList(mul_343, [0]);  mul_343 = None
        view_251: "f32[64]" = torch.ops.aten.reshape.default(sum_114, [64]);  sum_114 = None
        sum_115: "f32[64]" = torch.ops.aten.sum.dim_IntList(sum_111, [0]);  sum_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(view_248, getitem_2, primals_5, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  view_248 = getitem_2 = primals_5 = None
        getitem_96: "f32[64, 64, 8, 8]" = convolution_backward_18[0]
        getitem_97: "f32[64, 64, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
        add_93: "f32[64, 64, 8, 8]" = torch.ops.aten.add.Tensor(where_14, getitem_96);  where_14 = getitem_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        full_default_16: "f32[4096, 256]" = torch.ops.aten.full.default([4096, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_252: "f32[4096, 64]" = torch.ops.aten.reshape.default(add_93, [4096, 64]);  add_93 = None
        _low_memory_max_pool_offsets_to_indices: "i64[64, 64, 8, 8]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [16, 16], [2, 2], [1, 1], [1, 1]);  getitem_3 = None
        view_253: "i64[4096, 64]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [4096, 64]);  _low_memory_max_pool_offsets_to_indices = None
        scatter_add: "f32[4096, 256]" = torch.ops.aten.scatter_add.default(full_default_16, 1, view_253, view_252);  full_default_16 = view_253 = view_252 = None
        view_254: "f32[64, 64, 16, 16]" = torch.ops.aten.reshape.default(scatter_add, [64, 64, 16, 16]);  scatter_add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        view: "f32[64, 32, 2, 256]" = torch.ops.aten.reshape.default(convolution, [64, 32, 2, 256])
        sub: "f32[64, 32, 2, 256]" = torch.ops.aten.sub.Tensor(view, getitem_1)
        mul: "f32[64, 32, 2, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        view_1: "f32[64, 64, 16, 16]" = torch.ops.aten.reshape.default(mul, [64, 64, 16, 16]);  mul = None
        unsqueeze: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_3, 0)
        unsqueeze_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
        unsqueeze_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1: "f32[64, 64, 16, 16]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_2);  view_1 = unsqueeze_2 = None
        unsqueeze_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_4, 0);  primals_4 = None
        unsqueeze_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1: "f32[64, 64, 16, 16]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu: "f32[64, 64, 16, 16]" = torch.ops.aten.relu.default(add_1);  add_1 = None
        le_16: "b8[64, 64, 16, 16]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_16: "f32[64, 64, 16, 16]" = torch.ops.aten.where.self(le_16, full_default, view_254);  le_16 = full_default = view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        mul_344: "f32[64, 64, 16, 16]" = torch.ops.aten.mul.Tensor(where_16, convolution);  convolution = None
        view_255: "f32[64, 64, 256]" = torch.ops.aten.reshape.default(mul_344, [64, 64, 256]);  mul_344 = None
        sum_116: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(view_255, [2]);  view_255 = None
        view_256: "f32[64, 64, 256]" = torch.ops.aten.reshape.default(where_16, [64, 64, 256])
        sum_117: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(view_256, [2]);  view_256 = None
        mul_345: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_116, unsqueeze)
        view_257: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_345, [64, 32, 2]);  mul_345 = None
        sum_118: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_257, [2]);  view_257 = None
        mul_346: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_117, unsqueeze);  unsqueeze = None
        view_258: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_346, [64, 32, 2]);  mul_346 = None
        sum_119: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_258, [2]);  view_258 = None
        squeeze_1: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt, [2, 3]);  rsqrt = None
        unsqueeze_312: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_1, -1)
        view_259: "f32[1, 32, 2]" = torch.ops.aten.reshape.default(primals_3, [1, 32, 2]);  primals_3 = None
        mul_347: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(unsqueeze_312, view_259);  view_259 = None
        squeeze: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_1, [2, 3]);  getitem_1 = None
        mul_348: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_119, squeeze)
        sub_77: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_348, sum_118);  mul_348 = sum_118 = None
        mul_349: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_77, squeeze_1);  sub_77 = None
        mul_350: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_349, squeeze_1);  mul_349 = None
        mul_351: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_350, squeeze_1);  mul_350 = None
        mul_352: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_351, 0.001953125);  mul_351 = None
        neg_19: "f32[64, 32]" = torch.ops.aten.neg.default(mul_352)
        mul_353: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_19, squeeze);  neg_19 = None
        mul_354: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_1);  sum_119 = squeeze_1 = None
        mul_355: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_354, 0.001953125);  mul_354 = None
        sub_78: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_353, mul_355);  mul_353 = mul_355 = None
        unsqueeze_313: "f32[64, 32, 2, 1]" = torch.ops.aten.unsqueeze.default(mul_347, -1);  mul_347 = None
        unsqueeze_314: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_352, -1);  mul_352 = None
        unsqueeze_315: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        unsqueeze_316: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_78, -1);  sub_78 = None
        unsqueeze_317: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        view_260: "f32[64, 32, 2, 256]" = torch.ops.aten.reshape.default(where_16, [64, 32, 2, 256]);  where_16 = None
        mul_356: "f32[64, 32, 2, 256]" = torch.ops.aten.mul.Tensor(view_260, unsqueeze_313);  view_260 = unsqueeze_313 = None
        mul_357: "f32[64, 32, 2, 256]" = torch.ops.aten.mul.Tensor(view, unsqueeze_315);  view = unsqueeze_315 = None
        add_94: "f32[64, 32, 2, 256]" = torch.ops.aten.add.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
        add_95: "f32[64, 32, 2, 256]" = torch.ops.aten.add.Tensor(add_94, unsqueeze_317);  add_94 = unsqueeze_317 = None
        view_262: "f32[64, 64, 16, 16]" = torch.ops.aten.reshape.default(add_95, [64, 64, 16, 16]);  add_95 = None
        view_263: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(sum_116, [64, 32, 2]);  sum_116 = None
        view_264: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(sum_117, [64, 32, 2])
        unsqueeze_318: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze, -1);  squeeze = None
        mul_358: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(view_264, unsqueeze_318);  view_264 = unsqueeze_318 = None
        sub_79: "f32[64, 32, 2]" = torch.ops.aten.sub.Tensor(view_263, mul_358);  view_263 = mul_358 = None
        mul_359: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_312);  sub_79 = unsqueeze_312 = None
        sum_120: "f32[32, 2]" = torch.ops.aten.sum.dim_IntList(mul_359, [0]);  mul_359 = None
        view_265: "f32[64]" = torch.ops.aten.reshape.default(sum_120, [64]);  sum_120 = None
        sum_121: "f32[64]" = torch.ops.aten.sum.dim_IntList(sum_117, [0]);  sum_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:268 in _forward_impl, code: x = self.conv1(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(view_262, primals_2, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  view_262 = primals_2 = primals_1 = None
        getitem_100: "f32[64, 3, 7, 7]" = convolution_backward_19[1];  convolution_backward_19 = None
        return (getitem_100, None, view_265, sum_121, getitem_97, view_251, sum_115, getitem_94, view_240, sum_109, getitem_91, view_229, sum_103, getitem_88, view_218, sum_97, getitem_85, view_207, sum_91, getitem_82, view_196, sum_85, getitem_79, view_185, sum_79, getitem_76, view_174, sum_73, getitem_73, view_163, sum_67, getitem_70, view_152, sum_61, getitem_67, view_141, sum_55, getitem_64, view_130, sum_49, getitem_61, view_119, sum_43, getitem_58, view_108, sum_37, getitem_55, view_97, sum_31, getitem_52, view_86, sum_25, getitem_49, view_75, sum_19, getitem_46, view_64, sum_13, getitem_43, view_53, sum_7, mm_1, view_41)
