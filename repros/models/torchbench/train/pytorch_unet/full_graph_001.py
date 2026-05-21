class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_3: "f32[8, 3, 640, 959]", primals_7: "f32[64]", primals_9: "f32[64, 64, 3, 3]", primals_14: "f32[64]", primals_15: "f32[64]", primals_16: "f32[128, 64, 3, 3]", primals_21: "f32[128]", primals_23: "f32[128, 128, 3, 3]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[256, 128, 3, 3]", primals_35: "f32[256]", primals_37: "f32[256, 256, 3, 3]", primals_42: "f32[256]", primals_43: "f32[256]", primals_44: "f32[512, 256, 3, 3]", primals_49: "f32[512]", primals_51: "f32[512, 512, 3, 3]", primals_56: "f32[512]", primals_57: "f32[512]", primals_58: "f32[512, 512, 3, 3]", primals_63: "f32[512]", primals_65: "f32[512, 512, 3, 3]", primals_70: "f32[512]", primals_71: "f32[512]", primals_72: "f32[512, 1024, 3, 3]", primals_77: "f32[512]", primals_79: "f32[256, 512, 3, 3]", primals_84: "f32[256]", primals_85: "f32[256]", primals_86: "f32[256, 512, 3, 3]", primals_91: "f32[256]", primals_93: "f32[128, 256, 3, 3]", primals_98: "f32[128]", primals_99: "f32[128]", primals_100: "f32[128, 256, 3, 3]", primals_105: "f32[128]", primals_107: "f32[64, 128, 3, 3]", primals_112: "f32[64]", primals_113: "f32[64]", primals_114: "f32[64, 128, 3, 3]", primals_119: "f32[64]", primals_121: "f32[64, 64, 3, 3]", primals_126: "f32[64]", primals_128: "f32[2, 64, 1, 1]", convolution: "f32[8, 64, 640, 959]", squeeze_1: "f32[64]", relu: "f32[8, 64, 640, 959]", convolution_1: "f32[8, 64, 640, 959]", getitem_3: "f32[1, 64, 1, 1]", rsqrt_1: "f32[1, 64, 1, 1]", getitem_4: "f32[8, 64, 320, 479]", getitem_5: "i8[8, 64, 320, 479]", convolution_2: "f32[8, 128, 320, 479]", squeeze_7: "f32[128]", relu_2: "f32[8, 128, 320, 479]", convolution_3: "f32[8, 128, 320, 479]", getitem_9: "f32[1, 128, 1, 1]", rsqrt_3: "f32[1, 128, 1, 1]", getitem_10: "f32[8, 128, 160, 239]", getitem_11: "i8[8, 128, 160, 239]", convolution_4: "f32[8, 256, 160, 239]", squeeze_13: "f32[256]", relu_4: "f32[8, 256, 160, 239]", convolution_5: "f32[8, 256, 160, 239]", getitem_15: "f32[1, 256, 1, 1]", rsqrt_5: "f32[1, 256, 1, 1]", getitem_16: "f32[8, 256, 80, 119]", getitem_17: "i8[8, 256, 80, 119]", convolution_6: "f32[8, 512, 80, 119]", squeeze_19: "f32[512]", relu_6: "f32[8, 512, 80, 119]", convolution_7: "f32[8, 512, 80, 119]", getitem_21: "f32[1, 512, 1, 1]", rsqrt_7: "f32[1, 512, 1, 1]", getitem_22: "f32[8, 512, 40, 59]", getitem_23: "i8[8, 512, 40, 59]", convolution_8: "f32[8, 512, 40, 59]", squeeze_25: "f32[512]", relu_8: "f32[8, 512, 40, 59]", convolution_9: "f32[8, 512, 40, 59]", getitem_27: "f32[1, 512, 1, 1]", rsqrt_9: "f32[1, 512, 1, 1]", convert_element_type_1: "i64[80, 1]", clamp_max: "i64[80, 1]", convert_element_type_3: "i64[118]", clamp_max_1: "i64[118]", clamp_max_2: "f32[118]", clamp_max_3: "f32[80, 1]", cat: "f32[8, 1024, 80, 119]", convolution_10: "f32[8, 512, 80, 119]", squeeze_31: "f32[512]", relu_10: "f32[8, 512, 80, 119]", convolution_11: "f32[8, 256, 80, 119]", getitem_31: "f32[1, 256, 1, 1]", rsqrt_11: "f32[1, 256, 1, 1]", convert_element_type_5: "i64[160, 1]", clamp_max_4: "i64[160, 1]", convert_element_type_7: "i64[238]", clamp_max_5: "i64[238]", clamp_max_6: "f32[238]", clamp_max_7: "f32[160, 1]", cat_1: "f32[8, 512, 160, 239]", convolution_12: "f32[8, 256, 160, 239]", squeeze_37: "f32[256]", relu_12: "f32[8, 256, 160, 239]", convolution_13: "f32[8, 128, 160, 239]", getitem_35: "f32[1, 128, 1, 1]", rsqrt_13: "f32[1, 128, 1, 1]", convert_element_type_9: "i64[320, 1]", clamp_max_8: "i64[320, 1]", convert_element_type_11: "i64[478]", clamp_max_9: "i64[478]", clamp_max_10: "f32[478]", clamp_max_11: "f32[320, 1]", cat_2: "f32[8, 256, 320, 479]", convolution_14: "f32[8, 128, 320, 479]", squeeze_43: "f32[128]", relu_14: "f32[8, 128, 320, 479]", convolution_15: "f32[8, 64, 320, 479]", getitem_39: "f32[1, 64, 1, 1]", rsqrt_15: "f32[1, 64, 1, 1]", convert_element_type_13: "i64[640, 1]", clamp_max_12: "i64[640, 1]", convert_element_type_15: "i64[958]", clamp_max_13: "i64[958]", clamp_max_14: "f32[958]", clamp_max_15: "f32[640, 1]", cat_3: "f32[8, 128, 640, 959]", convolution_16: "f32[8, 64, 640, 959]", squeeze_49: "f32[64]", relu_16: "f32[8, 64, 640, 959]", convolution_17: "f32[8, 64, 640, 959]", squeeze_52: "f32[64]", relu_17: "f32[8, 64, 640, 959]", unsqueeze_74: "f32[1, 64, 1, 1]", unsqueeze_86: "f32[1, 64, 1, 1]", unsqueeze_110: "f32[1, 128, 1, 1]", unsqueeze_134: "f32[1, 256, 1, 1]", unsqueeze_158: "f32[1, 512, 1, 1]", unsqueeze_182: "f32[1, 512, 1, 1]", unsqueeze_206: "f32[1, 512, 1, 1]", unsqueeze_230: "f32[1, 256, 1, 1]", unsqueeze_254: "f32[1, 128, 1, 1]", unsqueeze_278: "f32[1, 64, 1, 1]", tangents_1: "f32[8, 2, 640, 959]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:77 in forward, code: return self.conv(x)
        sum_1: "f32[2]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(tangents_1, relu_17, primals_128, [2], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  tangents_1 = primals_128 = None
        getitem_44: "f32[8, 64, 640, 959]" = convolution_backward[0]
        getitem_45: "f32[2, 64, 1, 1]" = convolution_backward[1];  convolution_backward = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        le: "b8[8, 64, 640, 959]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 64, 640, 959]" = torch.ops.aten.where.self(le, full_default, getitem_44);  le = getitem_44 = None
        sum_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_38: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_74);  convolution_17 = unsqueeze_74 = None
        mul_146: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where, sub_38)
        sum_3: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_146, [0, 2, 3]);  mul_146 = None
        mul_147: "f32[64]" = torch.ops.aten.mul.Tensor(sum_2, 2.0366266944734097e-07)
        unsqueeze_75: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_147, 0);  mul_147 = None
        unsqueeze_76: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
        unsqueeze_77: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 3);  unsqueeze_76 = None
        mul_148: "f32[64]" = torch.ops.aten.mul.Tensor(sum_3, 2.0366266944734097e-07)
        mul_149: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_150: "f32[64]" = torch.ops.aten.mul.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        unsqueeze_78: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_150, 0);  mul_150 = None
        unsqueeze_79: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 2);  unsqueeze_78 = None
        unsqueeze_80: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 3);  unsqueeze_79 = None
        mul_151: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_126);  primals_126 = None
        unsqueeze_81: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_151, 0);  mul_151 = None
        unsqueeze_82: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
        unsqueeze_83: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 3);  unsqueeze_82 = None
        mul_152: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_80);  sub_38 = unsqueeze_80 = None
        sub_40: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(where, mul_152);  where = mul_152 = None
        sub_41: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(sub_40, unsqueeze_77);  sub_40 = unsqueeze_77 = None
        mul_153: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_83);  sub_41 = unsqueeze_83 = None
        mul_154: "f32[64]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_52);  sum_3 = squeeze_52 = None
        sum_4: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_153, relu_16, primals_121, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_153 = primals_121 = None
        getitem_47: "f32[8, 64, 640, 959]" = convolution_backward_1[0]
        getitem_48: "f32[64, 64, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
        le_1: "b8[8, 64, 640, 959]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_1: "f32[8, 64, 640, 959]" = torch.ops.aten.where.self(le_1, full_default, getitem_47);  le_1 = getitem_47 = None
        sum_5: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_42: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_86);  convolution_16 = unsqueeze_86 = None
        mul_155: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where_1, sub_42)
        sum_6: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 2, 3]);  mul_155 = None
        mul_156: "f32[64]" = torch.ops.aten.mul.Tensor(sum_5, 2.0366266944734097e-07)
        unsqueeze_87: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_156, 0);  mul_156 = None
        unsqueeze_88: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
        unsqueeze_89: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 3);  unsqueeze_88 = None
        mul_157: "f32[64]" = torch.ops.aten.mul.Tensor(sum_6, 2.0366266944734097e-07)
        mul_158: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_159: "f32[64]" = torch.ops.aten.mul.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
        unsqueeze_90: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_159, 0);  mul_159 = None
        unsqueeze_91: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 2);  unsqueeze_90 = None
        unsqueeze_92: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 3);  unsqueeze_91 = None
        mul_160: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_119);  primals_119 = None
        unsqueeze_93: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_160, 0);  mul_160 = None
        unsqueeze_94: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
        unsqueeze_95: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, 3);  unsqueeze_94 = None
        mul_161: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_92);  sub_42 = unsqueeze_92 = None
        sub_44: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(where_1, mul_161);  where_1 = mul_161 = None
        sub_45: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(sub_44, unsqueeze_89);  sub_44 = unsqueeze_89 = None
        mul_162: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_95);  sub_45 = unsqueeze_95 = None
        mul_163: "f32[64]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_49);  sum_6 = squeeze_49 = None
        sum_7: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_162, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_162, cat_3, primals_114, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_162 = cat_3 = primals_114 = None
        getitem_50: "f32[8, 128, 640, 959]" = convolution_backward_2[0]
        getitem_51: "f32[64, 128, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        slice_1: "f32[8, 64, 640, 959]" = torch.ops.aten.slice.Tensor(getitem_50, 1, 0, 64)
        slice_2: "f32[8, 64, 640, 959]" = torch.ops.aten.slice.Tensor(getitem_50, 1, 64, 128);  getitem_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_4: "f32[8, 64, 640, 958]" = torch.ops.aten.constant_pad_nd.default(slice_2, [0, -1, 0, 0]);  slice_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        mul_164: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(constant_pad_nd_4, clamp_max_15);  clamp_max_15 = None
        neg: "f32[8, 64, 640, 958]" = torch.ops.aten.neg.default(mul_164)
        add_110: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(constant_pad_nd_4, neg);  constant_pad_nd_4 = neg = None
        mul_165: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(mul_164, clamp_max_14)
        neg_1: "f32[8, 64, 640, 958]" = torch.ops.aten.neg.default(mul_165)
        add_111: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(mul_164, neg_1);  mul_164 = neg_1 = None
        mul_166: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(add_110, clamp_max_14);  clamp_max_14 = None
        neg_2: "f32[8, 64, 640, 958]" = torch.ops.aten.neg.default(mul_166)
        add_112: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(add_110, neg_2);  add_110 = neg_2 = None
        full_default_2: "f32[8, 64, 320, 479]" = torch.ops.aten.full.default([8, 64, 320, 479], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[8, 64, 320, 479]" = torch.ops.aten.index_put.default(full_default_2, [None, None, clamp_max_12, clamp_max_13], mul_165, True);  mul_165 = None
        index_put_1: "f32[8, 64, 320, 479]" = torch.ops.aten.index_put.default(full_default_2, [None, None, clamp_max_12, convert_element_type_15], add_111, True);  clamp_max_12 = add_111 = None
        add_113: "f32[8, 64, 320, 479]" = torch.ops.aten.add.Tensor(index_put, index_put_1);  index_put = index_put_1 = None
        index_put_2: "f32[8, 64, 320, 479]" = torch.ops.aten.index_put.default(full_default_2, [None, None, convert_element_type_13, clamp_max_13], mul_166, True);  clamp_max_13 = mul_166 = None
        add_114: "f32[8, 64, 320, 479]" = torch.ops.aten.add.Tensor(add_113, index_put_2);  add_113 = index_put_2 = None
        index_put_3: "f32[8, 64, 320, 479]" = torch.ops.aten.index_put.default(full_default_2, [None, None, convert_element_type_13, convert_element_type_15], add_112, True);  full_default_2 = convert_element_type_13 = convert_element_type_15 = add_112 = None
        add_115: "f32[8, 64, 320, 479]" = torch.ops.aten.add.Tensor(add_114, index_put_3);  add_114 = index_put_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_30: "f32[8, 64, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_39)
        mul_120: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_15);  sub_30 = None
        unsqueeze_60: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1)
        unsqueeze_61: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_126: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_61);  mul_120 = unsqueeze_61 = None
        unsqueeze_62: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1);  primals_113 = None
        unsqueeze_63: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_94: "f32[8, 64, 320, 479]" = torch.ops.aten.add.Tensor(mul_126, unsqueeze_63);  mul_126 = unsqueeze_63 = None
        relu_15: "f32[8, 64, 320, 479]" = torch.ops.aten.relu.default(add_94);  add_94 = None
        le_2: "b8[8, 64, 320, 479]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_2: "f32[8, 64, 320, 479]" = torch.ops.aten.where.self(le_2, full_default, add_115);  le_2 = add_115 = None
        squeeze_45: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        unsqueeze_96: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_97: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 2);  unsqueeze_96 = None
        unsqueeze_98: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 3);  unsqueeze_97 = None
        sum_8: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_46: "f32[8, 64, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_98);  convolution_15 = unsqueeze_98 = None
        mul_167: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(where_2, sub_46)
        sum_9: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_167, [0, 2, 3]);  mul_167 = None
        mul_168: "f32[64]" = torch.ops.aten.mul.Tensor(sum_8, 8.155010438413362e-07)
        unsqueeze_99: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_168, 0);  mul_168 = None
        unsqueeze_100: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
        unsqueeze_101: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, 3);  unsqueeze_100 = None
        mul_169: "f32[64]" = torch.ops.aten.mul.Tensor(sum_9, 8.155010438413362e-07)
        squeeze_46: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_170: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_171: "f32[64]" = torch.ops.aten.mul.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        unsqueeze_102: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_171, 0);  mul_171 = None
        unsqueeze_103: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 2);  unsqueeze_102 = None
        unsqueeze_104: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 3);  unsqueeze_103 = None
        mul_172: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_112);  primals_112 = None
        unsqueeze_105: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_172, 0);  mul_172 = None
        unsqueeze_106: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
        unsqueeze_107: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 3);  unsqueeze_106 = None
        mul_173: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_104);  sub_46 = unsqueeze_104 = None
        sub_48: "f32[8, 64, 320, 479]" = torch.ops.aten.sub.Tensor(where_2, mul_173);  where_2 = mul_173 = None
        sub_49: "f32[8, 64, 320, 479]" = torch.ops.aten.sub.Tensor(sub_48, unsqueeze_101);  sub_48 = unsqueeze_101 = None
        mul_174: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_107);  sub_49 = unsqueeze_107 = None
        mul_175: "f32[64]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_46);  sum_9 = squeeze_46 = None
        sum_10: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_174, relu_14, primals_107, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_174 = primals_107 = None
        getitem_53: "f32[8, 128, 320, 479]" = convolution_backward_3[0]
        getitem_54: "f32[64, 128, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
        le_3: "b8[8, 128, 320, 479]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_3: "f32[8, 128, 320, 479]" = torch.ops.aten.where.self(le_3, full_default, getitem_53);  le_3 = getitem_53 = None
        sum_11: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_50: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_110);  convolution_14 = unsqueeze_110 = None
        mul_176: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(where_3, sub_50)
        sum_12: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 2, 3]);  mul_176 = None
        mul_177: "f32[128]" = torch.ops.aten.mul.Tensor(sum_11, 8.155010438413362e-07)
        unsqueeze_111: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_177, 0);  mul_177 = None
        unsqueeze_112: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
        unsqueeze_113: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 3);  unsqueeze_112 = None
        mul_178: "f32[128]" = torch.ops.aten.mul.Tensor(sum_12, 8.155010438413362e-07)
        mul_179: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_180: "f32[128]" = torch.ops.aten.mul.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
        unsqueeze_114: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_180, 0);  mul_180 = None
        unsqueeze_115: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
        unsqueeze_116: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 3);  unsqueeze_115 = None
        mul_181: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_105);  primals_105 = None
        unsqueeze_117: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_181, 0);  mul_181 = None
        unsqueeze_118: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
        unsqueeze_119: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
        mul_182: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_116);  sub_50 = unsqueeze_116 = None
        sub_52: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(where_3, mul_182);  where_3 = mul_182 = None
        sub_53: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(sub_52, unsqueeze_113);  sub_52 = unsqueeze_113 = None
        mul_183: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_119);  sub_53 = unsqueeze_119 = None
        mul_184: "f32[128]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_43);  sum_12 = squeeze_43 = None
        sum_13: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_183, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_183, cat_2, primals_100, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_183 = cat_2 = primals_100 = None
        getitem_56: "f32[8, 256, 320, 479]" = convolution_backward_4[0]
        getitem_57: "f32[128, 256, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        slice_3: "f32[8, 128, 320, 479]" = torch.ops.aten.slice.Tensor(getitem_56, 1, 0, 128)
        slice_4: "f32[8, 128, 320, 479]" = torch.ops.aten.slice.Tensor(getitem_56, 1, 128, 256);  getitem_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_5: "f32[8, 128, 320, 478]" = torch.ops.aten.constant_pad_nd.default(slice_4, [0, -1, 0, 0]);  slice_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        mul_185: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(constant_pad_nd_5, clamp_max_11);  clamp_max_11 = None
        neg_3: "f32[8, 128, 320, 478]" = torch.ops.aten.neg.default(mul_185)
        add_116: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(constant_pad_nd_5, neg_3);  constant_pad_nd_5 = neg_3 = None
        mul_186: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(mul_185, clamp_max_10)
        neg_4: "f32[8, 128, 320, 478]" = torch.ops.aten.neg.default(mul_186)
        add_117: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(mul_185, neg_4);  mul_185 = neg_4 = None
        mul_187: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(add_116, clamp_max_10);  clamp_max_10 = None
        neg_5: "f32[8, 128, 320, 478]" = torch.ops.aten.neg.default(mul_187)
        add_118: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(add_116, neg_5);  add_116 = neg_5 = None
        full_default_8: "f32[8, 128, 160, 239]" = torch.ops.aten.full.default([8, 128, 160, 239], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_4: "f32[8, 128, 160, 239]" = torch.ops.aten.index_put.default(full_default_8, [None, None, clamp_max_8, clamp_max_9], mul_186, True);  mul_186 = None
        index_put_5: "f32[8, 128, 160, 239]" = torch.ops.aten.index_put.default(full_default_8, [None, None, clamp_max_8, convert_element_type_11], add_117, True);  clamp_max_8 = add_117 = None
        add_119: "f32[8, 128, 160, 239]" = torch.ops.aten.add.Tensor(index_put_4, index_put_5);  index_put_4 = index_put_5 = None
        index_put_6: "f32[8, 128, 160, 239]" = torch.ops.aten.index_put.default(full_default_8, [None, None, convert_element_type_9, clamp_max_9], mul_187, True);  clamp_max_9 = mul_187 = None
        add_120: "f32[8, 128, 160, 239]" = torch.ops.aten.add.Tensor(add_119, index_put_6);  add_119 = index_put_6 = None
        index_put_7: "f32[8, 128, 160, 239]" = torch.ops.aten.index_put.default(full_default_8, [None, None, convert_element_type_9, convert_element_type_11], add_118, True);  full_default_8 = convert_element_type_9 = convert_element_type_11 = add_118 = None
        add_121: "f32[8, 128, 160, 239]" = torch.ops.aten.add.Tensor(add_120, index_put_7);  add_120 = index_put_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_23: "f32[8, 128, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_35)
        mul_101: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_13);  sub_23 = None
        unsqueeze_52: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
        unsqueeze_53: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_107: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(mul_101, unsqueeze_53);  mul_101 = unsqueeze_53 = None
        unsqueeze_54: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
        unsqueeze_55: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_79: "f32[8, 128, 160, 239]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_55);  mul_107 = unsqueeze_55 = None
        relu_13: "f32[8, 128, 160, 239]" = torch.ops.aten.relu.default(add_79);  add_79 = None
        le_4: "b8[8, 128, 160, 239]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_4: "f32[8, 128, 160, 239]" = torch.ops.aten.where.self(le_4, full_default, add_121);  le_4 = add_121 = None
        squeeze_39: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        unsqueeze_120: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_121: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None
        unsqueeze_122: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
        sum_14: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_54: "f32[8, 128, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_122);  convolution_13 = unsqueeze_122 = None
        mul_188: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(where_4, sub_54)
        sum_15: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 2, 3]);  mul_188 = None
        mul_189: "f32[128]" = torch.ops.aten.mul.Tensor(sum_14, 3.268828451882845e-06)
        unsqueeze_123: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_189, 0);  mul_189 = None
        unsqueeze_124: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
        unsqueeze_125: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 3);  unsqueeze_124 = None
        mul_190: "f32[128]" = torch.ops.aten.mul.Tensor(sum_15, 3.268828451882845e-06)
        squeeze_40: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_191: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_192: "f32[128]" = torch.ops.aten.mul.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        unsqueeze_126: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_192, 0);  mul_192 = None
        unsqueeze_127: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 2);  unsqueeze_126 = None
        unsqueeze_128: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 3);  unsqueeze_127 = None
        mul_193: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_98);  primals_98 = None
        unsqueeze_129: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_193, 0);  mul_193 = None
        unsqueeze_130: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
        unsqueeze_131: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 3);  unsqueeze_130 = None
        mul_194: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_128);  sub_54 = unsqueeze_128 = None
        sub_56: "f32[8, 128, 160, 239]" = torch.ops.aten.sub.Tensor(where_4, mul_194);  where_4 = mul_194 = None
        sub_57: "f32[8, 128, 160, 239]" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_125);  sub_56 = unsqueeze_125 = None
        mul_195: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_131);  sub_57 = unsqueeze_131 = None
        mul_196: "f32[128]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_40);  sum_15 = squeeze_40 = None
        sum_16: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_195, [0, 2, 3])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_195, relu_12, primals_93, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_195 = primals_93 = None
        getitem_59: "f32[8, 256, 160, 239]" = convolution_backward_5[0]
        getitem_60: "f32[128, 256, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
        le_5: "b8[8, 256, 160, 239]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_5: "f32[8, 256, 160, 239]" = torch.ops.aten.where.self(le_5, full_default, getitem_59);  le_5 = getitem_59 = None
        sum_17: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_58: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_134);  convolution_12 = unsqueeze_134 = None
        mul_197: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(where_5, sub_58)
        sum_18: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 2, 3]);  mul_197 = None
        mul_198: "f32[256]" = torch.ops.aten.mul.Tensor(sum_17, 3.268828451882845e-06)
        unsqueeze_135: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_198, 0);  mul_198 = None
        unsqueeze_136: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
        unsqueeze_137: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 3);  unsqueeze_136 = None
        mul_199: "f32[256]" = torch.ops.aten.mul.Tensor(sum_18, 3.268828451882845e-06)
        mul_200: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_201: "f32[256]" = torch.ops.aten.mul.Tensor(mul_199, mul_200);  mul_199 = mul_200 = None
        unsqueeze_138: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_201, 0);  mul_201 = None
        unsqueeze_139: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 2);  unsqueeze_138 = None
        unsqueeze_140: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 3);  unsqueeze_139 = None
        mul_202: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_91);  primals_91 = None
        unsqueeze_141: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_202, 0);  mul_202 = None
        unsqueeze_142: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None
        unsqueeze_143: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 3);  unsqueeze_142 = None
        mul_203: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_140);  sub_58 = unsqueeze_140 = None
        sub_60: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(where_5, mul_203);  where_5 = mul_203 = None
        sub_61: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_137);  sub_60 = unsqueeze_137 = None
        mul_204: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_143);  sub_61 = unsqueeze_143 = None
        mul_205: "f32[256]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_37);  sum_18 = squeeze_37 = None
        sum_19: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 2, 3])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_204, cat_1, primals_86, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_204 = cat_1 = primals_86 = None
        getitem_62: "f32[8, 512, 160, 239]" = convolution_backward_6[0]
        getitem_63: "f32[256, 512, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        slice_5: "f32[8, 256, 160, 239]" = torch.ops.aten.slice.Tensor(getitem_62, 1, 0, 256)
        slice_6: "f32[8, 256, 160, 239]" = torch.ops.aten.slice.Tensor(getitem_62, 1, 256, 512);  getitem_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_6: "f32[8, 256, 160, 238]" = torch.ops.aten.constant_pad_nd.default(slice_6, [0, -1, 0, 0]);  slice_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        mul_206: "f32[8, 256, 160, 238]" = torch.ops.aten.mul.Tensor(constant_pad_nd_6, clamp_max_7);  clamp_max_7 = None
        neg_6: "f32[8, 256, 160, 238]" = torch.ops.aten.neg.default(mul_206)
        add_122: "f32[8, 256, 160, 238]" = torch.ops.aten.add.Tensor(constant_pad_nd_6, neg_6);  constant_pad_nd_6 = neg_6 = None
        mul_207: "f32[8, 256, 160, 238]" = torch.ops.aten.mul.Tensor(mul_206, clamp_max_6)
        neg_7: "f32[8, 256, 160, 238]" = torch.ops.aten.neg.default(mul_207)
        add_123: "f32[8, 256, 160, 238]" = torch.ops.aten.add.Tensor(mul_206, neg_7);  mul_206 = neg_7 = None
        mul_208: "f32[8, 256, 160, 238]" = torch.ops.aten.mul.Tensor(add_122, clamp_max_6);  clamp_max_6 = None
        neg_8: "f32[8, 256, 160, 238]" = torch.ops.aten.neg.default(mul_208)
        add_124: "f32[8, 256, 160, 238]" = torch.ops.aten.add.Tensor(add_122, neg_8);  add_122 = neg_8 = None
        full_default_14: "f32[8, 256, 80, 119]" = torch.ops.aten.full.default([8, 256, 80, 119], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_8: "f32[8, 256, 80, 119]" = torch.ops.aten.index_put.default(full_default_14, [None, None, clamp_max_4, clamp_max_5], mul_207, True);  mul_207 = None
        index_put_9: "f32[8, 256, 80, 119]" = torch.ops.aten.index_put.default(full_default_14, [None, None, clamp_max_4, convert_element_type_7], add_123, True);  clamp_max_4 = add_123 = None
        add_125: "f32[8, 256, 80, 119]" = torch.ops.aten.add.Tensor(index_put_8, index_put_9);  index_put_8 = index_put_9 = None
        index_put_10: "f32[8, 256, 80, 119]" = torch.ops.aten.index_put.default(full_default_14, [None, None, convert_element_type_5, clamp_max_5], mul_208, True);  clamp_max_5 = mul_208 = None
        add_126: "f32[8, 256, 80, 119]" = torch.ops.aten.add.Tensor(add_125, index_put_10);  add_125 = index_put_10 = None
        index_put_11: "f32[8, 256, 80, 119]" = torch.ops.aten.index_put.default(full_default_14, [None, None, convert_element_type_5, convert_element_type_7], add_124, True);  full_default_14 = convert_element_type_5 = convert_element_type_7 = add_124 = None
        add_127: "f32[8, 256, 80, 119]" = torch.ops.aten.add.Tensor(add_126, index_put_11);  add_126 = index_put_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_16: "f32[8, 256, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_31)
        mul_82: "f32[8, 256, 80, 119]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_11);  sub_16 = None
        unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_88: "f32[8, 256, 80, 119]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_45);  mul_82 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_64: "f32[8, 256, 80, 119]" = torch.ops.aten.add.Tensor(mul_88, unsqueeze_47);  mul_88 = unsqueeze_47 = None
        relu_11: "f32[8, 256, 80, 119]" = torch.ops.aten.relu.default(add_64);  add_64 = None
        le_6: "b8[8, 256, 80, 119]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_6: "f32[8, 256, 80, 119]" = torch.ops.aten.where.self(le_6, full_default, add_127);  le_6 = add_127 = None
        squeeze_33: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        unsqueeze_144: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_145: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 2);  unsqueeze_144 = None
        unsqueeze_146: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 3);  unsqueeze_145 = None
        sum_20: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_62: "f32[8, 256, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_146);  convolution_11 = unsqueeze_146 = None
        mul_209: "f32[8, 256, 80, 119]" = torch.ops.aten.mul.Tensor(where_6, sub_62)
        sum_21: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_209, [0, 2, 3]);  mul_209 = None
        mul_210: "f32[256]" = torch.ops.aten.mul.Tensor(sum_20, 1.3130252100840337e-05)
        unsqueeze_147: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_210, 0);  mul_210 = None
        unsqueeze_148: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
        unsqueeze_149: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 3);  unsqueeze_148 = None
        mul_211: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, 1.3130252100840337e-05)
        squeeze_34: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_212: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_213: "f32[256]" = torch.ops.aten.mul.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        unsqueeze_150: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_213, 0);  mul_213 = None
        unsqueeze_151: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 2);  unsqueeze_150 = None
        unsqueeze_152: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 3);  unsqueeze_151 = None
        mul_214: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_84);  primals_84 = None
        unsqueeze_153: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_214, 0);  mul_214 = None
        unsqueeze_154: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 2);  unsqueeze_153 = None
        unsqueeze_155: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 3);  unsqueeze_154 = None
        mul_215: "f32[8, 256, 80, 119]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_152);  sub_62 = unsqueeze_152 = None
        sub_64: "f32[8, 256, 80, 119]" = torch.ops.aten.sub.Tensor(where_6, mul_215);  where_6 = mul_215 = None
        sub_65: "f32[8, 256, 80, 119]" = torch.ops.aten.sub.Tensor(sub_64, unsqueeze_149);  sub_64 = unsqueeze_149 = None
        mul_216: "f32[8, 256, 80, 119]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_155);  sub_65 = unsqueeze_155 = None
        mul_217: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_34);  sum_21 = squeeze_34 = None
        sum_22: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_216, relu_10, primals_79, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_216 = primals_79 = None
        getitem_65: "f32[8, 512, 80, 119]" = convolution_backward_7[0]
        getitem_66: "f32[256, 512, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
        le_7: "b8[8, 512, 80, 119]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_7: "f32[8, 512, 80, 119]" = torch.ops.aten.where.self(le_7, full_default, getitem_65);  le_7 = getitem_65 = None
        sum_23: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_66: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_158);  convolution_10 = unsqueeze_158 = None
        mul_218: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(where_7, sub_66)
        sum_24: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 2, 3]);  mul_218 = None
        mul_219: "f32[512]" = torch.ops.aten.mul.Tensor(sum_23, 1.3130252100840337e-05)
        unsqueeze_159: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_219, 0);  mul_219 = None
        unsqueeze_160: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
        unsqueeze_161: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 3);  unsqueeze_160 = None
        mul_220: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, 1.3130252100840337e-05)
        mul_221: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_222: "f32[512]" = torch.ops.aten.mul.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
        unsqueeze_162: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_222, 0);  mul_222 = None
        unsqueeze_163: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 2);  unsqueeze_162 = None
        unsqueeze_164: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 3);  unsqueeze_163 = None
        mul_223: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_77);  primals_77 = None
        unsqueeze_165: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_223, 0);  mul_223 = None
        unsqueeze_166: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
        unsqueeze_167: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 3);  unsqueeze_166 = None
        mul_224: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_164);  sub_66 = unsqueeze_164 = None
        sub_68: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(where_7, mul_224);  where_7 = mul_224 = None
        sub_69: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_161);  sub_68 = unsqueeze_161 = None
        mul_225: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_167);  sub_69 = unsqueeze_167 = None
        mul_226: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_31);  sum_24 = squeeze_31 = None
        sum_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_225, cat, primals_72, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_225 = cat = primals_72 = None
        getitem_68: "f32[8, 1024, 80, 119]" = convolution_backward_8[0]
        getitem_69: "f32[512, 1024, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        slice_7: "f32[8, 512, 80, 119]" = torch.ops.aten.slice.Tensor(getitem_68, 1, 0, 512)
        slice_8: "f32[8, 512, 80, 119]" = torch.ops.aten.slice.Tensor(getitem_68, 1, 512, 1024);  getitem_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_7: "f32[8, 512, 80, 118]" = torch.ops.aten.constant_pad_nd.default(slice_8, [0, -1, 0, 0]);  slice_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        mul_227: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(constant_pad_nd_7, clamp_max_3);  clamp_max_3 = None
        neg_9: "f32[8, 512, 80, 118]" = torch.ops.aten.neg.default(mul_227)
        add_128: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(constant_pad_nd_7, neg_9);  constant_pad_nd_7 = neg_9 = None
        mul_228: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(mul_227, clamp_max_2)
        neg_10: "f32[8, 512, 80, 118]" = torch.ops.aten.neg.default(mul_228)
        add_129: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(mul_227, neg_10);  mul_227 = neg_10 = None
        mul_229: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(add_128, clamp_max_2);  clamp_max_2 = None
        neg_11: "f32[8, 512, 80, 118]" = torch.ops.aten.neg.default(mul_229)
        add_130: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(add_128, neg_11);  add_128 = neg_11 = None
        full_default_20: "f32[8, 512, 40, 59]" = torch.ops.aten.full.default([8, 512, 40, 59], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_12: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default_20, [None, None, clamp_max, clamp_max_1], mul_228, True);  mul_228 = None
        index_put_13: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default_20, [None, None, clamp_max, convert_element_type_3], add_129, True);  clamp_max = add_129 = None
        add_131: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(index_put_12, index_put_13);  index_put_12 = index_put_13 = None
        index_put_14: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default_20, [None, None, convert_element_type_1, clamp_max_1], mul_229, True);  clamp_max_1 = mul_229 = None
        add_132: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(add_131, index_put_14);  add_131 = index_put_14 = None
        index_put_15: "f32[8, 512, 40, 59]" = torch.ops.aten.index_put.default(full_default_20, [None, None, convert_element_type_1, convert_element_type_3], add_130, True);  full_default_20 = convert_element_type_1 = convert_element_type_3 = add_130 = None
        add_133: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(add_132, index_put_15);  add_132 = index_put_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_9: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_27)
        mul_63: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        unsqueeze_36: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1)
        unsqueeze_37: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1);  primals_71 = None
        unsqueeze_39: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_49: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        relu_9: "f32[8, 512, 40, 59]" = torch.ops.aten.relu.default(add_49);  add_49 = None
        le_8: "b8[8, 512, 40, 59]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_8: "f32[8, 512, 40, 59]" = torch.ops.aten.where.self(le_8, full_default, add_133);  le_8 = add_133 = None
        squeeze_27: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        unsqueeze_168: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_169: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 2);  unsqueeze_168 = None
        unsqueeze_170: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 3);  unsqueeze_169 = None
        sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_70: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_170);  convolution_9 = unsqueeze_170 = None
        mul_230: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(where_8, sub_70)
        sum_27: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 2, 3]);  mul_230 = None
        mul_231: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, 5.296610169491525e-05)
        unsqueeze_171: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_231, 0);  mul_231 = None
        unsqueeze_172: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
        unsqueeze_173: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 3);  unsqueeze_172 = None
        mul_232: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, 5.296610169491525e-05)
        squeeze_28: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_233: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_234: "f32[512]" = torch.ops.aten.mul.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        unsqueeze_174: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_234, 0);  mul_234 = None
        unsqueeze_175: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 2);  unsqueeze_174 = None
        unsqueeze_176: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 3);  unsqueeze_175 = None
        mul_235: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_70);  primals_70 = None
        unsqueeze_177: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_235, 0);  mul_235 = None
        unsqueeze_178: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
        unsqueeze_179: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
        mul_236: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_176);  sub_70 = unsqueeze_176 = None
        sub_72: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(where_8, mul_236);  where_8 = mul_236 = None
        sub_73: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_173);  sub_72 = unsqueeze_173 = None
        mul_237: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_179);  sub_73 = unsqueeze_179 = None
        mul_238: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_28);  sum_27 = squeeze_28 = None
        sum_28: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_237, [0, 2, 3])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_237, relu_8, primals_65, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_237 = primals_65 = None
        getitem_71: "f32[8, 512, 40, 59]" = convolution_backward_9[0]
        getitem_72: "f32[512, 512, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
        le_9: "b8[8, 512, 40, 59]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_9: "f32[8, 512, 40, 59]" = torch.ops.aten.where.self(le_9, full_default, getitem_71);  le_9 = getitem_71 = None
        sum_29: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_74: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_182);  convolution_8 = unsqueeze_182 = None
        mul_239: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(where_9, sub_74)
        sum_30: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 2, 3]);  mul_239 = None
        mul_240: "f32[512]" = torch.ops.aten.mul.Tensor(sum_29, 5.296610169491525e-05)
        unsqueeze_183: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_240, 0);  mul_240 = None
        unsqueeze_184: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
        unsqueeze_185: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 3);  unsqueeze_184 = None
        mul_241: "f32[512]" = torch.ops.aten.mul.Tensor(sum_30, 5.296610169491525e-05)
        mul_242: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_243: "f32[512]" = torch.ops.aten.mul.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
        unsqueeze_186: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_243, 0);  mul_243 = None
        unsqueeze_187: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 2);  unsqueeze_186 = None
        unsqueeze_188: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 3);  unsqueeze_187 = None
        mul_244: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_63);  primals_63 = None
        unsqueeze_189: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_244, 0);  mul_244 = None
        unsqueeze_190: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
        unsqueeze_191: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 3);  unsqueeze_190 = None
        mul_245: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_188);  sub_74 = unsqueeze_188 = None
        sub_76: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(where_9, mul_245);  where_9 = mul_245 = None
        sub_77: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(sub_76, unsqueeze_185);  sub_76 = unsqueeze_185 = None
        mul_246: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_191);  sub_77 = unsqueeze_191 = None
        mul_247: "f32[512]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_25);  sum_30 = squeeze_25 = None
        sum_31: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 2, 3])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_246, getitem_22, primals_58, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_246 = getitem_22 = primals_58 = None
        getitem_74: "f32[8, 512, 40, 59]" = convolution_backward_10[0]
        getitem_75: "f32[512, 512, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        full_default_26: "f32[4096, 9520]" = torch.ops.aten.full.default([4096, 9520], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_8: "f32[4096, 2360]" = torch.ops.aten.reshape.default(getitem_74, [4096, 2360]);  getitem_74 = None
        _low_memory_max_pool_offsets_to_indices_3: "i64[8, 512, 40, 59]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_23, [2, 2], [80, 119], [2, 2], [0, 0], [1, 1]);  getitem_23 = None
        view_9: "i64[4096, 2360]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_3, [4096, 2360]);  _low_memory_max_pool_offsets_to_indices_3 = None
        scatter_add: "f32[4096, 9520]" = torch.ops.aten.scatter_add.default(full_default_26, 1, view_9, view_8);  full_default_26 = view_9 = view_8 = None
        view_10: "f32[8, 512, 80, 119]" = torch.ops.aten.reshape.default(scatter_add, [8, 512, 80, 119]);  scatter_add = None
        add_134: "f32[8, 512, 80, 119]" = torch.ops.aten.add.Tensor(slice_7, view_10);  slice_7 = view_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_7: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_21)
        mul_49: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        unsqueeze_28: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
        unsqueeze_29: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
        unsqueeze_31: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_39: "f32[8, 512, 80, 119]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        relu_7: "f32[8, 512, 80, 119]" = torch.ops.aten.relu.default(add_39);  add_39 = None
        le_10: "b8[8, 512, 80, 119]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_10: "f32[8, 512, 80, 119]" = torch.ops.aten.where.self(le_10, full_default, add_134);  le_10 = add_134 = None
        squeeze_21: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        unsqueeze_192: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_193: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 2);  unsqueeze_192 = None
        unsqueeze_194: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 3);  unsqueeze_193 = None
        sum_32: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_78: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_194);  convolution_7 = unsqueeze_194 = None
        mul_248: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(where_10, sub_78)
        sum_33: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 2, 3]);  mul_248 = None
        mul_249: "f32[512]" = torch.ops.aten.mul.Tensor(sum_32, 1.3130252100840337e-05)
        unsqueeze_195: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_249, 0);  mul_249 = None
        unsqueeze_196: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
        unsqueeze_197: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 3);  unsqueeze_196 = None
        mul_250: "f32[512]" = torch.ops.aten.mul.Tensor(sum_33, 1.3130252100840337e-05)
        squeeze_22: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_251: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_252: "f32[512]" = torch.ops.aten.mul.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
        unsqueeze_198: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_252, 0);  mul_252 = None
        unsqueeze_199: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 2);  unsqueeze_198 = None
        unsqueeze_200: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 3);  unsqueeze_199 = None
        mul_253: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_56);  primals_56 = None
        unsqueeze_201: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_253, 0);  mul_253 = None
        unsqueeze_202: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
        unsqueeze_203: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 3);  unsqueeze_202 = None
        mul_254: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_200);  sub_78 = unsqueeze_200 = None
        sub_80: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(where_10, mul_254);  where_10 = mul_254 = None
        sub_81: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_197);  sub_80 = unsqueeze_197 = None
        mul_255: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_203);  sub_81 = unsqueeze_203 = None
        mul_256: "f32[512]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_22);  sum_33 = squeeze_22 = None
        sum_34: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 2, 3])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_255, relu_6, primals_51, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_255 = primals_51 = None
        getitem_77: "f32[8, 512, 80, 119]" = convolution_backward_11[0]
        getitem_78: "f32[512, 512, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
        le_11: "b8[8, 512, 80, 119]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_11: "f32[8, 512, 80, 119]" = torch.ops.aten.where.self(le_11, full_default, getitem_77);  le_11 = getitem_77 = None
        sum_35: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_82: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_206);  convolution_6 = unsqueeze_206 = None
        mul_257: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(where_11, sub_82)
        sum_36: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 2, 3]);  mul_257 = None
        mul_258: "f32[512]" = torch.ops.aten.mul.Tensor(sum_35, 1.3130252100840337e-05)
        unsqueeze_207: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_258, 0);  mul_258 = None
        unsqueeze_208: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
        unsqueeze_209: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 3);  unsqueeze_208 = None
        mul_259: "f32[512]" = torch.ops.aten.mul.Tensor(sum_36, 1.3130252100840337e-05)
        mul_260: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_261: "f32[512]" = torch.ops.aten.mul.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
        unsqueeze_210: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_261, 0);  mul_261 = None
        unsqueeze_211: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 2);  unsqueeze_210 = None
        unsqueeze_212: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 3);  unsqueeze_211 = None
        mul_262: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_49);  primals_49 = None
        unsqueeze_213: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_262, 0);  mul_262 = None
        unsqueeze_214: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
        unsqueeze_215: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 3);  unsqueeze_214 = None
        mul_263: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_212);  sub_82 = unsqueeze_212 = None
        sub_84: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(where_11, mul_263);  where_11 = mul_263 = None
        sub_85: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_209);  sub_84 = unsqueeze_209 = None
        mul_264: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_215);  sub_85 = unsqueeze_215 = None
        mul_265: "f32[512]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_19);  sum_36 = squeeze_19 = None
        sum_37: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_264, getitem_16, primals_44, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_264 = getitem_16 = primals_44 = None
        getitem_80: "f32[8, 256, 80, 119]" = convolution_backward_12[0]
        getitem_81: "f32[512, 256, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        full_default_29: "f32[2048, 38240]" = torch.ops.aten.full.default([2048, 38240], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_11: "f32[2048, 9520]" = torch.ops.aten.reshape.default(getitem_80, [2048, 9520]);  getitem_80 = None
        _low_memory_max_pool_offsets_to_indices_2: "i64[8, 256, 80, 119]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_17, [2, 2], [160, 239], [2, 2], [0, 0], [1, 1]);  getitem_17 = None
        view_12: "i64[2048, 9520]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_2, [2048, 9520]);  _low_memory_max_pool_offsets_to_indices_2 = None
        scatter_add_1: "f32[2048, 38240]" = torch.ops.aten.scatter_add.default(full_default_29, 1, view_12, view_11);  full_default_29 = view_12 = view_11 = None
        view_13: "f32[8, 256, 160, 239]" = torch.ops.aten.reshape.default(scatter_add_1, [8, 256, 160, 239]);  scatter_add_1 = None
        add_135: "f32[8, 256, 160, 239]" = torch.ops.aten.add.Tensor(slice_5, view_13);  slice_5 = view_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_5: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_15)
        mul_35: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        unsqueeze_20: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_21: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_23: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_29: "f32[8, 256, 160, 239]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        relu_5: "f32[8, 256, 160, 239]" = torch.ops.aten.relu.default(add_29);  add_29 = None
        le_12: "b8[8, 256, 160, 239]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_12: "f32[8, 256, 160, 239]" = torch.ops.aten.where.self(le_12, full_default, add_135);  le_12 = add_135 = None
        squeeze_15: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        unsqueeze_216: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_217: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 2);  unsqueeze_216 = None
        unsqueeze_218: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 3);  unsqueeze_217 = None
        sum_38: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_86: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_218);  convolution_5 = unsqueeze_218 = None
        mul_266: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(where_12, sub_86)
        sum_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 2, 3]);  mul_266 = None
        mul_267: "f32[256]" = torch.ops.aten.mul.Tensor(sum_38, 3.268828451882845e-06)
        unsqueeze_219: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_267, 0);  mul_267 = None
        unsqueeze_220: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
        unsqueeze_221: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 3);  unsqueeze_220 = None
        mul_268: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, 3.268828451882845e-06)
        squeeze_16: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_269: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_270: "f32[256]" = torch.ops.aten.mul.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
        unsqueeze_222: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_270, 0);  mul_270 = None
        unsqueeze_223: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
        unsqueeze_224: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 3);  unsqueeze_223 = None
        mul_271: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_42);  primals_42 = None
        unsqueeze_225: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
        unsqueeze_226: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
        unsqueeze_227: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 3);  unsqueeze_226 = None
        mul_272: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_224);  sub_86 = unsqueeze_224 = None
        sub_88: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(where_12, mul_272);  where_12 = mul_272 = None
        sub_89: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(sub_88, unsqueeze_221);  sub_88 = unsqueeze_221 = None
        mul_273: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_227);  sub_89 = unsqueeze_227 = None
        mul_274: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_16);  sum_39 = squeeze_16 = None
        sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_273, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_273, relu_4, primals_37, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_273 = primals_37 = None
        getitem_83: "f32[8, 256, 160, 239]" = convolution_backward_13[0]
        getitem_84: "f32[256, 256, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
        le_13: "b8[8, 256, 160, 239]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_13: "f32[8, 256, 160, 239]" = torch.ops.aten.where.self(le_13, full_default, getitem_83);  le_13 = getitem_83 = None
        sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_90: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_230);  convolution_4 = unsqueeze_230 = None
        mul_275: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(where_13, sub_90)
        sum_42: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 2, 3]);  mul_275 = None
        mul_276: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 3.268828451882845e-06)
        unsqueeze_231: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_276, 0);  mul_276 = None
        unsqueeze_232: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
        unsqueeze_233: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
        mul_277: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, 3.268828451882845e-06)
        mul_278: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_279: "f32[256]" = torch.ops.aten.mul.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_234: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_279, 0);  mul_279 = None
        unsqueeze_235: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
        unsqueeze_236: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
        mul_280: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_35);  primals_35 = None
        unsqueeze_237: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_280, 0);  mul_280 = None
        unsqueeze_238: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
        unsqueeze_239: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
        mul_281: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_236);  sub_90 = unsqueeze_236 = None
        sub_92: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(where_13, mul_281);  where_13 = mul_281 = None
        sub_93: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(sub_92, unsqueeze_233);  sub_92 = unsqueeze_233 = None
        mul_282: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_239);  sub_93 = unsqueeze_239 = None
        mul_283: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_13);  sum_42 = squeeze_13 = None
        sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3])
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_282, getitem_10, primals_30, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_282 = getitem_10 = primals_30 = None
        getitem_86: "f32[8, 128, 160, 239]" = convolution_backward_14[0]
        getitem_87: "f32[256, 128, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        full_default_32: "f32[1024, 153280]" = torch.ops.aten.full.default([1024, 153280], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_14: "f32[1024, 38240]" = torch.ops.aten.reshape.default(getitem_86, [1024, 38240]);  getitem_86 = None
        _low_memory_max_pool_offsets_to_indices_1: "i64[8, 128, 160, 239]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_11, [2, 2], [320, 479], [2, 2], [0, 0], [1, 1]);  getitem_11 = None
        view_15: "i64[1024, 38240]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_1, [1024, 38240]);  _low_memory_max_pool_offsets_to_indices_1 = None
        scatter_add_2: "f32[1024, 153280]" = torch.ops.aten.scatter_add.default(full_default_32, 1, view_15, view_14);  full_default_32 = view_15 = view_14 = None
        view_16: "f32[8, 128, 320, 479]" = torch.ops.aten.reshape.default(scatter_add_2, [8, 128, 320, 479]);  scatter_add_2 = None
        add_136: "f32[8, 128, 320, 479]" = torch.ops.aten.add.Tensor(slice_3, view_16);  slice_3 = view_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_3: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_21: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1)
        unsqueeze_13: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1);  primals_29 = None
        unsqueeze_15: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[8, 128, 320, 479]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        relu_3: "f32[8, 128, 320, 479]" = torch.ops.aten.relu.default(add_19);  add_19 = None
        le_14: "b8[8, 128, 320, 479]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_14: "f32[8, 128, 320, 479]" = torch.ops.aten.where.self(le_14, full_default, add_136);  le_14 = add_136 = None
        squeeze_9: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        unsqueeze_240: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_241: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
        unsqueeze_242: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
        sum_44: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_94: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_242);  convolution_3 = unsqueeze_242 = None
        mul_284: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(where_14, sub_94)
        sum_45: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_284, [0, 2, 3]);  mul_284 = None
        mul_285: "f32[128]" = torch.ops.aten.mul.Tensor(sum_44, 8.155010438413362e-07)
        unsqueeze_243: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_285, 0);  mul_285 = None
        unsqueeze_244: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
        unsqueeze_245: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
        mul_286: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, 8.155010438413362e-07)
        squeeze_10: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_287: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_288: "f32[128]" = torch.ops.aten.mul.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None
        unsqueeze_246: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_288, 0);  mul_288 = None
        unsqueeze_247: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
        unsqueeze_248: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
        mul_289: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_28);  primals_28 = None
        unsqueeze_249: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_289, 0);  mul_289 = None
        unsqueeze_250: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
        unsqueeze_251: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
        mul_290: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_248);  sub_94 = unsqueeze_248 = None
        sub_96: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(where_14, mul_290);  where_14 = mul_290 = None
        sub_97: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_245);  sub_96 = unsqueeze_245 = None
        mul_291: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_251);  sub_97 = unsqueeze_251 = None
        mul_292: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_10);  sum_45 = squeeze_10 = None
        sum_46: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 2, 3])
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_291, relu_2, primals_23, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_291 = primals_23 = None
        getitem_89: "f32[8, 128, 320, 479]" = convolution_backward_15[0]
        getitem_90: "f32[128, 128, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
        le_15: "b8[8, 128, 320, 479]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_15: "f32[8, 128, 320, 479]" = torch.ops.aten.where.self(le_15, full_default, getitem_89);  le_15 = getitem_89 = None
        sum_47: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_98: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_254);  convolution_2 = unsqueeze_254 = None
        mul_293: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(where_15, sub_98)
        sum_48: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3]);  mul_293 = None
        mul_294: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, 8.155010438413362e-07)
        unsqueeze_255: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
        unsqueeze_256: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
        unsqueeze_257: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
        mul_295: "f32[128]" = torch.ops.aten.mul.Tensor(sum_48, 8.155010438413362e-07)
        mul_296: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_297: "f32[128]" = torch.ops.aten.mul.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        unsqueeze_258: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_297, 0);  mul_297 = None
        unsqueeze_259: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 2);  unsqueeze_258 = None
        unsqueeze_260: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 3);  unsqueeze_259 = None
        mul_298: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_21);  primals_21 = None
        unsqueeze_261: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
        unsqueeze_262: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
        unsqueeze_263: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
        mul_299: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_260);  sub_98 = unsqueeze_260 = None
        sub_100: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(where_15, mul_299);  where_15 = mul_299 = None
        sub_101: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_257);  sub_100 = unsqueeze_257 = None
        mul_300: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_263);  sub_101 = unsqueeze_263 = None
        mul_301: "f32[128]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_7);  sum_48 = squeeze_7 = None
        sum_49: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 2, 3])
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_300, getitem_4, primals_16, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_300 = getitem_4 = primals_16 = None
        getitem_92: "f32[8, 64, 320, 479]" = convolution_backward_16[0]
        getitem_93: "f32[128, 64, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        full_default_35: "f32[512, 613760]" = torch.ops.aten.full.default([512, 613760], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_17: "f32[512, 153280]" = torch.ops.aten.reshape.default(getitem_92, [512, 153280]);  getitem_92 = None
        _low_memory_max_pool_offsets_to_indices: "i64[8, 64, 320, 479]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_5, [2, 2], [640, 959], [2, 2], [0, 0], [1, 1]);  getitem_5 = None
        view_18: "i64[512, 153280]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [512, 153280]);  _low_memory_max_pool_offsets_to_indices = None
        scatter_add_3: "f32[512, 613760]" = torch.ops.aten.scatter_add.default(full_default_35, 1, view_18, view_17);  full_default_35 = view_18 = view_17 = None
        view_19: "f32[8, 64, 640, 959]" = torch.ops.aten.reshape.default(scatter_add_3, [8, 64, 640, 959]);  scatter_add_3 = None
        add_137: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(slice_1, view_19);  slice_1 = view_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_1: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
        unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
        unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        relu_1: "f32[8, 64, 640, 959]" = torch.ops.aten.relu.default(add_9);  add_9 = None
        le_16: "b8[8, 64, 640, 959]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_16: "f32[8, 64, 640, 959]" = torch.ops.aten.where.self(le_16, full_default, add_137);  le_16 = add_137 = None
        squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        unsqueeze_264: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_265: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
        unsqueeze_266: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None
        sum_50: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_102: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_266);  convolution_1 = unsqueeze_266 = None
        mul_302: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where_16, sub_102)
        sum_51: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 2, 3]);  mul_302 = None
        mul_303: "f32[64]" = torch.ops.aten.mul.Tensor(sum_50, 2.0366266944734097e-07)
        unsqueeze_267: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_303, 0);  mul_303 = None
        unsqueeze_268: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
        unsqueeze_269: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
        mul_304: "f32[64]" = torch.ops.aten.mul.Tensor(sum_51, 2.0366266944734097e-07)
        squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_305: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_306: "f32[64]" = torch.ops.aten.mul.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
        unsqueeze_270: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_306, 0);  mul_306 = None
        unsqueeze_271: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
        unsqueeze_272: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
        mul_307: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_14);  primals_14 = None
        unsqueeze_273: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
        unsqueeze_274: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
        unsqueeze_275: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
        mul_308: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_272);  sub_102 = unsqueeze_272 = None
        sub_104: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(where_16, mul_308);  where_16 = mul_308 = None
        sub_105: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_269);  sub_104 = unsqueeze_269 = None
        mul_309: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_275);  sub_105 = unsqueeze_275 = None
        mul_310: "f32[64]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_4);  sum_51 = squeeze_4 = None
        sum_52: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_309, relu, primals_9, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_309 = primals_9 = None
        getitem_95: "f32[8, 64, 640, 959]" = convolution_backward_17[0]
        getitem_96: "f32[64, 64, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
        le_17: "b8[8, 64, 640, 959]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_17: "f32[8, 64, 640, 959]" = torch.ops.aten.where.self(le_17, full_default, getitem_95);  le_17 = full_default = getitem_95 = None
        sum_53: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_106: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_278);  convolution = unsqueeze_278 = None
        mul_311: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where_17, sub_106)
        sum_54: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 2, 3]);  mul_311 = None
        mul_312: "f32[64]" = torch.ops.aten.mul.Tensor(sum_53, 2.0366266944734097e-07)
        unsqueeze_279: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_312, 0);  mul_312 = None
        unsqueeze_280: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
        unsqueeze_281: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
        mul_313: "f32[64]" = torch.ops.aten.mul.Tensor(sum_54, 2.0366266944734097e-07)
        mul_314: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_315: "f32[64]" = torch.ops.aten.mul.Tensor(mul_313, mul_314);  mul_313 = mul_314 = None
        unsqueeze_282: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_315, 0);  mul_315 = None
        unsqueeze_283: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
        unsqueeze_284: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
        mul_316: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_7);  primals_7 = None
        unsqueeze_285: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
        unsqueeze_286: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
        unsqueeze_287: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
        mul_317: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_284);  sub_106 = unsqueeze_284 = None
        sub_108: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(where_17, mul_317);  where_17 = mul_317 = None
        sub_109: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(sub_108, unsqueeze_281);  sub_108 = unsqueeze_281 = None
        mul_318: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_287);  sub_109 = unsqueeze_287 = None
        mul_319: "f32[64]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_1);  sum_54 = squeeze_1 = None
        sum_55: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_318, primals_3, primals_1, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_318 = primals_3 = primals_1 = None
        getitem_99: "f32[64, 3, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
        return (getitem_99, sum_55, None, None, None, None, mul_319, sum_53, getitem_96, sum_52, None, None, None, mul_310, sum_50, getitem_93, sum_49, None, None, None, mul_301, sum_47, getitem_90, sum_46, None, None, None, mul_292, sum_44, getitem_87, sum_43, None, None, None, mul_283, sum_41, getitem_84, sum_40, None, None, None, mul_274, sum_38, getitem_81, sum_37, None, None, None, mul_265, sum_35, getitem_78, sum_34, None, None, None, mul_256, sum_32, getitem_75, sum_31, None, None, None, mul_247, sum_29, getitem_72, sum_28, None, None, None, mul_238, sum_26, getitem_69, sum_25, None, None, None, mul_226, sum_23, getitem_66, sum_22, None, None, None, mul_217, sum_20, getitem_63, sum_19, None, None, None, mul_205, sum_17, getitem_60, sum_16, None, None, None, mul_196, sum_14, getitem_57, sum_13, None, None, None, mul_184, sum_11, getitem_54, sum_10, None, None, None, mul_175, sum_8, getitem_51, sum_7, None, None, None, mul_163, sum_5, getitem_48, sum_4, None, None, None, mul_154, sum_2, getitem_45, sum_1)
