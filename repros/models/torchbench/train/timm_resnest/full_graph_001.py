class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[32, 3, 224, 224]", primals_6: "f32[32]", primals_8: "f32[32, 32, 3, 3]", primals_12: "f32[32]", primals_14: "f32[64, 32, 3, 3]", primals_18: "f32[64]", primals_19: "f32[64]", primals_20: "f32[64, 64, 1, 1]", primals_24: "f32[64]", primals_26: "f32[128, 32, 3, 3]", primals_30: "f32[128]", primals_31: "f32[128]", primals_32: "f32[32, 64, 1, 1]", primals_37: "f32[32]", primals_39: "f32[128, 32, 1, 1]", primals_41: "f32[256, 64, 1, 1]", primals_45: "f32[256]", primals_47: "f32[256, 64, 1, 1]", primals_51: "f32[256]", primals_53: "f32[128, 256, 1, 1]", primals_57: "f32[128]", primals_59: "f32[256, 64, 3, 3]", primals_63: "f32[256]", primals_64: "f32[256]", primals_65: "f32[64, 128, 1, 1]", primals_70: "f32[64]", primals_72: "f32[256, 64, 1, 1]", primals_74: "f32[512, 128, 1, 1]", primals_78: "f32[512]", primals_80: "f32[512, 256, 1, 1]", primals_84: "f32[512]", primals_86: "f32[256, 512, 1, 1]", primals_90: "f32[256]", primals_92: "f32[512, 128, 3, 3]", primals_96: "f32[512]", primals_97: "f32[512]", primals_98: "f32[128, 256, 1, 1]", primals_103: "f32[128]", primals_105: "f32[512, 128, 1, 1]", primals_107: "f32[1024, 256, 1, 1]", primals_111: "f32[1024]", primals_113: "f32[1024, 512, 1, 1]", primals_117: "f32[1024]", primals_119: "f32[512, 1024, 1, 1]", primals_123: "f32[512]", primals_125: "f32[1024, 256, 3, 3]", primals_129: "f32[1024]", primals_130: "f32[1024]", primals_131: "f32[256, 512, 1, 1]", primals_136: "f32[256]", primals_138: "f32[1024, 256, 1, 1]", primals_140: "f32[2048, 512, 1, 1]", primals_144: "f32[2048]", primals_145: "f32[2048]", primals_146: "f32[2048, 1024, 1, 1]", primals_150: "f32[2048]", primals_151: "f32[2048]", primals_152: "f32[1000, 2048]", convolution: "f32[32, 32, 112, 112]", squeeze_1: "f32[32]", relu: "f32[32, 32, 112, 112]", convolution_1: "f32[32, 32, 112, 112]", squeeze_4: "f32[32]", relu_1: "f32[32, 32, 112, 112]", convolution_2: "f32[32, 64, 112, 112]", getitem_5: "f32[1, 64, 1, 1]", rsqrt_2: "f32[1, 64, 1, 1]", getitem_6: "f32[32, 64, 56, 56]", getitem_7: "i8[32, 64, 56, 56]", convolution_3: "f32[32, 64, 56, 56]", squeeze_10: "f32[64]", relu_3: "f32[32, 64, 56, 56]", convolution_4: "f32[32, 128, 56, 56]", getitem_11: "f32[1, 128, 1, 1]", rsqrt_4: "f32[1, 128, 1, 1]", mean: "f32[32, 64, 1, 1]", convolution_5: "f32[32, 32, 1, 1]", squeeze_16: "f32[32]", relu_5: "f32[32, 32, 1, 1]", convolution_6: "f32[32, 128, 1, 1]", sum_3: "f32[32, 64, 56, 56]", convolution_7: "f32[32, 256, 56, 56]", squeeze_19: "f32[256]", convolution_8: "f32[32, 256, 56, 56]", squeeze_22: "f32[256]", relu_6: "f32[32, 256, 56, 56]", convolution_9: "f32[32, 128, 56, 56]", squeeze_25: "f32[128]", relu_7: "f32[32, 128, 56, 56]", convolution_10: "f32[32, 256, 56, 56]", getitem_21: "f32[1, 256, 1, 1]", rsqrt_9: "f32[1, 256, 1, 1]", mean_1: "f32[32, 128, 1, 1]", convolution_11: "f32[32, 64, 1, 1]", squeeze_31: "f32[64]", relu_9: "f32[32, 64, 1, 1]", convolution_12: "f32[32, 256, 1, 1]", sum_6: "f32[32, 128, 56, 56]", avg_pool2d: "f32[32, 128, 28, 28]", convolution_13: "f32[32, 512, 28, 28]", squeeze_34: "f32[512]", avg_pool2d_1: "f32[32, 256, 28, 28]", convolution_14: "f32[32, 512, 28, 28]", squeeze_37: "f32[512]", relu_10: "f32[32, 512, 28, 28]", convolution_15: "f32[32, 256, 28, 28]", squeeze_40: "f32[256]", relu_11: "f32[32, 256, 28, 28]", convolution_16: "f32[32, 512, 28, 28]", getitem_31: "f32[1, 512, 1, 1]", rsqrt_14: "f32[1, 512, 1, 1]", mean_2: "f32[32, 256, 1, 1]", convolution_17: "f32[32, 128, 1, 1]", squeeze_46: "f32[128]", relu_13: "f32[32, 128, 1, 1]", convolution_18: "f32[32, 512, 1, 1]", sum_9: "f32[32, 256, 28, 28]", avg_pool2d_2: "f32[32, 256, 14, 14]", convolution_19: "f32[32, 1024, 14, 14]", squeeze_49: "f32[1024]", avg_pool2d_3: "f32[32, 512, 14, 14]", convolution_20: "f32[32, 1024, 14, 14]", squeeze_52: "f32[1024]", relu_14: "f32[32, 1024, 14, 14]", convolution_21: "f32[32, 512, 14, 14]", squeeze_55: "f32[512]", relu_15: "f32[32, 512, 14, 14]", convolution_22: "f32[32, 1024, 14, 14]", getitem_41: "f32[1, 1024, 1, 1]", rsqrt_19: "f32[1, 1024, 1, 1]", mean_3: "f32[32, 512, 1, 1]", convolution_23: "f32[32, 256, 1, 1]", squeeze_61: "f32[256]", relu_17: "f32[32, 256, 1, 1]", convolution_24: "f32[32, 1024, 1, 1]", sum_12: "f32[32, 512, 14, 14]", avg_pool2d_4: "f32[32, 512, 7, 7]", convolution_25: "f32[32, 2048, 7, 7]", getitem_45: "f32[1, 2048, 1, 1]", rsqrt_21: "f32[1, 2048, 1, 1]", avg_pool2d_5: "f32[32, 1024, 7, 7]", convolution_26: "f32[32, 2048, 7, 7]", getitem_47: "f32[1, 2048, 1, 1]", rsqrt_22: "f32[1, 2048, 1, 1]", view_24: "f32[32, 2048]", unsqueeze_119: "f32[1, 256, 1, 1]", unsqueeze_144: "f32[1, 512, 1, 1]", unsqueeze_156: "f32[1, 1024, 1, 1]", unsqueeze_168: "f32[1, 1024, 1, 1]", unsqueeze_181: "f32[1, 128, 1, 1]", unsqueeze_206: "f32[1, 256, 1, 1]", unsqueeze_218: "f32[1, 512, 1, 1]", unsqueeze_230: "f32[1, 512, 1, 1]", unsqueeze_243: "f32[1, 64, 1, 1]", unsqueeze_268: "f32[1, 128, 1, 1]", unsqueeze_280: "f32[1, 256, 1, 1]", unsqueeze_292: "f32[1, 256, 1, 1]", unsqueeze_305: "f32[1, 32, 1, 1]", unsqueeze_330: "f32[1, 64, 1, 1]", unsqueeze_354: "f32[1, 32, 1, 1]", unsqueeze_366: "f32[1, 32, 1, 1]", tangents_1: "f32[32, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:772 in forward_head, code: return x if pre_logits else self.fc(x)
        permute_4: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        permute_5: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm: "f32[32, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_5);  permute_5 = None
        permute_6: "f32[1000, 32]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_6, view_24);  permute_6 = view_24 = None
        sum_13: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_25: "f32[1000]" = torch.ops.aten.reshape.default(sum_13, [1000]);  sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_26: "f32[32, 2048, 1, 1]" = torch.ops.aten.reshape.default(mm, [32, 2048, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        expand: "f32[32, 2048, 7, 7]" = torch.ops.aten.expand.default(view_26, [32, 2048, 7, 7]);  view_26 = None
        div_4: "f32[32, 2048, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        sub_25: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_45)
        mul_151: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_21);  sub_25 = None
        unsqueeze_84: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_85: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_157: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_85);  mul_151 = unsqueeze_85 = None
        unsqueeze_86: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_87: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_112: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_87);  mul_157 = unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        sub_26: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_47)
        mul_158: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_22);  sub_26 = None
        unsqueeze_88: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_89: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_164: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_158, unsqueeze_89);  mul_158 = unsqueeze_89 = None
        unsqueeze_90: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_151, -1);  primals_151 = None
        unsqueeze_91: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_117: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_91);  mul_164 = unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:141 in forward, code: out += shortcut
        add_118: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_112, add_117);  add_112 = add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:142 in forward, code: out = self.act3(out)
        relu_18: "f32[32, 2048, 7, 7]" = torch.ops.aten.relu.default(add_118);  add_118 = None
        le: "b8[32, 2048, 7, 7]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 2048, 7, 7]" = torch.ops.aten.where.self(le, full_default, div_4);  le = div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        squeeze_66: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        unsqueeze_92: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_93: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, 2);  unsqueeze_92 = None
        unsqueeze_94: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 3);  unsqueeze_93 = None
        sum_14: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_27: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_94);  convolution_26 = unsqueeze_94 = None
        mul_165: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_27)
        sum_15: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_165, [0, 2, 3]);  mul_165 = None
        mul_166: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_14, 0.0006377551020408163)
        unsqueeze_95: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_166, 0);  mul_166 = None
        unsqueeze_96: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
        unsqueeze_97: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 3);  unsqueeze_96 = None
        mul_167: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, 0.0006377551020408163)
        squeeze_67: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_168: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_169: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None
        unsqueeze_98: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_169, 0);  mul_169 = None
        unsqueeze_99: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, 2);  unsqueeze_98 = None
        unsqueeze_100: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 3);  unsqueeze_99 = None
        mul_170: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_150);  primals_150 = None
        unsqueeze_101: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_170, 0);  mul_170 = None
        unsqueeze_102: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None
        unsqueeze_103: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 3);  unsqueeze_102 = None
        mul_171: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_100);  sub_27 = unsqueeze_100 = None
        sub_29: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_171);  mul_171 = None
        sub_30: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_29, unsqueeze_97);  sub_29 = unsqueeze_97 = None
        mul_172: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_103);  sub_30 = unsqueeze_103 = None
        mul_173: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_67);  sum_15 = squeeze_67 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_172, avg_pool2d_5, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_172 = avg_pool2d_5 = primals_146 = None
        getitem_48: "f32[32, 1024, 7, 7]" = convolution_backward[0]
        getitem_49: "f32[2048, 1024, 1, 1]" = convolution_backward[1];  convolution_backward = None
        avg_pool2d_backward: "f32[32, 1024, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(getitem_48, relu_14, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        squeeze_63: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        unsqueeze_104: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_105: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, 2);  unsqueeze_104 = None
        unsqueeze_106: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 3);  unsqueeze_105 = None
        sum_16: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_31: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_106);  convolution_25 = unsqueeze_106 = None
        mul_174: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_31)
        sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 2, 3]);  mul_174 = None
        mul_175: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_16, 0.0006377551020408163)
        unsqueeze_107: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_175, 0);  mul_175 = None
        unsqueeze_108: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
        unsqueeze_109: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 3);  unsqueeze_108 = None
        mul_176: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
        squeeze_64: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_177: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_178: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        unsqueeze_110: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_178, 0);  mul_178 = None
        unsqueeze_111: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, 2);  unsqueeze_110 = None
        unsqueeze_112: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 3);  unsqueeze_111 = None
        mul_179: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_144);  primals_144 = None
        unsqueeze_113: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_179, 0);  mul_179 = None
        unsqueeze_114: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
        unsqueeze_115: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 3);  unsqueeze_114 = None
        mul_180: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_112);  sub_31 = unsqueeze_112 = None
        sub_33: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_180);  where = mul_180 = None
        sub_34: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_33, unsqueeze_109);  sub_33 = unsqueeze_109 = None
        mul_181: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_115);  sub_34 = unsqueeze_115 = None
        mul_182: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_64);  sum_17 = squeeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:132 in forward, code: out = self.conv3(out)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_181, avg_pool2d_4, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_181 = avg_pool2d_4 = primals_140 = None
        getitem_51: "f32[32, 512, 7, 7]" = convolution_backward_1[0]
        getitem_52: "f32[2048, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:130 in forward, code: out = self.avd_last(out)
        avg_pool2d_backward_1: "f32[32, 512, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(getitem_51, sum_12, [3, 3], [2, 2], [1, 1], False, True, None);  getitem_51 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        unsqueeze_116: "f32[32, 1, 512, 14, 14]" = torch.ops.aten.unsqueeze.default(avg_pool2d_backward_1, 1);  avg_pool2d_backward_1 = None
        expand_1: "f32[32, 2, 512, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_116, [32, 2, 512, 14, 14]);  unsqueeze_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        sub_22: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_41)
        mul_136: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_19);  sub_22 = None
        unsqueeze_76: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1)
        unsqueeze_77: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_142: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_77);  mul_136 = unsqueeze_77 = None
        unsqueeze_78: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1);  primals_130 = None
        unsqueeze_79: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_102: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_79);  mul_142 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        relu_16: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_102);  add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_19: "f32[32, 2, 512, 14, 14]" = torch.ops.aten.reshape.default(relu_16, [32, 2, 512, 14, 14])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        mul_183: "f32[32, 2, 512, 14, 14]" = torch.ops.aten.mul.Tensor(expand_1, view_19);  view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_20: "f32[32, 1, 2, 512]" = torch.ops.aten.reshape.default(convolution_24, [32, 1, 2, -1]);  convolution_24 = None
        permute_3: "f32[32, 2, 1, 512]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        amax_3: "f32[32, 1, 1, 512]" = torch.ops.aten.amax.default(permute_3, [1], True)
        sub_24: "f32[32, 2, 1, 512]" = torch.ops.aten.sub.Tensor(permute_3, amax_3);  permute_3 = amax_3 = None
        exp_3: "f32[32, 2, 1, 512]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_11: "f32[32, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(exp_3, [1], True)
        div_3: "f32[32, 2, 1, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_11);  exp_3 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_21: "f32[32, 1024]" = torch.ops.aten.reshape.default(div_3, [32, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_22: "f32[32, 1024, 1, 1]" = torch.ops.aten.reshape.default(view_21, [32, -1, 1, 1]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_23: "f32[32, 2, 512, 1, 1]" = torch.ops.aten.reshape.default(view_22, [32, 2, 512, 1, 1]);  view_22 = None
        mul_184: "f32[32, 2, 512, 14, 14]" = torch.ops.aten.mul.Tensor(expand_1, view_23);  expand_1 = view_23 = None
        sum_18: "f32[32, 2, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [3, 4], True);  mul_183 = None
        view_27: "f32[32, 1024, 1, 1]" = torch.ops.aten.reshape.default(sum_18, [32, 1024, 1, 1]);  sum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_28: "f32[32, 1024]" = torch.ops.aten.reshape.default(view_27, [32, 1024]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_29: "f32[32, 2, 1, 512]" = torch.ops.aten.reshape.default(view_28, [32, 2, 1, 512]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        mul_185: "f32[32, 2, 1, 512]" = torch.ops.aten.mul.Tensor(view_29, div_3);  view_29 = None
        sum_19: "f32[32, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_185, [1], True)
        neg: "f32[32, 2, 1, 512]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma: "f32[32, 2, 1, 512]" = torch.ops.prims.fma.default(neg, sum_19, mul_185);  neg = sum_19 = mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        permute_9: "f32[32, 1, 2, 512]" = torch.ops.aten.permute.default(fma, [0, 2, 1, 3]);  fma = None
        view_30: "f32[32, 1024, 1, 1]" = torch.ops.aten.reshape.default(permute_9, [32, 1024, 1, 1]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:105 in forward, code: x_attn = self.fc2(x_gap)
        sum_20: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_30, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(view_30, relu_17, primals_138, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_30 = primals_138 = None
        getitem_54: "f32[32, 256, 1, 1]" = convolution_backward_2[0]
        getitem_55: "f32[1024, 256, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:104 in forward, code: x_gap = self.act1(x_gap)
        le_1: "b8[32, 256, 1, 1]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_1: "f32[32, 256, 1, 1]" = torch.ops.aten.where.self(le_1, full_default, getitem_54);  le_1 = getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        sum_21: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_35: "f32[32, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_119);  convolution_23 = unsqueeze_119 = None
        mul_186: "f32[32, 256, 1, 1]" = torch.ops.aten.mul.Tensor(where_1, sub_35)
        sum_22: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_186, [0, 2, 3]);  mul_186 = None
        mul_187: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, 0.03125)
        unsqueeze_120: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_187, 0);  mul_187 = None
        unsqueeze_121: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None
        unsqueeze_122: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
        mul_188: "f32[256]" = torch.ops.aten.mul.Tensor(sum_22, 0.03125)
        mul_189: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_190: "f32[256]" = torch.ops.aten.mul.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
        unsqueeze_123: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_190, 0);  mul_190 = None
        unsqueeze_124: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
        unsqueeze_125: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 3);  unsqueeze_124 = None
        mul_191: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_136);  primals_136 = None
        unsqueeze_126: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_191, 0);  mul_191 = None
        unsqueeze_127: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 2);  unsqueeze_126 = None
        unsqueeze_128: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 3);  unsqueeze_127 = None
        mul_192: "f32[32, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_125);  sub_35 = unsqueeze_125 = None
        sub_37: "f32[32, 256, 1, 1]" = torch.ops.aten.sub.Tensor(where_1, mul_192);  where_1 = mul_192 = None
        sub_38: "f32[32, 256, 1, 1]" = torch.ops.aten.sub.Tensor(sub_37, unsqueeze_122);  sub_37 = unsqueeze_122 = None
        mul_193: "f32[32, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_128);  sub_38 = unsqueeze_128 = None
        mul_194: "f32[256]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_61);  sum_22 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:102 in forward, code: x_gap = self.fc1(x_gap)
        sum_23: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_193, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_193, mean_3, primals_131, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_193 = mean_3 = primals_131 = None
        getitem_57: "f32[32, 512, 1, 1]" = convolution_backward_3[0]
        getitem_58: "f32[256, 512, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:101 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        expand_2: "f32[32, 512, 14, 14]" = torch.ops.aten.expand.default(getitem_57, [32, 512, 14, 14]);  getitem_57 = None
        div_5: "f32[32, 512, 14, 14]" = torch.ops.aten.div.Scalar(expand_2, 196);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:98 in forward, code: x_gap = x.sum(dim=1)
        unsqueeze_129: "f32[32, 1, 512, 14, 14]" = torch.ops.aten.unsqueeze.default(div_5, 1);  div_5 = None
        expand_3: "f32[32, 2, 512, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_129, [32, 2, 512, 14, 14]);  unsqueeze_129 = None
        add_119: "f32[32, 2, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_184, expand_3);  mul_184 = expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_31: "f32[32, 1024, 14, 14]" = torch.ops.aten.reshape.default(add_119, [32, 1024, 14, 14]);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        le_2: "b8[32, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_2: "f32[32, 1024, 14, 14]" = torch.ops.aten.where.self(le_2, full_default, view_31);  le_2 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        squeeze_57: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        unsqueeze_130: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_131: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 2);  unsqueeze_130 = None
        unsqueeze_132: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 3);  unsqueeze_131 = None
        sum_24: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_39: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_132);  convolution_22 = unsqueeze_132 = None
        mul_195: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_2, sub_39)
        sum_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_195, [0, 2, 3]);  mul_195 = None
        mul_196: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_24, 0.00015943877551020407)
        unsqueeze_133: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_196, 0);  mul_196 = None
        unsqueeze_134: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
        unsqueeze_135: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 3);  unsqueeze_134 = None
        mul_197: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_25, 0.00015943877551020407)
        squeeze_58: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_198: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_199: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        unsqueeze_136: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_199, 0);  mul_199 = None
        unsqueeze_137: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 2);  unsqueeze_136 = None
        unsqueeze_138: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 3);  unsqueeze_137 = None
        mul_200: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_129);  primals_129 = None
        unsqueeze_139: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_200, 0);  mul_200 = None
        unsqueeze_140: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
        unsqueeze_141: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
        mul_201: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_138);  sub_39 = unsqueeze_138 = None
        sub_41: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_2, mul_201);  where_2 = mul_201 = None
        sub_42: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_41, unsqueeze_135);  sub_41 = unsqueeze_135 = None
        mul_202: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_141);  sub_42 = unsqueeze_141 = None
        mul_203: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_58);  sum_25 = squeeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:90 in forward, code: x = self.conv(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_202, relu_15, primals_125, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_202 = primals_125 = None
        getitem_60: "f32[32, 512, 14, 14]" = convolution_backward_4[0]
        getitem_61: "f32[1024, 256, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:119 in forward, code: out = self.act1(out)
        le_3: "b8[32, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_3: "f32[32, 512, 14, 14]" = torch.ops.aten.where.self(le_3, full_default, getitem_60);  le_3 = getitem_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_43: "f32[32, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_144);  convolution_21 = unsqueeze_144 = None
        mul_204: "f32[32, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_3, sub_43)
        sum_27: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 2, 3]);  mul_204 = None
        mul_205: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, 0.00015943877551020407)
        unsqueeze_145: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_205, 0);  mul_205 = None
        unsqueeze_146: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
        unsqueeze_147: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
        mul_206: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, 0.00015943877551020407)
        mul_207: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_208: "f32[512]" = torch.ops.aten.mul.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
        unsqueeze_148: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_208, 0);  mul_208 = None
        unsqueeze_149: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
        unsqueeze_150: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
        mul_209: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_123);  primals_123 = None
        unsqueeze_151: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_209, 0);  mul_209 = None
        unsqueeze_152: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
        unsqueeze_153: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
        mul_210: "f32[32, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_150);  sub_43 = unsqueeze_150 = None
        sub_45: "f32[32, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_3, mul_210);  where_3 = mul_210 = None
        sub_46: "f32[32, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_147);  sub_45 = unsqueeze_147 = None
        mul_211: "f32[32, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_153);  sub_46 = unsqueeze_153 = None
        mul_212: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_55);  sum_27 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:117 in forward, code: out = self.conv1(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_211, relu_14, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_211 = primals_119 = None
        getitem_63: "f32[32, 1024, 14, 14]" = convolution_backward_5[0]
        getitem_64: "f32[512, 1024, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
        add_120: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(avg_pool2d_backward, getitem_63);  avg_pool2d_backward = getitem_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:142 in forward, code: out = self.act3(out)
        le_4: "b8[32, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_4: "f32[32, 1024, 14, 14]" = torch.ops.aten.where.self(le_4, full_default, add_120);  le_4 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        sum_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_47: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_156);  convolution_20 = unsqueeze_156 = None
        mul_213: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_47)
        sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 2, 3]);  mul_213 = None
        mul_214: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_28, 0.00015943877551020407)
        unsqueeze_157: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_214, 0);  mul_214 = None
        unsqueeze_158: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
        unsqueeze_159: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
        mul_215: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, 0.00015943877551020407)
        mul_216: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_217: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_215, mul_216);  mul_215 = mul_216 = None
        unsqueeze_160: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_217, 0);  mul_217 = None
        unsqueeze_161: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
        unsqueeze_162: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
        mul_218: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_117);  primals_117 = None
        unsqueeze_163: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_218, 0);  mul_218 = None
        unsqueeze_164: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        unsqueeze_165: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        mul_219: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_162);  sub_47 = unsqueeze_162 = None
        sub_49: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_4, mul_219);  mul_219 = None
        sub_50: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_49, unsqueeze_159);  sub_49 = unsqueeze_159 = None
        mul_220: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_165);  sub_50 = unsqueeze_165 = None
        mul_221: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_52);  sum_29 = squeeze_52 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_220, avg_pool2d_3, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_220 = avg_pool2d_3 = primals_113 = None
        getitem_66: "f32[32, 512, 14, 14]" = convolution_backward_6[0]
        getitem_67: "f32[1024, 512, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
        avg_pool2d_backward_2: "f32[32, 512, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(getitem_66, relu_10, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        sum_30: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_51: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_168);  convolution_19 = unsqueeze_168 = None
        mul_222: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_51)
        sum_31: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_222, [0, 2, 3]);  mul_222 = None
        mul_223: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_30, 0.00015943877551020407)
        unsqueeze_169: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_223, 0);  mul_223 = None
        unsqueeze_170: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
        unsqueeze_171: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
        mul_224: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_31, 0.00015943877551020407)
        mul_225: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_226: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
        unsqueeze_172: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_226, 0);  mul_226 = None
        unsqueeze_173: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
        unsqueeze_174: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
        mul_227: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_111);  primals_111 = None
        unsqueeze_175: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_227, 0);  mul_227 = None
        unsqueeze_176: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
        unsqueeze_177: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
        mul_228: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_174);  sub_51 = unsqueeze_174 = None
        sub_53: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_4, mul_228);  where_4 = mul_228 = None
        sub_54: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_171);  sub_53 = unsqueeze_171 = None
        mul_229: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_177);  sub_54 = unsqueeze_177 = None
        mul_230: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_49);  sum_31 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:132 in forward, code: out = self.conv3(out)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_229, avg_pool2d_2, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_229 = avg_pool2d_2 = primals_107 = None
        getitem_69: "f32[32, 256, 14, 14]" = convolution_backward_7[0]
        getitem_70: "f32[1024, 256, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:130 in forward, code: out = self.avd_last(out)
        avg_pool2d_backward_3: "f32[32, 256, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(getitem_69, sum_9, [3, 3], [2, 2], [1, 1], False, True, None);  getitem_69 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        unsqueeze_178: "f32[32, 1, 256, 28, 28]" = torch.ops.aten.unsqueeze.default(avg_pool2d_backward_3, 1);  avg_pool2d_backward_3 = None
        expand_4: "f32[32, 2, 256, 28, 28]" = torch.ops.aten.expand.default(unsqueeze_178, [32, 2, 256, 28, 28]);  unsqueeze_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        sub_16: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_31)
        mul_100: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = None
        unsqueeze_56: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_57: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_106: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_57);  mul_100 = unsqueeze_57 = None
        unsqueeze_58: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_59: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_76: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_59);  mul_106 = unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        relu_12: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_76);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_13: "f32[32, 2, 256, 28, 28]" = torch.ops.aten.reshape.default(relu_12, [32, 2, 256, 28, 28])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        mul_231: "f32[32, 2, 256, 28, 28]" = torch.ops.aten.mul.Tensor(expand_4, view_13);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_14: "f32[32, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_18, [32, 1, 2, -1]);  convolution_18 = None
        permute_2: "f32[32, 2, 1, 256]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        amax_2: "f32[32, 1, 1, 256]" = torch.ops.aten.amax.default(permute_2, [1], True)
        sub_18: "f32[32, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_2, amax_2);  permute_2 = amax_2 = None
        exp_2: "f32[32, 2, 1, 256]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_8: "f32[32, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True)
        div_2: "f32[32, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_2, sum_8);  exp_2 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_15: "f32[32, 512]" = torch.ops.aten.reshape.default(div_2, [32, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_16: "f32[32, 512, 1, 1]" = torch.ops.aten.reshape.default(view_15, [32, -1, 1, 1]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_17: "f32[32, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_16, [32, 2, 256, 1, 1]);  view_16 = None
        mul_232: "f32[32, 2, 256, 28, 28]" = torch.ops.aten.mul.Tensor(expand_4, view_17);  expand_4 = view_17 = None
        sum_32: "f32[32, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [3, 4], True);  mul_231 = None
        view_32: "f32[32, 512, 1, 1]" = torch.ops.aten.reshape.default(sum_32, [32, 512, 1, 1]);  sum_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_33: "f32[32, 512]" = torch.ops.aten.reshape.default(view_32, [32, 512]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_34: "f32[32, 2, 1, 256]" = torch.ops.aten.reshape.default(view_33, [32, 2, 1, 256]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        mul_233: "f32[32, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_34, div_2);  view_34 = None
        sum_33: "f32[32, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_233, [1], True)
        neg_1: "f32[32, 2, 1, 256]" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_1: "f32[32, 2, 1, 256]" = torch.ops.prims.fma.default(neg_1, sum_33, mul_233);  neg_1 = sum_33 = mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        permute_10: "f32[32, 1, 2, 256]" = torch.ops.aten.permute.default(fma_1, [0, 2, 1, 3]);  fma_1 = None
        view_35: "f32[32, 512, 1, 1]" = torch.ops.aten.reshape.default(permute_10, [32, 512, 1, 1]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:105 in forward, code: x_attn = self.fc2(x_gap)
        sum_34: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_35, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(view_35, relu_13, primals_105, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_35 = primals_105 = None
        getitem_72: "f32[32, 128, 1, 1]" = convolution_backward_8[0]
        getitem_73: "f32[512, 128, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:104 in forward, code: x_gap = self.act1(x_gap)
        le_5: "b8[32, 128, 1, 1]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_5: "f32[32, 128, 1, 1]" = torch.ops.aten.where.self(le_5, full_default, getitem_72);  le_5 = getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        sum_35: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_55: "f32[32, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_181);  convolution_17 = unsqueeze_181 = None
        mul_234: "f32[32, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_5, sub_55)
        sum_36: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 2, 3]);  mul_234 = None
        mul_235: "f32[128]" = torch.ops.aten.mul.Tensor(sum_35, 0.03125)
        unsqueeze_182: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_235, 0);  mul_235 = None
        unsqueeze_183: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
        unsqueeze_184: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
        mul_236: "f32[128]" = torch.ops.aten.mul.Tensor(sum_36, 0.03125)
        mul_237: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_238: "f32[128]" = torch.ops.aten.mul.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
        unsqueeze_185: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_238, 0);  mul_238 = None
        unsqueeze_186: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
        unsqueeze_187: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
        mul_239: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_103);  primals_103 = None
        unsqueeze_188: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_239, 0);  mul_239 = None
        unsqueeze_189: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
        unsqueeze_190: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
        mul_240: "f32[32, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_187);  sub_55 = unsqueeze_187 = None
        sub_57: "f32[32, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_5, mul_240);  where_5 = mul_240 = None
        sub_58: "f32[32, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_184);  sub_57 = unsqueeze_184 = None
        mul_241: "f32[32, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_190);  sub_58 = unsqueeze_190 = None
        mul_242: "f32[128]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_46);  sum_36 = squeeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:102 in forward, code: x_gap = self.fc1(x_gap)
        sum_37: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 2, 3])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_241, mean_2, primals_98, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_241 = mean_2 = primals_98 = None
        getitem_75: "f32[32, 256, 1, 1]" = convolution_backward_9[0]
        getitem_76: "f32[128, 256, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:101 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        expand_5: "f32[32, 256, 28, 28]" = torch.ops.aten.expand.default(getitem_75, [32, 256, 28, 28]);  getitem_75 = None
        div_6: "f32[32, 256, 28, 28]" = torch.ops.aten.div.Scalar(expand_5, 784);  expand_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:98 in forward, code: x_gap = x.sum(dim=1)
        unsqueeze_191: "f32[32, 1, 256, 28, 28]" = torch.ops.aten.unsqueeze.default(div_6, 1);  div_6 = None
        expand_6: "f32[32, 2, 256, 28, 28]" = torch.ops.aten.expand.default(unsqueeze_191, [32, 2, 256, 28, 28]);  unsqueeze_191 = None
        add_121: "f32[32, 2, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_232, expand_6);  mul_232 = expand_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_36: "f32[32, 512, 28, 28]" = torch.ops.aten.reshape.default(add_121, [32, 512, 28, 28]);  add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        le_6: "b8[32, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_6: "f32[32, 512, 28, 28]" = torch.ops.aten.where.self(le_6, full_default, view_36);  le_6 = view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        squeeze_42: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        unsqueeze_192: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_193: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 2);  unsqueeze_192 = None
        unsqueeze_194: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 3);  unsqueeze_193 = None
        sum_38: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_59: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_194);  convolution_16 = unsqueeze_194 = None
        mul_243: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_6, sub_59)
        sum_39: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 2, 3]);  mul_243 = None
        mul_244: "f32[512]" = torch.ops.aten.mul.Tensor(sum_38, 3.985969387755102e-05)
        unsqueeze_195: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_244, 0);  mul_244 = None
        unsqueeze_196: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
        unsqueeze_197: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 3);  unsqueeze_196 = None
        mul_245: "f32[512]" = torch.ops.aten.mul.Tensor(sum_39, 3.985969387755102e-05)
        squeeze_43: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_246: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_247: "f32[512]" = torch.ops.aten.mul.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
        unsqueeze_198: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_247, 0);  mul_247 = None
        unsqueeze_199: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 2);  unsqueeze_198 = None
        unsqueeze_200: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 3);  unsqueeze_199 = None
        mul_248: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_96);  primals_96 = None
        unsqueeze_201: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_248, 0);  mul_248 = None
        unsqueeze_202: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
        unsqueeze_203: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 3);  unsqueeze_202 = None
        mul_249: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_200);  sub_59 = unsqueeze_200 = None
        sub_61: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_6, mul_249);  where_6 = mul_249 = None
        sub_62: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_197);  sub_61 = unsqueeze_197 = None
        mul_250: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_203);  sub_62 = unsqueeze_203 = None
        mul_251: "f32[512]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_43);  sum_39 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:90 in forward, code: x = self.conv(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_250, relu_11, primals_92, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_250 = primals_92 = None
        getitem_78: "f32[32, 256, 28, 28]" = convolution_backward_10[0]
        getitem_79: "f32[512, 128, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:119 in forward, code: out = self.act1(out)
        le_7: "b8[32, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_7: "f32[32, 256, 28, 28]" = torch.ops.aten.where.self(le_7, full_default, getitem_78);  le_7 = getitem_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_63: "f32[32, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_206);  convolution_15 = unsqueeze_206 = None
        mul_252: "f32[32, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_7, sub_63)
        sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 2, 3]);  mul_252 = None
        mul_253: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, 3.985969387755102e-05)
        unsqueeze_207: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_253, 0);  mul_253 = None
        unsqueeze_208: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
        unsqueeze_209: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 3);  unsqueeze_208 = None
        mul_254: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 3.985969387755102e-05)
        mul_255: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_256: "f32[256]" = torch.ops.aten.mul.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
        unsqueeze_210: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_256, 0);  mul_256 = None
        unsqueeze_211: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 2);  unsqueeze_210 = None
        unsqueeze_212: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 3);  unsqueeze_211 = None
        mul_257: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_90);  primals_90 = None
        unsqueeze_213: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_257, 0);  mul_257 = None
        unsqueeze_214: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
        unsqueeze_215: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 3);  unsqueeze_214 = None
        mul_258: "f32[32, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_212);  sub_63 = unsqueeze_212 = None
        sub_65: "f32[32, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_7, mul_258);  where_7 = mul_258 = None
        sub_66: "f32[32, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_209);  sub_65 = unsqueeze_209 = None
        mul_259: "f32[32, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_215);  sub_66 = unsqueeze_215 = None
        mul_260: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_40);  sum_41 = squeeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:117 in forward, code: out = self.conv1(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_259, relu_10, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_259 = primals_86 = None
        getitem_81: "f32[32, 512, 28, 28]" = convolution_backward_11[0]
        getitem_82: "f32[256, 512, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
        add_122: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_2, getitem_81);  avg_pool2d_backward_2 = getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:142 in forward, code: out = self.act3(out)
        le_8: "b8[32, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_8: "f32[32, 512, 28, 28]" = torch.ops.aten.where.self(le_8, full_default, add_122);  le_8 = add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        sum_42: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_67: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_218);  convolution_14 = unsqueeze_218 = None
        mul_261: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_8, sub_67)
        sum_43: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 2, 3]);  mul_261 = None
        mul_262: "f32[512]" = torch.ops.aten.mul.Tensor(sum_42, 3.985969387755102e-05)
        unsqueeze_219: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_262, 0);  mul_262 = None
        unsqueeze_220: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
        unsqueeze_221: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 3);  unsqueeze_220 = None
        mul_263: "f32[512]" = torch.ops.aten.mul.Tensor(sum_43, 3.985969387755102e-05)
        mul_264: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_265: "f32[512]" = torch.ops.aten.mul.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_222: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_265, 0);  mul_265 = None
        unsqueeze_223: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
        unsqueeze_224: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 3);  unsqueeze_223 = None
        mul_266: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_84);  primals_84 = None
        unsqueeze_225: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_266, 0);  mul_266 = None
        unsqueeze_226: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
        unsqueeze_227: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 3);  unsqueeze_226 = None
        mul_267: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_224);  sub_67 = unsqueeze_224 = None
        sub_69: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_8, mul_267);  mul_267 = None
        sub_70: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_221);  sub_69 = unsqueeze_221 = None
        mul_268: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_227);  sub_70 = unsqueeze_227 = None
        mul_269: "f32[512]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_37);  sum_43 = squeeze_37 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_268, avg_pool2d_1, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_268 = avg_pool2d_1 = primals_80 = None
        getitem_84: "f32[32, 256, 28, 28]" = convolution_backward_12[0]
        getitem_85: "f32[512, 256, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
        avg_pool2d_backward_4: "f32[32, 256, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(getitem_84, relu_6, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        sum_44: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_71: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_230);  convolution_13 = unsqueeze_230 = None
        mul_270: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_8, sub_71)
        sum_45: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 2, 3]);  mul_270 = None
        mul_271: "f32[512]" = torch.ops.aten.mul.Tensor(sum_44, 3.985969387755102e-05)
        unsqueeze_231: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
        unsqueeze_232: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
        unsqueeze_233: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
        mul_272: "f32[512]" = torch.ops.aten.mul.Tensor(sum_45, 3.985969387755102e-05)
        mul_273: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_274: "f32[512]" = torch.ops.aten.mul.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
        unsqueeze_234: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
        unsqueeze_235: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
        unsqueeze_236: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
        mul_275: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_78);  primals_78 = None
        unsqueeze_237: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_275, 0);  mul_275 = None
        unsqueeze_238: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
        unsqueeze_239: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
        mul_276: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_236);  sub_71 = unsqueeze_236 = None
        sub_73: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_8, mul_276);  where_8 = mul_276 = None
        sub_74: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_233);  sub_73 = unsqueeze_233 = None
        mul_277: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_239);  sub_74 = unsqueeze_239 = None
        mul_278: "f32[512]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_34);  sum_45 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:132 in forward, code: out = self.conv3(out)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_277, avg_pool2d, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_277 = avg_pool2d = primals_74 = None
        getitem_87: "f32[32, 128, 28, 28]" = convolution_backward_13[0]
        getitem_88: "f32[512, 128, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:130 in forward, code: out = self.avd_last(out)
        avg_pool2d_backward_5: "f32[32, 128, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(getitem_87, sum_6, [3, 3], [2, 2], [1, 1], False, True, None);  getitem_87 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        unsqueeze_240: "f32[32, 1, 128, 56, 56]" = torch.ops.aten.unsqueeze.default(avg_pool2d_backward_5, 1);  avg_pool2d_backward_5 = None
        expand_7: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.expand.default(unsqueeze_240, [32, 2, 128, 56, 56]);  unsqueeze_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        sub_10: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_64: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = None
        unsqueeze_36: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
        unsqueeze_37: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_70: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_37);  mul_64 = unsqueeze_37 = None
        unsqueeze_38: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
        unsqueeze_39: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_50: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_39);  mul_70 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        relu_8: "f32[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_50);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_7: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.reshape.default(relu_8, [32, 2, 128, 56, 56])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        mul_279: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.mul.Tensor(expand_7, view_7);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_8: "f32[32, 1, 2, 128]" = torch.ops.aten.reshape.default(convolution_12, [32, 1, 2, -1]);  convolution_12 = None
        permute_1: "f32[32, 2, 1, 128]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        amax_1: "f32[32, 1, 1, 128]" = torch.ops.aten.amax.default(permute_1, [1], True)
        sub_12: "f32[32, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_1, amax_1);  permute_1 = amax_1 = None
        exp_1: "f32[32, 2, 1, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_5: "f32[32, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True)
        div_1: "f32[32, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_5);  exp_1 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_9: "f32[32, 256]" = torch.ops.aten.reshape.default(div_1, [32, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_10: "f32[32, 256, 1, 1]" = torch.ops.aten.reshape.default(view_9, [32, -1, 1, 1]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_11: "f32[32, 2, 128, 1, 1]" = torch.ops.aten.reshape.default(view_10, [32, 2, 128, 1, 1]);  view_10 = None
        mul_280: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.mul.Tensor(expand_7, view_11);  expand_7 = view_11 = None
        sum_46: "f32[32, 2, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [3, 4], True);  mul_279 = None
        view_37: "f32[32, 256, 1, 1]" = torch.ops.aten.reshape.default(sum_46, [32, 256, 1, 1]);  sum_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_38: "f32[32, 256]" = torch.ops.aten.reshape.default(view_37, [32, 256]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_39: "f32[32, 2, 1, 128]" = torch.ops.aten.reshape.default(view_38, [32, 2, 1, 128]);  view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        mul_281: "f32[32, 2, 1, 128]" = torch.ops.aten.mul.Tensor(view_39, div_1);  view_39 = None
        sum_47: "f32[32, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_281, [1], True)
        neg_2: "f32[32, 2, 1, 128]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_2: "f32[32, 2, 1, 128]" = torch.ops.prims.fma.default(neg_2, sum_47, mul_281);  neg_2 = sum_47 = mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        permute_11: "f32[32, 1, 2, 128]" = torch.ops.aten.permute.default(fma_2, [0, 2, 1, 3]);  fma_2 = None
        view_40: "f32[32, 256, 1, 1]" = torch.ops.aten.reshape.default(permute_11, [32, 256, 1, 1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:105 in forward, code: x_attn = self.fc2(x_gap)
        sum_48: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_40, [0, 2, 3])
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(view_40, relu_9, primals_72, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_40 = primals_72 = None
        getitem_90: "f32[32, 64, 1, 1]" = convolution_backward_14[0]
        getitem_91: "f32[256, 64, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:104 in forward, code: x_gap = self.act1(x_gap)
        le_9: "b8[32, 64, 1, 1]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_9: "f32[32, 64, 1, 1]" = torch.ops.aten.where.self(le_9, full_default, getitem_90);  le_9 = getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        sum_49: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_75: "f32[32, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_243);  convolution_11 = unsqueeze_243 = None
        mul_282: "f32[32, 64, 1, 1]" = torch.ops.aten.mul.Tensor(where_9, sub_75)
        sum_50: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3]);  mul_282 = None
        mul_283: "f32[64]" = torch.ops.aten.mul.Tensor(sum_49, 0.03125)
        unsqueeze_244: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_283, 0);  mul_283 = None
        unsqueeze_245: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
        unsqueeze_246: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
        mul_284: "f32[64]" = torch.ops.aten.mul.Tensor(sum_50, 0.03125)
        mul_285: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_286: "f32[64]" = torch.ops.aten.mul.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_247: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
        unsqueeze_248: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_287: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_70);  primals_70 = None
        unsqueeze_250: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
        unsqueeze_251: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_288: "f32[32, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_249);  sub_75 = unsqueeze_249 = None
        sub_77: "f32[32, 64, 1, 1]" = torch.ops.aten.sub.Tensor(where_9, mul_288);  where_9 = mul_288 = None
        sub_78: "f32[32, 64, 1, 1]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_246);  sub_77 = unsqueeze_246 = None
        mul_289: "f32[32, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_252);  sub_78 = unsqueeze_252 = None
        mul_290: "f32[64]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_31);  sum_50 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:102 in forward, code: x_gap = self.fc1(x_gap)
        sum_51: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 2, 3])
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_289, mean_1, primals_65, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_289 = mean_1 = primals_65 = None
        getitem_93: "f32[32, 128, 1, 1]" = convolution_backward_15[0]
        getitem_94: "f32[64, 128, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:101 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        expand_8: "f32[32, 128, 56, 56]" = torch.ops.aten.expand.default(getitem_93, [32, 128, 56, 56]);  getitem_93 = None
        div_7: "f32[32, 128, 56, 56]" = torch.ops.aten.div.Scalar(expand_8, 3136);  expand_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:98 in forward, code: x_gap = x.sum(dim=1)
        unsqueeze_253: "f32[32, 1, 128, 56, 56]" = torch.ops.aten.unsqueeze.default(div_7, 1);  div_7 = None
        expand_9: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.expand.default(unsqueeze_253, [32, 2, 128, 56, 56]);  unsqueeze_253 = None
        add_123: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_280, expand_9);  mul_280 = expand_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_41: "f32[32, 256, 56, 56]" = torch.ops.aten.reshape.default(add_123, [32, 256, 56, 56]);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        le_10: "b8[32, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_10: "f32[32, 256, 56, 56]" = torch.ops.aten.where.self(le_10, full_default, view_41);  le_10 = view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        squeeze_27: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        unsqueeze_254: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_255: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
        unsqueeze_256: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
        sum_52: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_79: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_256);  convolution_10 = unsqueeze_256 = None
        mul_291: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_10, sub_79)
        sum_53: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 2, 3]);  mul_291 = None
        mul_292: "f32[256]" = torch.ops.aten.mul.Tensor(sum_52, 9.964923469387754e-06)
        unsqueeze_257: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_292, 0);  mul_292 = None
        unsqueeze_258: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
        unsqueeze_259: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
        mul_293: "f32[256]" = torch.ops.aten.mul.Tensor(sum_53, 9.964923469387754e-06)
        squeeze_28: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_294: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_295: "f32[256]" = torch.ops.aten.mul.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
        unsqueeze_260: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
        unsqueeze_261: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
        unsqueeze_262: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
        mul_296: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_63);  primals_63 = None
        unsqueeze_263: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_296, 0);  mul_296 = None
        unsqueeze_264: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        unsqueeze_265: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
        mul_297: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_262);  sub_79 = unsqueeze_262 = None
        sub_81: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_10, mul_297);  where_10 = mul_297 = None
        sub_82: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_259);  sub_81 = unsqueeze_259 = None
        mul_298: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_265);  sub_82 = unsqueeze_265 = None
        mul_299: "f32[256]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_28);  sum_53 = squeeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:90 in forward, code: x = self.conv(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_298, relu_7, primals_59, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_298 = primals_59 = None
        getitem_96: "f32[32, 128, 56, 56]" = convolution_backward_16[0]
        getitem_97: "f32[256, 64, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:119 in forward, code: out = self.act1(out)
        le_11: "b8[32, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_11: "f32[32, 128, 56, 56]" = torch.ops.aten.where.self(le_11, full_default, getitem_96);  le_11 = getitem_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        sum_54: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_83: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_268);  convolution_9 = unsqueeze_268 = None
        mul_300: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_11, sub_83)
        sum_55: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 2, 3]);  mul_300 = None
        mul_301: "f32[128]" = torch.ops.aten.mul.Tensor(sum_54, 9.964923469387754e-06)
        unsqueeze_269: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
        unsqueeze_270: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
        unsqueeze_271: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
        mul_302: "f32[128]" = torch.ops.aten.mul.Tensor(sum_55, 9.964923469387754e-06)
        mul_303: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_304: "f32[128]" = torch.ops.aten.mul.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        unsqueeze_272: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_304, 0);  mul_304 = None
        unsqueeze_273: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
        unsqueeze_274: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
        mul_305: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_57);  primals_57 = None
        unsqueeze_275: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_305, 0);  mul_305 = None
        unsqueeze_276: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
        unsqueeze_277: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
        mul_306: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_274);  sub_83 = unsqueeze_274 = None
        sub_85: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_11, mul_306);  where_11 = mul_306 = None
        sub_86: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_271);  sub_85 = unsqueeze_271 = None
        mul_307: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_277);  sub_86 = unsqueeze_277 = None
        mul_308: "f32[128]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_25);  sum_55 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:117 in forward, code: out = self.conv1(x)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_307, relu_6, primals_53, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_307 = primals_53 = None
        getitem_99: "f32[32, 256, 56, 56]" = convolution_backward_17[0]
        getitem_100: "f32[128, 256, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
        add_124: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_4, getitem_99);  avg_pool2d_backward_4 = getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:142 in forward, code: out = self.act3(out)
        le_12: "b8[32, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_12: "f32[32, 256, 56, 56]" = torch.ops.aten.where.self(le_12, full_default, add_124);  le_12 = add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        sum_56: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_87: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_280);  convolution_8 = unsqueeze_280 = None
        mul_309: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_12, sub_87)
        sum_57: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 2, 3]);  mul_309 = None
        mul_310: "f32[256]" = torch.ops.aten.mul.Tensor(sum_56, 9.964923469387754e-06)
        unsqueeze_281: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_310, 0);  mul_310 = None
        unsqueeze_282: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
        unsqueeze_283: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
        mul_311: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, 9.964923469387754e-06)
        mul_312: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_313: "f32[256]" = torch.ops.aten.mul.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
        unsqueeze_284: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_313, 0);  mul_313 = None
        unsqueeze_285: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
        unsqueeze_286: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
        mul_314: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_51);  primals_51 = None
        unsqueeze_287: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
        unsqueeze_288: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
        unsqueeze_289: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
        mul_315: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_286);  sub_87 = unsqueeze_286 = None
        sub_89: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_12, mul_315);  mul_315 = None
        sub_90: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_283);  sub_89 = unsqueeze_283 = None
        mul_316: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_289);  sub_90 = unsqueeze_289 = None
        mul_317: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_22);  sum_57 = squeeze_22 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_316, getitem_6, primals_47, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_316 = primals_47 = None
        getitem_102: "f32[32, 64, 56, 56]" = convolution_backward_18[0]
        getitem_103: "f32[256, 64, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        sum_58: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_91: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_292);  convolution_7 = unsqueeze_292 = None
        mul_318: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_12, sub_91)
        sum_59: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 2, 3]);  mul_318 = None
        mul_319: "f32[256]" = torch.ops.aten.mul.Tensor(sum_58, 9.964923469387754e-06)
        unsqueeze_293: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
        unsqueeze_294: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
        unsqueeze_295: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
        mul_320: "f32[256]" = torch.ops.aten.mul.Tensor(sum_59, 9.964923469387754e-06)
        mul_321: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_322: "f32[256]" = torch.ops.aten.mul.Tensor(mul_320, mul_321);  mul_320 = mul_321 = None
        unsqueeze_296: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
        unsqueeze_297: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
        unsqueeze_298: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
        mul_323: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_45);  primals_45 = None
        unsqueeze_299: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_323, 0);  mul_323 = None
        unsqueeze_300: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
        unsqueeze_301: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
        mul_324: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_298);  sub_91 = unsqueeze_298 = None
        sub_93: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_12, mul_324);  where_12 = mul_324 = None
        sub_94: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_295);  sub_93 = unsqueeze_295 = None
        mul_325: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_301);  sub_94 = unsqueeze_301 = None
        mul_326: "f32[256]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_19);  sum_59 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:132 in forward, code: out = self.conv3(out)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_325, sum_3, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_325 = sum_3 = primals_41 = None
        getitem_105: "f32[32, 64, 56, 56]" = convolution_backward_19[0]
        getitem_106: "f32[256, 64, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        unsqueeze_302: "f32[32, 1, 64, 56, 56]" = torch.ops.aten.unsqueeze.default(getitem_105, 1);  getitem_105 = None
        expand_10: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.expand.default(unsqueeze_302, [32, 2, 64, 56, 56]);  unsqueeze_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        sub_4: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_28: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        relu_4: "f32[32, 128, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_1: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.reshape.default(relu_4, [32, 2, 64, 56, 56])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        mul_327: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.mul.Tensor(expand_10, view_1);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_2: "f32[32, 1, 2, 64]" = torch.ops.aten.reshape.default(convolution_6, [32, 1, 2, -1]);  convolution_6 = None
        permute: "f32[32, 2, 1, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        amax: "f32[32, 1, 1, 64]" = torch.ops.aten.amax.default(permute, [1], True)
        sub_6: "f32[32, 2, 1, 64]" = torch.ops.aten.sub.Tensor(permute, amax);  permute = amax = None
        exp: "f32[32, 2, 1, 64]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_2: "f32[32, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[32, 2, 1, 64]" = torch.ops.aten.div.Tensor(exp, sum_2);  exp = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_3: "f32[32, 128]" = torch.ops.aten.reshape.default(div, [32, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_4: "f32[32, 128, 1, 1]" = torch.ops.aten.reshape.default(view_3, [32, -1, 1, 1]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_5: "f32[32, 2, 64, 1, 1]" = torch.ops.aten.reshape.default(view_4, [32, 2, 64, 1, 1]);  view_4 = None
        mul_328: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.mul.Tensor(expand_10, view_5);  expand_10 = view_5 = None
        sum_60: "f32[32, 2, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [3, 4], True);  mul_327 = None
        view_42: "f32[32, 128, 1, 1]" = torch.ops.aten.reshape.default(sum_60, [32, 128, 1, 1]);  sum_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_43: "f32[32, 128]" = torch.ops.aten.reshape.default(view_42, [32, 128]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_44: "f32[32, 2, 1, 64]" = torch.ops.aten.reshape.default(view_43, [32, 2, 1, 64]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        mul_329: "f32[32, 2, 1, 64]" = torch.ops.aten.mul.Tensor(view_44, div);  view_44 = None
        sum_61: "f32[32, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(mul_329, [1], True)
        neg_3: "f32[32, 2, 1, 64]" = torch.ops.aten.neg.default(div);  div = None
        fma_3: "f32[32, 2, 1, 64]" = torch.ops.prims.fma.default(neg_3, sum_61, mul_329);  neg_3 = sum_61 = mul_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        permute_12: "f32[32, 1, 2, 64]" = torch.ops.aten.permute.default(fma_3, [0, 2, 1, 3]);  fma_3 = None
        view_45: "f32[32, 128, 1, 1]" = torch.ops.aten.reshape.default(permute_12, [32, 128, 1, 1]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:105 in forward, code: x_attn = self.fc2(x_gap)
        sum_62: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_45, [0, 2, 3])
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(view_45, relu_5, primals_39, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_45 = primals_39 = None
        getitem_108: "f32[32, 32, 1, 1]" = convolution_backward_20[0]
        getitem_109: "f32[128, 32, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:104 in forward, code: x_gap = self.act1(x_gap)
        le_13: "b8[32, 32, 1, 1]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_13: "f32[32, 32, 1, 1]" = torch.ops.aten.where.self(le_13, full_default, getitem_108);  le_13 = getitem_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        sum_63: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_95: "f32[32, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_305);  convolution_5 = unsqueeze_305 = None
        mul_330: "f32[32, 32, 1, 1]" = torch.ops.aten.mul.Tensor(where_13, sub_95)
        sum_64: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 2, 3]);  mul_330 = None
        mul_331: "f32[32]" = torch.ops.aten.mul.Tensor(sum_63, 0.03125)
        unsqueeze_306: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
        unsqueeze_307: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 2);  unsqueeze_306 = None
        unsqueeze_308: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 3);  unsqueeze_307 = None
        mul_332: "f32[32]" = torch.ops.aten.mul.Tensor(sum_64, 0.03125)
        mul_333: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_334: "f32[32]" = torch.ops.aten.mul.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
        unsqueeze_309: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
        unsqueeze_310: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
        unsqueeze_311: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
        mul_335: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_37);  primals_37 = None
        unsqueeze_312: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_335, 0);  mul_335 = None
        unsqueeze_313: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
        unsqueeze_314: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
        mul_336: "f32[32, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_311);  sub_95 = unsqueeze_311 = None
        sub_97: "f32[32, 32, 1, 1]" = torch.ops.aten.sub.Tensor(where_13, mul_336);  where_13 = mul_336 = None
        sub_98: "f32[32, 32, 1, 1]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_308);  sub_97 = unsqueeze_308 = None
        mul_337: "f32[32, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_314);  sub_98 = unsqueeze_314 = None
        mul_338: "f32[32]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_16);  sum_64 = squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:102 in forward, code: x_gap = self.fc1(x_gap)
        sum_65: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 2, 3])
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_337, mean, primals_32, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_337 = mean = primals_32 = None
        getitem_111: "f32[32, 64, 1, 1]" = convolution_backward_21[0]
        getitem_112: "f32[32, 64, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:101 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        expand_11: "f32[32, 64, 56, 56]" = torch.ops.aten.expand.default(getitem_111, [32, 64, 56, 56]);  getitem_111 = None
        div_8: "f32[32, 64, 56, 56]" = torch.ops.aten.div.Scalar(expand_11, 3136);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:98 in forward, code: x_gap = x.sum(dim=1)
        unsqueeze_315: "f32[32, 1, 64, 56, 56]" = torch.ops.aten.unsqueeze.default(div_8, 1);  div_8 = None
        expand_12: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.expand.default(unsqueeze_315, [32, 2, 64, 56, 56]);  unsqueeze_315 = None
        add_125: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_328, expand_12);  mul_328 = expand_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_46: "f32[32, 128, 56, 56]" = torch.ops.aten.reshape.default(add_125, [32, 128, 56, 56]);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        le_14: "b8[32, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_14: "f32[32, 128, 56, 56]" = torch.ops.aten.where.self(le_14, full_default, view_46);  le_14 = view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        squeeze_12: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        unsqueeze_316: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_317: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
        unsqueeze_318: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
        sum_66: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_99: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_318);  convolution_4 = unsqueeze_318 = None
        mul_339: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_14, sub_99)
        sum_67: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_339, [0, 2, 3]);  mul_339 = None
        mul_340: "f32[128]" = torch.ops.aten.mul.Tensor(sum_66, 9.964923469387754e-06)
        unsqueeze_319: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_340, 0);  mul_340 = None
        unsqueeze_320: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        unsqueeze_321: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        mul_341: "f32[128]" = torch.ops.aten.mul.Tensor(sum_67, 9.964923469387754e-06)
        squeeze_13: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_342: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_343: "f32[128]" = torch.ops.aten.mul.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
        unsqueeze_322: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_343, 0);  mul_343 = None
        unsqueeze_323: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
        unsqueeze_324: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
        mul_344: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_325: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_344, 0);  mul_344 = None
        unsqueeze_326: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_345: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_324);  sub_99 = unsqueeze_324 = None
        sub_101: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_14, mul_345);  where_14 = mul_345 = None
        sub_102: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_321);  sub_101 = unsqueeze_321 = None
        mul_346: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_327);  sub_102 = unsqueeze_327 = None
        mul_347: "f32[128]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_13);  sum_67 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:90 in forward, code: x = self.conv(x)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_346, relu_3, primals_26, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_346 = primals_26 = None
        getitem_114: "f32[32, 64, 56, 56]" = convolution_backward_22[0]
        getitem_115: "f32[128, 32, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:119 in forward, code: out = self.act1(out)
        le_15: "b8[32, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_15: "f32[32, 64, 56, 56]" = torch.ops.aten.where.self(le_15, full_default, getitem_114);  le_15 = getitem_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        sum_68: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_103: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_330);  convolution_3 = unsqueeze_330 = None
        mul_348: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_15, sub_103)
        sum_69: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_348, [0, 2, 3]);  mul_348 = None
        mul_349: "f32[64]" = torch.ops.aten.mul.Tensor(sum_68, 9.964923469387754e-06)
        unsqueeze_331: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_349, 0);  mul_349 = None
        unsqueeze_332: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_350: "f32[64]" = torch.ops.aten.mul.Tensor(sum_69, 9.964923469387754e-06)
        mul_351: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_352: "f32[64]" = torch.ops.aten.mul.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
        unsqueeze_334: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_352, 0);  mul_352 = None
        unsqueeze_335: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
        unsqueeze_336: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
        mul_353: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_337: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
        unsqueeze_338: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
        mul_354: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_336);  sub_103 = unsqueeze_336 = None
        sub_105: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_15, mul_354);  where_15 = mul_354 = None
        sub_106: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_333);  sub_105 = unsqueeze_333 = None
        mul_355: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_339);  sub_106 = unsqueeze_339 = None
        mul_356: "f32[64]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_10);  sum_69 = squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:117 in forward, code: out = self.conv1(x)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_355, getitem_6, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_355 = getitem_6 = primals_20 = None
        getitem_117: "f32[32, 64, 56, 56]" = convolution_backward_23[0]
        getitem_118: "f32[64, 64, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
        add_126: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_102, getitem_117);  getitem_102 = getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:748 in forward_features, code: x = self.maxpool(x)
        full_default_16: "f32[2048, 12544]" = torch.ops.aten.full.default([2048, 12544], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_47: "f32[2048, 3136]" = torch.ops.aten.reshape.default(add_126, [2048, 3136]);  add_126 = None
        _low_memory_max_pool_offsets_to_indices: "i64[32, 64, 56, 56]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_7, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_7 = None
        view_48: "i64[2048, 3136]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [2048, 3136]);  _low_memory_max_pool_offsets_to_indices = None
        scatter_add: "f32[2048, 12544]" = torch.ops.aten.scatter_add.default(full_default_16, 1, view_48, view_47);  full_default_16 = view_48 = view_47 = None
        view_49: "f32[32, 64, 112, 112]" = torch.ops.aten.reshape.default(scatter_add, [32, 64, 112, 112]);  scatter_add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:746 in forward_features, code: x = self.bn1(x)
        sub_2: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[32, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:747 in forward_features, code: x = self.act1(x)
        relu_2: "f32[32, 64, 112, 112]" = torch.ops.aten.relu.default(add_14);  add_14 = None
        le_16: "b8[32, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_16: "f32[32, 64, 112, 112]" = torch.ops.aten.where.self(le_16, full_default, view_49);  le_16 = view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:746 in forward_features, code: x = self.bn1(x)
        squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        unsqueeze_340: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_341: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
        unsqueeze_342: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
        sum_70: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_107: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_342);  convolution_2 = unsqueeze_342 = None
        mul_357: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_16, sub_107)
        sum_71: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 2, 3]);  mul_357 = None
        mul_358: "f32[64]" = torch.ops.aten.mul.Tensor(sum_70, 2.4912308673469386e-06)
        unsqueeze_343: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
        unsqueeze_344: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
        mul_359: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, 2.4912308673469386e-06)
        squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_360: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_361: "f32[64]" = torch.ops.aten.mul.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
        unsqueeze_346: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_361, 0);  mul_361 = None
        unsqueeze_347: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
        unsqueeze_348: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
        mul_362: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_349: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_362, 0);  mul_362 = None
        unsqueeze_350: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
        unsqueeze_351: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
        mul_363: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_348);  sub_107 = unsqueeze_348 = None
        sub_109: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_16, mul_363);  where_16 = mul_363 = None
        sub_110: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_345);  sub_109 = unsqueeze_345 = None
        mul_364: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_351);  sub_110 = unsqueeze_351 = None
        mul_365: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_7);  sum_71 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:745 in forward_features, code: x = self.conv1(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_364, relu_1, primals_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_364 = primals_14 = None
        getitem_120: "f32[32, 32, 112, 112]" = convolution_backward_24[0]
        getitem_121: "f32[64, 32, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
        le_17: "b8[32, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_17: "f32[32, 32, 112, 112]" = torch.ops.aten.where.self(le_17, full_default, getitem_120);  le_17 = getitem_120 = None
        sum_72: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_111: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_354);  convolution_1 = unsqueeze_354 = None
        mul_366: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_17, sub_111)
        sum_73: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 2, 3]);  mul_366 = None
        mul_367: "f32[32]" = torch.ops.aten.mul.Tensor(sum_72, 2.4912308673469386e-06)
        unsqueeze_355: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
        unsqueeze_356: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        unsqueeze_357: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        mul_368: "f32[32]" = torch.ops.aten.mul.Tensor(sum_73, 2.4912308673469386e-06)
        mul_369: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_370: "f32[32]" = torch.ops.aten.mul.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_358: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
        unsqueeze_359: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
        unsqueeze_360: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
        mul_371: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_361: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
        unsqueeze_362: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_372: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_360);  sub_111 = unsqueeze_360 = None
        sub_113: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_17, mul_372);  where_17 = mul_372 = None
        sub_114: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_357);  sub_113 = unsqueeze_357 = None
        mul_373: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_363);  sub_114 = unsqueeze_363 = None
        mul_374: "f32[32]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_4);  sum_73 = squeeze_4 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_373, relu, primals_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_373 = primals_8 = None
        getitem_123: "f32[32, 32, 112, 112]" = convolution_backward_25[0]
        getitem_124: "f32[32, 32, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
        le_18: "b8[32, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_18: "f32[32, 32, 112, 112]" = torch.ops.aten.where.self(le_18, full_default, getitem_123);  le_18 = full_default = getitem_123 = None
        sum_74: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        sub_115: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_366);  convolution = unsqueeze_366 = None
        mul_375: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_18, sub_115)
        sum_75: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 2, 3]);  mul_375 = None
        mul_376: "f32[32]" = torch.ops.aten.mul.Tensor(sum_74, 2.4912308673469386e-06)
        unsqueeze_367: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
        unsqueeze_368: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_377: "f32[32]" = torch.ops.aten.mul.Tensor(sum_75, 2.4912308673469386e-06)
        mul_378: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_379: "f32[32]" = torch.ops.aten.mul.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
        unsqueeze_370: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
        unsqueeze_371: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
        unsqueeze_372: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
        mul_380: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_373: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
        unsqueeze_374: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_381: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_372);  sub_115 = unsqueeze_372 = None
        sub_117: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_18, mul_381);  where_18 = mul_381 = None
        sub_118: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_369);  sub_117 = unsqueeze_369 = None
        mul_382: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_375);  sub_118 = unsqueeze_375 = None
        mul_383: "f32[32]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_1);  sum_75 = squeeze_1 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_382, primals_2, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_382 = primals_2 = primals_1 = None
        getitem_127: "f32[32, 3, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
        return (getitem_127, None, None, None, None, mul_383, sum_74, getitem_124, None, None, None, mul_374, sum_72, getitem_121, None, None, None, mul_365, sum_70, getitem_118, None, None, None, mul_356, sum_68, getitem_115, None, None, None, mul_347, sum_66, getitem_112, sum_65, None, None, None, mul_338, sum_63, getitem_109, sum_62, getitem_106, None, None, None, mul_326, sum_58, getitem_103, None, None, None, mul_317, sum_56, getitem_100, None, None, None, mul_308, sum_54, getitem_97, None, None, None, mul_299, sum_52, getitem_94, sum_51, None, None, None, mul_290, sum_49, getitem_91, sum_48, getitem_88, None, None, None, mul_278, sum_44, getitem_85, None, None, None, mul_269, sum_42, getitem_82, None, None, None, mul_260, sum_40, getitem_79, None, None, None, mul_251, sum_38, getitem_76, sum_37, None, None, None, mul_242, sum_35, getitem_73, sum_34, getitem_70, None, None, None, mul_230, sum_30, getitem_67, None, None, None, mul_221, sum_28, getitem_64, None, None, None, mul_212, sum_26, getitem_61, None, None, None, mul_203, sum_24, getitem_58, sum_23, None, None, None, mul_194, sum_21, getitem_55, sum_20, getitem_52, None, None, None, mul_182, sum_16, getitem_49, None, None, None, mul_173, sum_14, mm_1, view_25)
