class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3]", primals_2: "f32[128, 3, 32, 32]", primals_6: "f32[16]", primals_8: "f32[16, 16, 3, 3]", primals_12: "f32[16]", primals_14: "f32[16, 16, 3, 3]", primals_18: "f32[16]", primals_20: "f32[16, 16, 3, 3]", primals_24: "f32[16]", primals_26: "f32[16, 16, 3, 3]", primals_30: "f32[16]", primals_32: "f32[16, 16, 3, 3]", primals_36: "f32[16]", primals_38: "f32[16, 16, 3, 3]", primals_42: "f32[16]", primals_44: "f32[32, 16, 3, 3]", primals_48: "f32[32]", primals_50: "f32[32, 32, 3, 3]", primals_54: "f32[32]", primals_56: "f32[32, 16, 1, 1]", primals_58: "f32[32, 32, 3, 3]", primals_62: "f32[32]", primals_64: "f32[32, 32, 3, 3]", primals_68: "f32[32]", primals_70: "f32[32, 32, 3, 3]", primals_74: "f32[32]", primals_76: "f32[32, 32, 3, 3]", primals_80: "f32[32]", primals_82: "f32[64, 32, 3, 3]", primals_86: "f32[64]", primals_88: "f32[64, 64, 3, 3]", primals_92: "f32[64]", primals_94: "f32[64, 32, 1, 1]", primals_96: "f32[64, 64, 3, 3]", primals_100: "f32[64]", primals_102: "f32[64, 64, 3, 3]", primals_106: "f32[64]", primals_108: "f32[64, 64, 3, 3]", primals_112: "f32[64]", primals_114: "f32[64, 64, 3, 3]", primals_118: "f32[64]", primals_120: "f32[10, 64]", convolution: "f32[128, 16, 32, 32]", squeeze_1: "f32[16]", relu: "f32[128, 16, 32, 32]", convolution_1: "f32[128, 16, 32, 32]", squeeze_4: "f32[16]", relu_1: "f32[128, 16, 32, 32]", convolution_2: "f32[128, 16, 32, 32]", squeeze_7: "f32[16]", relu_2: "f32[128, 16, 32, 32]", convolution_3: "f32[128, 16, 32, 32]", squeeze_10: "f32[16]", relu_3: "f32[128, 16, 32, 32]", convolution_4: "f32[128, 16, 32, 32]", squeeze_13: "f32[16]", relu_4: "f32[128, 16, 32, 32]", convolution_5: "f32[128, 16, 32, 32]", squeeze_16: "f32[16]", relu_5: "f32[128, 16, 32, 32]", convolution_6: "f32[128, 16, 32, 32]", squeeze_19: "f32[16]", relu_6: "f32[128, 16, 32, 32]", convolution_7: "f32[128, 32, 16, 16]", squeeze_22: "f32[32]", relu_7: "f32[128, 32, 16, 16]", convolution_8: "f32[128, 32, 16, 16]", squeeze_25: "f32[32]", relu_8: "f32[128, 32, 16, 16]", convolution_10: "f32[128, 32, 16, 16]", squeeze_28: "f32[32]", relu_9: "f32[128, 32, 16, 16]", convolution_11: "f32[128, 32, 16, 16]", squeeze_31: "f32[32]", relu_10: "f32[128, 32, 16, 16]", convolution_12: "f32[128, 32, 16, 16]", squeeze_34: "f32[32]", relu_11: "f32[128, 32, 16, 16]", convolution_13: "f32[128, 32, 16, 16]", squeeze_37: "f32[32]", relu_12: "f32[128, 32, 16, 16]", convolution_14: "f32[128, 64, 8, 8]", squeeze_40: "f32[64]", relu_13: "f32[128, 64, 8, 8]", convolution_15: "f32[128, 64, 8, 8]", squeeze_43: "f32[64]", relu_14: "f32[128, 64, 8, 8]", convolution_17: "f32[128, 64, 8, 8]", squeeze_46: "f32[64]", relu_15: "f32[128, 64, 8, 8]", convolution_18: "f32[128, 64, 8, 8]", squeeze_49: "f32[64]", relu_16: "f32[128, 64, 8, 8]", convolution_19: "f32[128, 64, 8, 8]", squeeze_52: "f32[64]", relu_17: "f32[128, 64, 8, 8]", convolution_20: "f32[128, 64, 8, 8]", squeeze_55: "f32[64]", view: "f32[128, 64]", le: "b8[128, 64, 8, 8]", unsqueeze_78: "f32[1, 64, 1, 1]", unsqueeze_90: "f32[1, 64, 1, 1]", unsqueeze_102: "f32[1, 64, 1, 1]", unsqueeze_114: "f32[1, 64, 1, 1]", unsqueeze_126: "f32[1, 64, 1, 1]", unsqueeze_138: "f32[1, 64, 1, 1]", unsqueeze_150: "f32[1, 32, 1, 1]", unsqueeze_162: "f32[1, 32, 1, 1]", unsqueeze_174: "f32[1, 32, 1, 1]", unsqueeze_186: "f32[1, 32, 1, 1]", unsqueeze_198: "f32[1, 32, 1, 1]", unsqueeze_210: "f32[1, 32, 1, 1]", unsqueeze_222: "f32[1, 16, 1, 1]", unsqueeze_234: "f32[1, 16, 1, 1]", unsqueeze_246: "f32[1, 16, 1, 1]", unsqueeze_258: "f32[1, 16, 1, 1]", unsqueeze_270: "f32[1, 16, 1, 1]", unsqueeze_282: "f32[1, 16, 1, 1]", unsqueeze_294: "f32[1, 16, 1, 1]", tangents_1: "f32[128, 10]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:139 in forward, code: x = self.output_net(x)
        permute: "f32[64, 10]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_1: "f32[10, 64]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[128, 64]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "f32[10, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[10, 64]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 10]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_1: "f32[10]" = torch.ops.aten.reshape.default(sum_1, [10]);  sum_1 = None
        view_2: "f32[128, 64, 1, 1]" = torch.ops.aten.reshape.default(mm, [128, 64, 1, 1]);  mm = None
        expand: "f32[128, 64, 8, 8]" = torch.ops.aten.expand.default(view_2, [128, 64, 8, 8]);  view_2 = None
        div: "f32[128, 64, 8, 8]" = torch.ops.aten.div.Scalar(expand, 64);  expand = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[128, 64, 8, 8]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        sum_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_19: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_78);  convolution_20 = unsqueeze_78 = None
        mul_133: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(where, sub_19)
        sum_3: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_133, [0, 2, 3]);  mul_133 = None
        mul_134: "f32[64]" = torch.ops.aten.mul.Tensor(sum_2, 0.0001220703125)
        unsqueeze_79: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_134, 0);  mul_134 = None
        unsqueeze_80: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 2);  unsqueeze_79 = None
        unsqueeze_81: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, 3);  unsqueeze_80 = None
        mul_135: "f32[64]" = torch.ops.aten.mul.Tensor(sum_3, 0.0001220703125)
        mul_136: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_137: "f32[64]" = torch.ops.aten.mul.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
        unsqueeze_82: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_137, 0);  mul_137 = None
        unsqueeze_83: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 2);  unsqueeze_82 = None
        unsqueeze_84: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 3);  unsqueeze_83 = None
        mul_138: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_118);  primals_118 = None
        unsqueeze_85: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_138, 0);  mul_138 = None
        unsqueeze_86: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 2);  unsqueeze_85 = None
        unsqueeze_87: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, 3);  unsqueeze_86 = None
        mul_139: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_84);  sub_19 = unsqueeze_84 = None
        sub_21: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(where, mul_139);  mul_139 = None
        sub_22: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(sub_21, unsqueeze_81);  sub_21 = unsqueeze_81 = None
        mul_140: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_87);  sub_22 = unsqueeze_87 = None
        mul_141: "f32[64]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_55);  sum_3 = squeeze_55 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_140, relu_17, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_140 = primals_114 = None
        getitem_38: "f32[128, 64, 8, 8]" = convolution_backward[0]
        getitem_39: "f32[64, 64, 3, 3]" = convolution_backward[1];  convolution_backward = None
        le_1: "b8[128, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_1: "f32[128, 64, 8, 8]" = torch.ops.aten.where.self(le_1, full_default, getitem_38);  le_1 = getitem_38 = None
        sum_4: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_23: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_90);  convolution_19 = unsqueeze_90 = None
        mul_142: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(where_1, sub_23)
        sum_5: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 2, 3]);  mul_142 = None
        mul_143: "f32[64]" = torch.ops.aten.mul.Tensor(sum_4, 0.0001220703125)
        unsqueeze_91: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_143, 0);  mul_143 = None
        unsqueeze_92: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 2);  unsqueeze_91 = None
        unsqueeze_93: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, 3);  unsqueeze_92 = None
        mul_144: "f32[64]" = torch.ops.aten.mul.Tensor(sum_5, 0.0001220703125)
        mul_145: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_146: "f32[64]" = torch.ops.aten.mul.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_94: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_146, 0);  mul_146 = None
        unsqueeze_95: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, 2);  unsqueeze_94 = None
        unsqueeze_96: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 3);  unsqueeze_95 = None
        mul_147: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_112);  primals_112 = None
        unsqueeze_97: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_147, 0);  mul_147 = None
        unsqueeze_98: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 2);  unsqueeze_97 = None
        unsqueeze_99: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, 3);  unsqueeze_98 = None
        mul_148: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_96);  sub_23 = unsqueeze_96 = None
        sub_25: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(where_1, mul_148);  where_1 = mul_148 = None
        sub_26: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(sub_25, unsqueeze_93);  sub_25 = unsqueeze_93 = None
        mul_149: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_99);  sub_26 = unsqueeze_99 = None
        mul_150: "f32[64]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_52);  sum_5 = squeeze_52 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_149, relu_16, primals_108, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_149 = primals_108 = None
        getitem_41: "f32[128, 64, 8, 8]" = convolution_backward_1[0]
        getitem_42: "f32[64, 64, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
        add_104: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(where, getitem_41);  where = getitem_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        le_2: "b8[128, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_2: "f32[128, 64, 8, 8]" = torch.ops.aten.where.self(le_2, full_default, add_104);  le_2 = add_104 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        sum_6: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_27: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_102);  convolution_18 = unsqueeze_102 = None
        mul_151: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(where_2, sub_27)
        sum_7: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_151, [0, 2, 3]);  mul_151 = None
        mul_152: "f32[64]" = torch.ops.aten.mul.Tensor(sum_6, 0.0001220703125)
        unsqueeze_103: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_152, 0);  mul_152 = None
        unsqueeze_104: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 2);  unsqueeze_103 = None
        unsqueeze_105: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, 3);  unsqueeze_104 = None
        mul_153: "f32[64]" = torch.ops.aten.mul.Tensor(sum_7, 0.0001220703125)
        mul_154: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_155: "f32[64]" = torch.ops.aten.mul.Tensor(mul_153, mul_154);  mul_153 = mul_154 = None
        unsqueeze_106: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_155, 0);  mul_155 = None
        unsqueeze_107: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 2);  unsqueeze_106 = None
        unsqueeze_108: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 3);  unsqueeze_107 = None
        mul_156: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_106);  primals_106 = None
        unsqueeze_109: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_156, 0);  mul_156 = None
        unsqueeze_110: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 2);  unsqueeze_109 = None
        unsqueeze_111: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, 3);  unsqueeze_110 = None
        mul_157: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_108);  sub_27 = unsqueeze_108 = None
        sub_29: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(where_2, mul_157);  mul_157 = None
        sub_30: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(sub_29, unsqueeze_105);  sub_29 = unsqueeze_105 = None
        mul_158: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_111);  sub_30 = unsqueeze_111 = None
        mul_159: "f32[64]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_49);  sum_7 = squeeze_49 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_158, relu_15, primals_102, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_158 = primals_102 = None
        getitem_44: "f32[128, 64, 8, 8]" = convolution_backward_2[0]
        getitem_45: "f32[64, 64, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
        le_3: "b8[128, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_3: "f32[128, 64, 8, 8]" = torch.ops.aten.where.self(le_3, full_default, getitem_44);  le_3 = getitem_44 = None
        sum_8: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_31: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_114);  convolution_17 = unsqueeze_114 = None
        mul_160: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(where_3, sub_31)
        sum_9: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 2, 3]);  mul_160 = None
        mul_161: "f32[64]" = torch.ops.aten.mul.Tensor(sum_8, 0.0001220703125)
        unsqueeze_115: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_161, 0);  mul_161 = None
        unsqueeze_116: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 2);  unsqueeze_115 = None
        unsqueeze_117: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, 3);  unsqueeze_116 = None
        mul_162: "f32[64]" = torch.ops.aten.mul.Tensor(sum_9, 0.0001220703125)
        mul_163: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_164: "f32[64]" = torch.ops.aten.mul.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        unsqueeze_118: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_164, 0);  mul_164 = None
        unsqueeze_119: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 2);  unsqueeze_118 = None
        unsqueeze_120: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 3);  unsqueeze_119 = None
        mul_165: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_100);  primals_100 = None
        unsqueeze_121: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_165, 0);  mul_165 = None
        unsqueeze_122: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 2);  unsqueeze_121 = None
        unsqueeze_123: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 3);  unsqueeze_122 = None
        mul_166: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_120);  sub_31 = unsqueeze_120 = None
        sub_33: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(where_3, mul_166);  where_3 = mul_166 = None
        sub_34: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(sub_33, unsqueeze_117);  sub_33 = unsqueeze_117 = None
        mul_167: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_123);  sub_34 = unsqueeze_123 = None
        mul_168: "f32[64]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_46);  sum_9 = squeeze_46 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_167, relu_14, primals_96, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_167 = primals_96 = None
        getitem_47: "f32[128, 64, 8, 8]" = convolution_backward_3[0]
        getitem_48: "f32[64, 64, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
        add_105: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(where_2, getitem_47);  where_2 = getitem_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        le_4: "b8[128, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_4: "f32[128, 64, 8, 8]" = torch.ops.aten.where.self(le_4, full_default, add_105);  le_4 = add_105 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:53 in forward, code: x = self.downsample(x)
        sum_10: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_4, relu_12, primals_94, [64], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  primals_94 = None
        getitem_50: "f32[128, 32, 16, 16]" = convolution_backward_4[0]
        getitem_51: "f32[64, 32, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        sum_11: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_35: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_126);  convolution_15 = unsqueeze_126 = None
        mul_169: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(where_4, sub_35)
        sum_12: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 2, 3]);  mul_169 = None
        mul_170: "f32[64]" = torch.ops.aten.mul.Tensor(sum_11, 0.0001220703125)
        unsqueeze_127: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_170, 0);  mul_170 = None
        unsqueeze_128: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
        unsqueeze_129: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 3);  unsqueeze_128 = None
        mul_171: "f32[64]" = torch.ops.aten.mul.Tensor(sum_12, 0.0001220703125)
        mul_172: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_173: "f32[64]" = torch.ops.aten.mul.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
        unsqueeze_130: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_173, 0);  mul_173 = None
        unsqueeze_131: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 2);  unsqueeze_130 = None
        unsqueeze_132: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 3);  unsqueeze_131 = None
        mul_174: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_92);  primals_92 = None
        unsqueeze_133: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_174, 0);  mul_174 = None
        unsqueeze_134: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
        unsqueeze_135: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 3);  unsqueeze_134 = None
        mul_175: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_132);  sub_35 = unsqueeze_132 = None
        sub_37: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(where_4, mul_175);  where_4 = mul_175 = None
        sub_38: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(sub_37, unsqueeze_129);  sub_37 = unsqueeze_129 = None
        mul_176: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_135);  sub_38 = unsqueeze_135 = None
        mul_177: "f32[64]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_43);  sum_12 = squeeze_43 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_176, relu_13, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_176 = primals_88 = None
        getitem_53: "f32[128, 64, 8, 8]" = convolution_backward_5[0]
        getitem_54: "f32[64, 64, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
        le_5: "b8[128, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_5: "f32[128, 64, 8, 8]" = torch.ops.aten.where.self(le_5, full_default, getitem_53);  le_5 = getitem_53 = None
        sum_13: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_39: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_138);  convolution_14 = unsqueeze_138 = None
        mul_178: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(where_5, sub_39)
        sum_14: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_178, [0, 2, 3]);  mul_178 = None
        mul_179: "f32[64]" = torch.ops.aten.mul.Tensor(sum_13, 0.0001220703125)
        unsqueeze_139: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_179, 0);  mul_179 = None
        unsqueeze_140: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
        unsqueeze_141: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
        mul_180: "f32[64]" = torch.ops.aten.mul.Tensor(sum_14, 0.0001220703125)
        mul_181: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_182: "f32[64]" = torch.ops.aten.mul.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
        unsqueeze_142: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_182, 0);  mul_182 = None
        unsqueeze_143: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
        unsqueeze_144: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
        mul_183: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_86);  primals_86 = None
        unsqueeze_145: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_183, 0);  mul_183 = None
        unsqueeze_146: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
        unsqueeze_147: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
        mul_184: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_144);  sub_39 = unsqueeze_144 = None
        sub_41: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(where_5, mul_184);  where_5 = mul_184 = None
        sub_42: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(sub_41, unsqueeze_141);  sub_41 = unsqueeze_141 = None
        mul_185: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_147);  sub_42 = unsqueeze_147 = None
        mul_186: "f32[64]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_40);  sum_14 = squeeze_40 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_185, relu_12, primals_82, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_185 = primals_82 = None
        getitem_56: "f32[128, 32, 16, 16]" = convolution_backward_6[0]
        getitem_57: "f32[64, 32, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
        add_106: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(getitem_50, getitem_56);  getitem_50 = getitem_56 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        le_6: "b8[128, 32, 16, 16]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_6: "f32[128, 32, 16, 16]" = torch.ops.aten.where.self(le_6, full_default, add_106);  le_6 = add_106 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        sum_15: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_43: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_150);  convolution_13 = unsqueeze_150 = None
        mul_187: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(where_6, sub_43)
        sum_16: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_187, [0, 2, 3]);  mul_187 = None
        mul_188: "f32[32]" = torch.ops.aten.mul.Tensor(sum_15, 3.0517578125e-05)
        unsqueeze_151: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_188, 0);  mul_188 = None
        unsqueeze_152: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
        unsqueeze_153: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
        mul_189: "f32[32]" = torch.ops.aten.mul.Tensor(sum_16, 3.0517578125e-05)
        mul_190: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_191: "f32[32]" = torch.ops.aten.mul.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None
        unsqueeze_154: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_191, 0);  mul_191 = None
        unsqueeze_155: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
        unsqueeze_156: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
        mul_192: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_80);  primals_80 = None
        unsqueeze_157: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_192, 0);  mul_192 = None
        unsqueeze_158: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
        unsqueeze_159: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
        mul_193: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_156);  sub_43 = unsqueeze_156 = None
        sub_45: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(where_6, mul_193);  mul_193 = None
        sub_46: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_153);  sub_45 = unsqueeze_153 = None
        mul_194: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_159);  sub_46 = unsqueeze_159 = None
        mul_195: "f32[32]" = torch.ops.aten.mul.Tensor(sum_16, squeeze_37);  sum_16 = squeeze_37 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_194, relu_11, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_194 = primals_76 = None
        getitem_59: "f32[128, 32, 16, 16]" = convolution_backward_7[0]
        getitem_60: "f32[32, 32, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
        le_7: "b8[128, 32, 16, 16]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_7: "f32[128, 32, 16, 16]" = torch.ops.aten.where.self(le_7, full_default, getitem_59);  le_7 = getitem_59 = None
        sum_17: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_47: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_162);  convolution_12 = unsqueeze_162 = None
        mul_196: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(where_7, sub_47)
        sum_18: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_196, [0, 2, 3]);  mul_196 = None
        mul_197: "f32[32]" = torch.ops.aten.mul.Tensor(sum_17, 3.0517578125e-05)
        unsqueeze_163: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_197, 0);  mul_197 = None
        unsqueeze_164: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        unsqueeze_165: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        mul_198: "f32[32]" = torch.ops.aten.mul.Tensor(sum_18, 3.0517578125e-05)
        mul_199: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_200: "f32[32]" = torch.ops.aten.mul.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None
        unsqueeze_166: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_200, 0);  mul_200 = None
        unsqueeze_167: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
        unsqueeze_168: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
        mul_201: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_74);  primals_74 = None
        unsqueeze_169: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_201, 0);  mul_201 = None
        unsqueeze_170: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
        unsqueeze_171: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
        mul_202: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_168);  sub_47 = unsqueeze_168 = None
        sub_49: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(where_7, mul_202);  where_7 = mul_202 = None
        sub_50: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(sub_49, unsqueeze_165);  sub_49 = unsqueeze_165 = None
        mul_203: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_171);  sub_50 = unsqueeze_171 = None
        mul_204: "f32[32]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_34);  sum_18 = squeeze_34 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_203, relu_10, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_203 = primals_70 = None
        getitem_62: "f32[128, 32, 16, 16]" = convolution_backward_8[0]
        getitem_63: "f32[32, 32, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
        add_107: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(where_6, getitem_62);  where_6 = getitem_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        le_8: "b8[128, 32, 16, 16]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_8: "f32[128, 32, 16, 16]" = torch.ops.aten.where.self(le_8, full_default, add_107);  le_8 = add_107 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        sum_19: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_51: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_174);  convolution_11 = unsqueeze_174 = None
        mul_205: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(where_8, sub_51)
        sum_20: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_205, [0, 2, 3]);  mul_205 = None
        mul_206: "f32[32]" = torch.ops.aten.mul.Tensor(sum_19, 3.0517578125e-05)
        unsqueeze_175: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_206, 0);  mul_206 = None
        unsqueeze_176: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
        unsqueeze_177: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
        mul_207: "f32[32]" = torch.ops.aten.mul.Tensor(sum_20, 3.0517578125e-05)
        mul_208: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_209: "f32[32]" = torch.ops.aten.mul.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_178: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_209, 0);  mul_209 = None
        unsqueeze_179: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
        unsqueeze_180: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
        mul_210: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_68);  primals_68 = None
        unsqueeze_181: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_210, 0);  mul_210 = None
        unsqueeze_182: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
        unsqueeze_183: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
        mul_211: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_180);  sub_51 = unsqueeze_180 = None
        sub_53: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(where_8, mul_211);  mul_211 = None
        sub_54: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_177);  sub_53 = unsqueeze_177 = None
        mul_212: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_183);  sub_54 = unsqueeze_183 = None
        mul_213: "f32[32]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_31);  sum_20 = squeeze_31 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_212, relu_9, primals_64, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_212 = primals_64 = None
        getitem_65: "f32[128, 32, 16, 16]" = convolution_backward_9[0]
        getitem_66: "f32[32, 32, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
        le_9: "b8[128, 32, 16, 16]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_9: "f32[128, 32, 16, 16]" = torch.ops.aten.where.self(le_9, full_default, getitem_65);  le_9 = getitem_65 = None
        sum_21: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_55: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_186);  convolution_10 = unsqueeze_186 = None
        mul_214: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(where_9, sub_55)
        sum_22: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 2, 3]);  mul_214 = None
        mul_215: "f32[32]" = torch.ops.aten.mul.Tensor(sum_21, 3.0517578125e-05)
        unsqueeze_187: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_215, 0);  mul_215 = None
        unsqueeze_188: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
        unsqueeze_189: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
        mul_216: "f32[32]" = torch.ops.aten.mul.Tensor(sum_22, 3.0517578125e-05)
        mul_217: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_218: "f32[32]" = torch.ops.aten.mul.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
        unsqueeze_190: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_218, 0);  mul_218 = None
        unsqueeze_191: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
        unsqueeze_192: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
        mul_219: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_62);  primals_62 = None
        unsqueeze_193: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_219, 0);  mul_219 = None
        unsqueeze_194: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
        unsqueeze_195: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
        mul_220: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_192);  sub_55 = unsqueeze_192 = None
        sub_57: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(where_9, mul_220);  where_9 = mul_220 = None
        sub_58: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_189);  sub_57 = unsqueeze_189 = None
        mul_221: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_195);  sub_58 = unsqueeze_195 = None
        mul_222: "f32[32]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_28);  sum_22 = squeeze_28 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_221, relu_8, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_221 = primals_58 = None
        getitem_68: "f32[128, 32, 16, 16]" = convolution_backward_10[0]
        getitem_69: "f32[32, 32, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
        add_108: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(where_8, getitem_68);  where_8 = getitem_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        le_10: "b8[128, 32, 16, 16]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_10: "f32[128, 32, 16, 16]" = torch.ops.aten.where.self(le_10, full_default, add_108);  le_10 = add_108 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:53 in forward, code: x = self.downsample(x)
        sum_23: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_10, relu_6, primals_56, [32], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  primals_56 = None
        getitem_71: "f32[128, 16, 32, 32]" = convolution_backward_11[0]
        getitem_72: "f32[32, 16, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        sum_24: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_59: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_198);  convolution_8 = unsqueeze_198 = None
        mul_223: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(where_10, sub_59)
        sum_25: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_223, [0, 2, 3]);  mul_223 = None
        mul_224: "f32[32]" = torch.ops.aten.mul.Tensor(sum_24, 3.0517578125e-05)
        unsqueeze_199: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_224, 0);  mul_224 = None
        unsqueeze_200: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
        unsqueeze_201: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
        mul_225: "f32[32]" = torch.ops.aten.mul.Tensor(sum_25, 3.0517578125e-05)
        mul_226: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_227: "f32[32]" = torch.ops.aten.mul.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        unsqueeze_202: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_227, 0);  mul_227 = None
        unsqueeze_203: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
        unsqueeze_204: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
        mul_228: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_205: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_228, 0);  mul_228 = None
        unsqueeze_206: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
        unsqueeze_207: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
        mul_229: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_204);  sub_59 = unsqueeze_204 = None
        sub_61: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(where_10, mul_229);  where_10 = mul_229 = None
        sub_62: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_201);  sub_61 = unsqueeze_201 = None
        mul_230: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_207);  sub_62 = unsqueeze_207 = None
        mul_231: "f32[32]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_25);  sum_25 = squeeze_25 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_230, relu_7, primals_50, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_230 = primals_50 = None
        getitem_74: "f32[128, 32, 16, 16]" = convolution_backward_12[0]
        getitem_75: "f32[32, 32, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
        le_11: "b8[128, 32, 16, 16]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_11: "f32[128, 32, 16, 16]" = torch.ops.aten.where.self(le_11, full_default, getitem_74);  le_11 = getitem_74 = None
        sum_26: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_63: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_210);  convolution_7 = unsqueeze_210 = None
        mul_232: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(where_11, sub_63)
        sum_27: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 2, 3]);  mul_232 = None
        mul_233: "f32[32]" = torch.ops.aten.mul.Tensor(sum_26, 3.0517578125e-05)
        unsqueeze_211: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_233, 0);  mul_233 = None
        unsqueeze_212: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        unsqueeze_213: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
        mul_234: "f32[32]" = torch.ops.aten.mul.Tensor(sum_27, 3.0517578125e-05)
        mul_235: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_236: "f32[32]" = torch.ops.aten.mul.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
        unsqueeze_214: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_236, 0);  mul_236 = None
        unsqueeze_215: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        unsqueeze_216: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
        mul_237: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_217: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_237, 0);  mul_237 = None
        unsqueeze_218: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
        unsqueeze_219: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
        mul_238: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_216);  sub_63 = unsqueeze_216 = None
        sub_65: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(where_11, mul_238);  where_11 = mul_238 = None
        sub_66: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_213);  sub_65 = unsqueeze_213 = None
        mul_239: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_219);  sub_66 = unsqueeze_219 = None
        mul_240: "f32[32]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_22);  sum_27 = squeeze_22 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_239, relu_6, primals_44, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_239 = primals_44 = None
        getitem_77: "f32[128, 16, 32, 32]" = convolution_backward_13[0]
        getitem_78: "f32[32, 16, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
        add_109: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(getitem_71, getitem_77);  getitem_71 = getitem_77 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        le_12: "b8[128, 16, 32, 32]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_12: "f32[128, 16, 32, 32]" = torch.ops.aten.where.self(le_12, full_default, add_109);  le_12 = add_109 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        sum_28: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_67: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_222);  convolution_6 = unsqueeze_222 = None
        mul_241: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(where_12, sub_67)
        sum_29: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 2, 3]);  mul_241 = None
        mul_242: "f32[16]" = torch.ops.aten.mul.Tensor(sum_28, 7.62939453125e-06)
        unsqueeze_223: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_242, 0);  mul_242 = None
        unsqueeze_224: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        unsqueeze_225: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        mul_243: "f32[16]" = torch.ops.aten.mul.Tensor(sum_29, 7.62939453125e-06)
        mul_244: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_245: "f32[16]" = torch.ops.aten.mul.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
        unsqueeze_226: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_245, 0);  mul_245 = None
        unsqueeze_227: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        mul_246: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_229: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_246, 0);  mul_246 = None
        unsqueeze_230: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_247: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_228);  sub_67 = unsqueeze_228 = None
        sub_69: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(where_12, mul_247);  mul_247 = None
        sub_70: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_225);  sub_69 = unsqueeze_225 = None
        mul_248: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_231);  sub_70 = unsqueeze_231 = None
        mul_249: "f32[16]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_19);  sum_29 = squeeze_19 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_248, relu_5, primals_38, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_248 = primals_38 = None
        getitem_80: "f32[128, 16, 32, 32]" = convolution_backward_14[0]
        getitem_81: "f32[16, 16, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
        le_13: "b8[128, 16, 32, 32]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_13: "f32[128, 16, 32, 32]" = torch.ops.aten.where.self(le_13, full_default, getitem_80);  le_13 = getitem_80 = None
        sum_30: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_71: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_234);  convolution_5 = unsqueeze_234 = None
        mul_250: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(where_13, sub_71)
        sum_31: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 2, 3]);  mul_250 = None
        mul_251: "f32[16]" = torch.ops.aten.mul.Tensor(sum_30, 7.62939453125e-06)
        unsqueeze_235: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_251, 0);  mul_251 = None
        unsqueeze_236: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_252: "f32[16]" = torch.ops.aten.mul.Tensor(sum_31, 7.62939453125e-06)
        mul_253: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_254: "f32[16]" = torch.ops.aten.mul.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
        unsqueeze_238: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_254, 0);  mul_254 = None
        unsqueeze_239: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        mul_255: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_241: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_255, 0);  mul_255 = None
        unsqueeze_242: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_256: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_240);  sub_71 = unsqueeze_240 = None
        sub_73: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(where_13, mul_256);  where_13 = mul_256 = None
        sub_74: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_237);  sub_73 = unsqueeze_237 = None
        mul_257: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_243);  sub_74 = unsqueeze_243 = None
        mul_258: "f32[16]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_16);  sum_31 = squeeze_16 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_257, relu_4, primals_32, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_257 = primals_32 = None
        getitem_83: "f32[128, 16, 32, 32]" = convolution_backward_15[0]
        getitem_84: "f32[16, 16, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
        add_110: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(where_12, getitem_83);  where_12 = getitem_83 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        le_14: "b8[128, 16, 32, 32]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_14: "f32[128, 16, 32, 32]" = torch.ops.aten.where.self(le_14, full_default, add_110);  le_14 = add_110 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        sum_32: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_75: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_246);  convolution_4 = unsqueeze_246 = None
        mul_259: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(where_14, sub_75)
        sum_33: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 2, 3]);  mul_259 = None
        mul_260: "f32[16]" = torch.ops.aten.mul.Tensor(sum_32, 7.62939453125e-06)
        unsqueeze_247: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_260, 0);  mul_260 = None
        unsqueeze_248: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_261: "f32[16]" = torch.ops.aten.mul.Tensor(sum_33, 7.62939453125e-06)
        mul_262: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_263: "f32[16]" = torch.ops.aten.mul.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
        unsqueeze_250: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_263, 0);  mul_263 = None
        unsqueeze_251: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_264: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_253: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_264, 0);  mul_264 = None
        unsqueeze_254: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_265: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_252);  sub_75 = unsqueeze_252 = None
        sub_77: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(where_14, mul_265);  mul_265 = None
        sub_78: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_249);  sub_77 = unsqueeze_249 = None
        mul_266: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_255);  sub_78 = unsqueeze_255 = None
        mul_267: "f32[16]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_13);  sum_33 = squeeze_13 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_266, relu_3, primals_26, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_266 = primals_26 = None
        getitem_86: "f32[128, 16, 32, 32]" = convolution_backward_16[0]
        getitem_87: "f32[16, 16, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
        le_15: "b8[128, 16, 32, 32]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_15: "f32[128, 16, 32, 32]" = torch.ops.aten.where.self(le_15, full_default, getitem_86);  le_15 = getitem_86 = None
        sum_34: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_79: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_258);  convolution_3 = unsqueeze_258 = None
        mul_268: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(where_15, sub_79)
        sum_35: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 2, 3]);  mul_268 = None
        mul_269: "f32[16]" = torch.ops.aten.mul.Tensor(sum_34, 7.62939453125e-06)
        unsqueeze_259: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_269, 0);  mul_269 = None
        unsqueeze_260: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        unsqueeze_261: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
        mul_270: "f32[16]" = torch.ops.aten.mul.Tensor(sum_35, 7.62939453125e-06)
        mul_271: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_272: "f32[16]" = torch.ops.aten.mul.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        unsqueeze_262: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_272, 0);  mul_272 = None
        unsqueeze_263: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        mul_273: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_265: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_273, 0);  mul_273 = None
        unsqueeze_266: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_274: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_264);  sub_79 = unsqueeze_264 = None
        sub_81: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(where_15, mul_274);  where_15 = mul_274 = None
        sub_82: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_261);  sub_81 = unsqueeze_261 = None
        mul_275: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_267);  sub_82 = unsqueeze_267 = None
        mul_276: "f32[16]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_10);  sum_35 = squeeze_10 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_275, relu_2, primals_20, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_275 = primals_20 = None
        getitem_89: "f32[128, 16, 32, 32]" = convolution_backward_17[0]
        getitem_90: "f32[16, 16, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
        add_111: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(where_14, getitem_89);  where_14 = getitem_89 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        le_16: "b8[128, 16, 32, 32]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_16: "f32[128, 16, 32, 32]" = torch.ops.aten.where.self(le_16, full_default, add_111);  le_16 = add_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        sum_36: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_83: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_270);  convolution_2 = unsqueeze_270 = None
        mul_277: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(where_16, sub_83)
        sum_37: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 2, 3]);  mul_277 = None
        mul_278: "f32[16]" = torch.ops.aten.mul.Tensor(sum_36, 7.62939453125e-06)
        unsqueeze_271: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_278, 0);  mul_278 = None
        unsqueeze_272: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        unsqueeze_273: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        mul_279: "f32[16]" = torch.ops.aten.mul.Tensor(sum_37, 7.62939453125e-06)
        mul_280: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_281: "f32[16]" = torch.ops.aten.mul.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
        unsqueeze_274: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_281, 0);  mul_281 = None
        unsqueeze_275: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        mul_282: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_277: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_282, 0);  mul_282 = None
        unsqueeze_278: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_283: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_276);  sub_83 = unsqueeze_276 = None
        sub_85: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(where_16, mul_283);  mul_283 = None
        sub_86: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_273);  sub_85 = unsqueeze_273 = None
        mul_284: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_279);  sub_86 = unsqueeze_279 = None
        mul_285: "f32[16]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_7);  sum_37 = squeeze_7 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_284, relu_1, primals_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_284 = primals_14 = None
        getitem_92: "f32[128, 16, 32, 32]" = convolution_backward_18[0]
        getitem_93: "f32[16, 16, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
        le_17: "b8[128, 16, 32, 32]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_17: "f32[128, 16, 32, 32]" = torch.ops.aten.where.self(le_17, full_default, getitem_92);  le_17 = getitem_92 = None
        sum_38: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_87: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_282);  convolution_1 = unsqueeze_282 = None
        mul_286: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(where_17, sub_87)
        sum_39: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 2, 3]);  mul_286 = None
        mul_287: "f32[16]" = torch.ops.aten.mul.Tensor(sum_38, 7.62939453125e-06)
        unsqueeze_283: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
        unsqueeze_284: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_288: "f32[16]" = torch.ops.aten.mul.Tensor(sum_39, 7.62939453125e-06)
        mul_289: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_290: "f32[16]" = torch.ops.aten.mul.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        unsqueeze_286: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_290, 0);  mul_290 = None
        unsqueeze_287: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
        unsqueeze_288: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
        mul_291: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_289: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_291, 0);  mul_291 = None
        unsqueeze_290: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
        mul_292: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_288);  sub_87 = unsqueeze_288 = None
        sub_89: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(where_17, mul_292);  where_17 = mul_292 = None
        sub_90: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_285);  sub_89 = unsqueeze_285 = None
        mul_293: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_291);  sub_90 = unsqueeze_291 = None
        mul_294: "f32[16]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_4);  sum_39 = squeeze_4 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_293, relu, primals_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_293 = primals_8 = None
        getitem_95: "f32[128, 16, 32, 32]" = convolution_backward_19[0]
        getitem_96: "f32[16, 16, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
        add_112: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(where_16, getitem_95);  where_16 = getitem_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:137 in forward, code: x = self.input_net(x)
        le_18: "b8[128, 16, 32, 32]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_18: "f32[128, 16, 32, 32]" = torch.ops.aten.where.self(le_18, full_default, add_112);  le_18 = full_default = add_112 = None
        sum_40: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        sub_91: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_294);  convolution = unsqueeze_294 = None
        mul_295: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(where_18, sub_91)
        sum_41: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 2, 3]);  mul_295 = None
        mul_296: "f32[16]" = torch.ops.aten.mul.Tensor(sum_40, 7.62939453125e-06)
        unsqueeze_295: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_296, 0);  mul_296 = None
        unsqueeze_296: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
        mul_297: "f32[16]" = torch.ops.aten.mul.Tensor(sum_41, 7.62939453125e-06)
        mul_298: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_299: "f32[16]" = torch.ops.aten.mul.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
        unsqueeze_298: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_299, 0);  mul_299 = None
        unsqueeze_299: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
        unsqueeze_300: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
        mul_300: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_301: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_300, 0);  mul_300 = None
        unsqueeze_302: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
        mul_301: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_300);  sub_91 = unsqueeze_300 = None
        sub_93: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(where_18, mul_301);  where_18 = mul_301 = None
        sub_94: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_297);  sub_93 = unsqueeze_297 = None
        mul_302: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_303);  sub_94 = unsqueeze_303 = None
        mul_303: "f32[16]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_1);  sum_41 = squeeze_1 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_302, primals_2, primals_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_302 = primals_2 = primals_1 = None
        getitem_99: "f32[16, 3, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
        return (getitem_99, None, None, None, None, mul_303, sum_40, getitem_96, None, None, None, mul_294, sum_38, getitem_93, None, None, None, mul_285, sum_36, getitem_90, None, None, None, mul_276, sum_34, getitem_87, None, None, None, mul_267, sum_32, getitem_84, None, None, None, mul_258, sum_30, getitem_81, None, None, None, mul_249, sum_28, getitem_78, None, None, None, mul_240, sum_26, getitem_75, None, None, None, mul_231, sum_24, getitem_72, sum_23, getitem_69, None, None, None, mul_222, sum_21, getitem_66, None, None, None, mul_213, sum_19, getitem_63, None, None, None, mul_204, sum_17, getitem_60, None, None, None, mul_195, sum_15, getitem_57, None, None, None, mul_186, sum_13, getitem_54, None, None, None, mul_177, sum_11, getitem_51, sum_10, getitem_48, None, None, None, mul_168, sum_8, getitem_45, None, None, None, mul_159, sum_6, getitem_42, None, None, None, mul_150, sum_4, getitem_39, None, None, None, mul_141, sum_2, mm_1, view_1)
