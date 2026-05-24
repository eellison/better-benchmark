import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 9, 3, 3]", primals_2: "f32[1024, 9, 128, 128]", primals_6: "f32[64]", primals_8: "f32[64, 64, 3, 3]", primals_12: "f32[64]", primals_14: "f32[64, 64, 3, 3]", primals_18: "f32[64]", primals_20: "f32[64, 64, 1, 1]", primals_24: "f32[64]", primals_26: "f32[64, 64, 3, 3]", primals_30: "f32[64]", primals_32: "f32[64, 64, 3, 3]", primals_36: "f32[64]", primals_38: "f32[128, 64, 3, 3]", primals_42: "f32[128]", primals_44: "f32[128, 128, 3, 3]", primals_48: "f32[128]", primals_50: "f32[128, 64, 1, 1]", primals_54: "f32[128]", primals_56: "f32[128, 128, 3, 3]", primals_60: "f32[128]", primals_62: "f32[128, 128, 3, 3]", primals_66: "f32[128]", primals_68: "f32[256, 128, 3, 3]", primals_72: "f32[256]", primals_74: "f32[256, 256, 3, 3]", primals_78: "f32[256]", primals_80: "f32[256, 128, 1, 1]", primals_84: "f32[256]", primals_86: "f32[256, 256, 3, 3]", primals_90: "f32[256]", primals_92: "f32[256, 256, 3, 3]", primals_96: "f32[256]", primals_98: "f32[512, 256, 3, 3]", primals_102: "f32[512]", primals_104: "f32[512, 512, 3, 3]", primals_108: "f32[512]", primals_110: "f32[512, 256, 1, 1]", primals_114: "f32[512]", primals_116: "f32[512, 512, 3, 3]", primals_120: "f32[512]", primals_122: "f32[512, 512, 3, 3]", primals_126: "f32[512]", primals_128: "f32[65, 512]", convolution: "f32[1024, 64, 64, 64]", squeeze_1: "f32[64]", relu: "f32[1024, 64, 64, 64]", convolution_1: "f32[1024, 64, 32, 32]", squeeze_4: "f32[64]", relu_1: "f32[1024, 64, 32, 32]", convolution_2: "f32[1024, 64, 32, 32]", squeeze_7: "f32[64]", convolution_3: "f32[1024, 64, 32, 32]", squeeze_10: "f32[64]", relu_2: "f32[1024, 64, 32, 32]", convolution_4: "f32[1024, 64, 32, 32]", squeeze_13: "f32[64]", relu_3: "f32[1024, 64, 32, 32]", convolution_5: "f32[1024, 64, 32, 32]", squeeze_16: "f32[64]", relu_4: "f32[1024, 64, 32, 32]", convolution_6: "f32[1024, 128, 16, 16]", squeeze_19: "f32[128]", relu_5: "f32[1024, 128, 16, 16]", convolution_7: "f32[1024, 128, 16, 16]", squeeze_22: "f32[128]", convolution_8: "f32[1024, 128, 16, 16]", squeeze_25: "f32[128]", relu_6: "f32[1024, 128, 16, 16]", convolution_9: "f32[1024, 128, 16, 16]", squeeze_28: "f32[128]", relu_7: "f32[1024, 128, 16, 16]", convolution_10: "f32[1024, 128, 16, 16]", squeeze_31: "f32[128]", relu_8: "f32[1024, 128, 16, 16]", convolution_11: "f32[1024, 256, 8, 8]", squeeze_34: "f32[256]", relu_9: "f32[1024, 256, 8, 8]", convolution_12: "f32[1024, 256, 8, 8]", squeeze_37: "f32[256]", convolution_13: "f32[1024, 256, 8, 8]", squeeze_40: "f32[256]", relu_10: "f32[1024, 256, 8, 8]", convolution_14: "f32[1024, 256, 8, 8]", squeeze_43: "f32[256]", relu_11: "f32[1024, 256, 8, 8]", convolution_15: "f32[1024, 256, 8, 8]", squeeze_46: "f32[256]", relu_12: "f32[1024, 256, 8, 8]", convolution_16: "f32[1024, 512, 4, 4]", squeeze_49: "f32[512]", relu_13: "f32[1024, 512, 4, 4]", convolution_17: "f32[1024, 512, 4, 4]", squeeze_52: "f32[512]", convolution_18: "f32[1024, 512, 4, 4]", squeeze_55: "f32[512]", relu_14: "f32[1024, 512, 4, 4]", convolution_19: "f32[1024, 512, 4, 4]", squeeze_58: "f32[512]", relu_15: "f32[1024, 512, 4, 4]", convolution_20: "f32[1024, 512, 4, 4]", squeeze_61: "f32[512]", relu_16: "f32[1024, 512, 4, 4]", view: "f32[1024, 512]", sigmoid: "f32[1024, 65]", unsqueeze_86: "f32[1, 512, 1, 1]", unsqueeze_98: "f32[1, 512, 1, 1]", unsqueeze_110: "f32[1, 512, 1, 1]", unsqueeze_122: "f32[1, 512, 1, 1]", unsqueeze_134: "f32[1, 512, 1, 1]", unsqueeze_146: "f32[1, 256, 1, 1]", unsqueeze_158: "f32[1, 256, 1, 1]", unsqueeze_170: "f32[1, 256, 1, 1]", unsqueeze_182: "f32[1, 256, 1, 1]", unsqueeze_194: "f32[1, 256, 1, 1]", unsqueeze_206: "f32[1, 128, 1, 1]", unsqueeze_218: "f32[1, 128, 1, 1]", unsqueeze_230: "f32[1, 128, 1, 1]", unsqueeze_242: "f32[1, 128, 1, 1]", unsqueeze_254: "f32[1, 128, 1, 1]", unsqueeze_266: "f32[1, 64, 1, 1]", unsqueeze_278: "f32[1, 64, 1, 1]", unsqueeze_290: "f32[1, 64, 1, 1]", unsqueeze_302: "f32[1, 64, 1, 1]", unsqueeze_314: "f32[1, 64, 1, 1]", unsqueeze_326: "f32[1, 64, 1, 1]", tangents_1: "f32[1024, 65]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:136 in forward, code: x = torch.sigmoid(x)
        sub_21: "f32[1024, 65]" = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_147: "f32[1024, 65]" = torch.ops.aten.mul.Tensor(sigmoid, sub_21);  sigmoid = sub_21 = None
        mul_148: "f32[1024, 65]" = torch.ops.aten.mul.Tensor(tangents_1, mul_147);  tangents_1 = mul_147 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:135 in forward, code: x = self.fc(x)
        permute: "f32[512, 65]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_1: "f32[65, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[1024, 512]" = torch.ops.aten.mm.default(mul_148, permute_1);  permute_1 = None
        permute_2: "f32[65, 1024]" = torch.ops.aten.permute.default(mul_148, [1, 0])
        mm_1: "f32[65, 512]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 65]" = torch.ops.aten.sum.dim_IntList(mul_148, [0], True);  mul_148 = None
        view_1: "f32[65]" = torch.ops.aten.reshape.default(sum_1, [65]);  sum_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:134 in forward, code: x = x.view(x.size(0), -1)
        view_2: "f32[1024, 512, 1, 1]" = torch.ops.aten.reshape.default(mm, [1024, 512, 1, 1]);  mm = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:133 in forward, code: x = F.avg_pool2d(x, 4)
        avg_pool2d_backward: "f32[1024, 512, 4, 4]" = torch.ops.aten.avg_pool2d_backward.default(view_2, relu_16, [4, 4], [], [0, 0], False, True, None);  view_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        le: "b8[1024, 512, 4, 4]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1024, 512, 4, 4]" = torch.ops.aten.where.self(le, full_default, avg_pool2d_backward);  le = avg_pool2d_backward = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        sum_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_22: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_86);  convolution_20 = unsqueeze_86 = None
        mul_149: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(where, sub_22)
        sum_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_149, [0, 2, 3]);  mul_149 = None
        mul_150: "f32[512]" = torch.ops.aten.mul.Tensor(sum_2, 6.103515625e-05)
        unsqueeze_87: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_150, 0);  mul_150 = None
        unsqueeze_88: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
        unsqueeze_89: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 3);  unsqueeze_88 = None
        mul_151: "f32[512]" = torch.ops.aten.mul.Tensor(sum_3, 6.103515625e-05)
        mul_152: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_153: "f32[512]" = torch.ops.aten.mul.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_90: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_153, 0);  mul_153 = None
        unsqueeze_91: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 2);  unsqueeze_90 = None
        unsqueeze_92: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 3);  unsqueeze_91 = None
        mul_154: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_126);  primals_126 = None
        unsqueeze_93: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_154, 0);  mul_154 = None
        unsqueeze_94: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
        unsqueeze_95: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, 3);  unsqueeze_94 = None
        mul_155: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_92);  sub_22 = unsqueeze_92 = None
        sub_24: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(where, mul_155);  mul_155 = None
        sub_25: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(sub_24, unsqueeze_89);  sub_24 = unsqueeze_89 = None
        mul_156: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_95);  sub_25 = unsqueeze_95 = None
        mul_157: "f32[512]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_61);  sum_3 = squeeze_61 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_156, relu_15, primals_122, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_156 = primals_122 = None
        getitem_42: "f32[1024, 512, 4, 4]" = convolution_backward[0]
        getitem_43: "f32[512, 512, 3, 3]" = convolution_backward[1];  convolution_backward = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        le_1: "b8[1024, 512, 4, 4]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_1: "f32[1024, 512, 4, 4]" = torch.ops.aten.where.self(le_1, full_default, getitem_42);  le_1 = getitem_42 = None
        sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_26: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_98);  convolution_19 = unsqueeze_98 = None
        mul_158: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(where_1, sub_26)
        sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 2, 3]);  mul_158 = None
        mul_159: "f32[512]" = torch.ops.aten.mul.Tensor(sum_4, 6.103515625e-05)
        unsqueeze_99: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_159, 0);  mul_159 = None
        unsqueeze_100: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
        unsqueeze_101: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, 3);  unsqueeze_100 = None
        mul_160: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, 6.103515625e-05)
        mul_161: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_162: "f32[512]" = torch.ops.aten.mul.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
        unsqueeze_102: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_162, 0);  mul_162 = None
        unsqueeze_103: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 2);  unsqueeze_102 = None
        unsqueeze_104: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 3);  unsqueeze_103 = None
        mul_163: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_120);  primals_120 = None
        unsqueeze_105: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_163, 0);  mul_163 = None
        unsqueeze_106: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
        unsqueeze_107: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 3);  unsqueeze_106 = None
        mul_164: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_104);  sub_26 = unsqueeze_104 = None
        sub_28: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(where_1, mul_164);  where_1 = mul_164 = None
        sub_29: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(sub_28, unsqueeze_101);  sub_28 = unsqueeze_101 = None
        mul_165: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_107);  sub_29 = unsqueeze_107 = None
        mul_166: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_58);  sum_5 = squeeze_58 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_165, relu_14, primals_116, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_165 = primals_116 = None
        getitem_45: "f32[1024, 512, 4, 4]" = convolution_backward_1[0]
        getitem_46: "f32[512, 512, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
        add_113: "f32[1024, 512, 4, 4]" = torch.ops.aten.add.Tensor(where, getitem_45);  where = getitem_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        le_2: "b8[1024, 512, 4, 4]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_2: "f32[1024, 512, 4, 4]" = torch.ops.aten.where.self(le_2, full_default, add_113);  le_2 = add_113 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_30: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_110);  convolution_18 = unsqueeze_110 = None
        mul_167: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(where_2, sub_30)
        sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_167, [0, 2, 3]);  mul_167 = None
        mul_168: "f32[512]" = torch.ops.aten.mul.Tensor(sum_6, 6.103515625e-05)
        unsqueeze_111: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_168, 0);  mul_168 = None
        unsqueeze_112: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
        unsqueeze_113: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 3);  unsqueeze_112 = None
        mul_169: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, 6.103515625e-05)
        mul_170: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_171: "f32[512]" = torch.ops.aten.mul.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        unsqueeze_114: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_171, 0);  mul_171 = None
        unsqueeze_115: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
        unsqueeze_116: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 3);  unsqueeze_115 = None
        mul_172: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_117: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_172, 0);  mul_172 = None
        unsqueeze_118: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
        unsqueeze_119: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
        mul_173: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_116);  sub_30 = unsqueeze_116 = None
        sub_32: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(where_2, mul_173);  mul_173 = None
        sub_33: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(sub_32, unsqueeze_113);  sub_32 = unsqueeze_113 = None
        mul_174: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_119);  sub_33 = unsqueeze_119 = None
        mul_175: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_55);  sum_7 = squeeze_55 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_174, relu_12, primals_110, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_174 = primals_110 = None
        getitem_48: "f32[1024, 256, 8, 8]" = convolution_backward_2[0]
        getitem_49: "f32[512, 256, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        sum_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_34: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_122);  convolution_17 = unsqueeze_122 = None
        mul_176: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(where_2, sub_34)
        sum_9: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 2, 3]);  mul_176 = None
        mul_177: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, 6.103515625e-05)
        unsqueeze_123: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_177, 0);  mul_177 = None
        unsqueeze_124: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
        unsqueeze_125: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 3);  unsqueeze_124 = None
        mul_178: "f32[512]" = torch.ops.aten.mul.Tensor(sum_9, 6.103515625e-05)
        mul_179: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_180: "f32[512]" = torch.ops.aten.mul.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
        unsqueeze_126: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_180, 0);  mul_180 = None
        unsqueeze_127: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 2);  unsqueeze_126 = None
        unsqueeze_128: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 3);  unsqueeze_127 = None
        mul_181: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_129: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_181, 0);  mul_181 = None
        unsqueeze_130: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
        unsqueeze_131: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 3);  unsqueeze_130 = None
        mul_182: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_128);  sub_34 = unsqueeze_128 = None
        sub_36: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(where_2, mul_182);  where_2 = mul_182 = None
        sub_37: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(sub_36, unsqueeze_125);  sub_36 = unsqueeze_125 = None
        mul_183: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_131);  sub_37 = unsqueeze_131 = None
        mul_184: "f32[512]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_52);  sum_9 = squeeze_52 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_183, relu_13, primals_104, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_183 = primals_104 = None
        getitem_51: "f32[1024, 512, 4, 4]" = convolution_backward_3[0]
        getitem_52: "f32[512, 512, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        le_3: "b8[1024, 512, 4, 4]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_3: "f32[1024, 512, 4, 4]" = torch.ops.aten.where.self(le_3, full_default, getitem_51);  le_3 = getitem_51 = None
        sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_38: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_134);  convolution_16 = unsqueeze_134 = None
        mul_185: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(where_3, sub_38)
        sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_185, [0, 2, 3]);  mul_185 = None
        mul_186: "f32[512]" = torch.ops.aten.mul.Tensor(sum_10, 6.103515625e-05)
        unsqueeze_135: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_186, 0);  mul_186 = None
        unsqueeze_136: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
        unsqueeze_137: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 3);  unsqueeze_136 = None
        mul_187: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, 6.103515625e-05)
        mul_188: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_189: "f32[512]" = torch.ops.aten.mul.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
        unsqueeze_138: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_189, 0);  mul_189 = None
        unsqueeze_139: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 2);  unsqueeze_138 = None
        unsqueeze_140: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 3);  unsqueeze_139 = None
        mul_190: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_141: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_190, 0);  mul_190 = None
        unsqueeze_142: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None
        unsqueeze_143: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 3);  unsqueeze_142 = None
        mul_191: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_140);  sub_38 = unsqueeze_140 = None
        sub_40: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(where_3, mul_191);  where_3 = mul_191 = None
        sub_41: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(sub_40, unsqueeze_137);  sub_40 = unsqueeze_137 = None
        mul_192: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_143);  sub_41 = unsqueeze_143 = None
        mul_193: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_49);  sum_11 = squeeze_49 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_192, relu_12, primals_98, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_192 = primals_98 = None
        getitem_54: "f32[1024, 256, 8, 8]" = convolution_backward_4[0]
        getitem_55: "f32[512, 256, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
        add_114: "f32[1024, 256, 8, 8]" = torch.ops.aten.add.Tensor(getitem_48, getitem_54);  getitem_48 = getitem_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        le_4: "b8[1024, 256, 8, 8]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_4: "f32[1024, 256, 8, 8]" = torch.ops.aten.where.self(le_4, full_default, add_114);  le_4 = add_114 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        sum_12: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_42: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_146);  convolution_15 = unsqueeze_146 = None
        mul_194: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(where_4, sub_42)
        sum_13: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_194, [0, 2, 3]);  mul_194 = None
        mul_195: "f32[256]" = torch.ops.aten.mul.Tensor(sum_12, 1.52587890625e-05)
        unsqueeze_147: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_195, 0);  mul_195 = None
        unsqueeze_148: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
        unsqueeze_149: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 3);  unsqueeze_148 = None
        mul_196: "f32[256]" = torch.ops.aten.mul.Tensor(sum_13, 1.52587890625e-05)
        mul_197: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_198: "f32[256]" = torch.ops.aten.mul.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
        unsqueeze_150: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_198, 0);  mul_198 = None
        unsqueeze_151: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 2);  unsqueeze_150 = None
        unsqueeze_152: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 3);  unsqueeze_151 = None
        mul_199: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_153: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_199, 0);  mul_199 = None
        unsqueeze_154: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 2);  unsqueeze_153 = None
        unsqueeze_155: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 3);  unsqueeze_154 = None
        mul_200: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_152);  sub_42 = unsqueeze_152 = None
        sub_44: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(where_4, mul_200);  mul_200 = None
        sub_45: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(sub_44, unsqueeze_149);  sub_44 = unsqueeze_149 = None
        mul_201: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_155);  sub_45 = unsqueeze_155 = None
        mul_202: "f32[256]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_46);  sum_13 = squeeze_46 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_201, relu_11, primals_92, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_201 = primals_92 = None
        getitem_57: "f32[1024, 256, 8, 8]" = convolution_backward_5[0]
        getitem_58: "f32[256, 256, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        le_5: "b8[1024, 256, 8, 8]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_5: "f32[1024, 256, 8, 8]" = torch.ops.aten.where.self(le_5, full_default, getitem_57);  le_5 = getitem_57 = None
        sum_14: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_46: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_158);  convolution_14 = unsqueeze_158 = None
        mul_203: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(where_5, sub_46)
        sum_15: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_203, [0, 2, 3]);  mul_203 = None
        mul_204: "f32[256]" = torch.ops.aten.mul.Tensor(sum_14, 1.52587890625e-05)
        unsqueeze_159: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_204, 0);  mul_204 = None
        unsqueeze_160: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
        unsqueeze_161: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 3);  unsqueeze_160 = None
        mul_205: "f32[256]" = torch.ops.aten.mul.Tensor(sum_15, 1.52587890625e-05)
        mul_206: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_207: "f32[256]" = torch.ops.aten.mul.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
        unsqueeze_162: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_207, 0);  mul_207 = None
        unsqueeze_163: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 2);  unsqueeze_162 = None
        unsqueeze_164: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 3);  unsqueeze_163 = None
        mul_208: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_165: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_208, 0);  mul_208 = None
        unsqueeze_166: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
        unsqueeze_167: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 3);  unsqueeze_166 = None
        mul_209: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_164);  sub_46 = unsqueeze_164 = None
        sub_48: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(where_5, mul_209);  where_5 = mul_209 = None
        sub_49: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(sub_48, unsqueeze_161);  sub_48 = unsqueeze_161 = None
        mul_210: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_167);  sub_49 = unsqueeze_167 = None
        mul_211: "f32[256]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_43);  sum_15 = squeeze_43 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_210, relu_10, primals_86, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_210 = primals_86 = None
        getitem_60: "f32[1024, 256, 8, 8]" = convolution_backward_6[0]
        getitem_61: "f32[256, 256, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
        add_115: "f32[1024, 256, 8, 8]" = torch.ops.aten.add.Tensor(where_4, getitem_60);  where_4 = getitem_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        le_6: "b8[1024, 256, 8, 8]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_6: "f32[1024, 256, 8, 8]" = torch.ops.aten.where.self(le_6, full_default, add_115);  le_6 = add_115 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        sum_16: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_50: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_170);  convolution_13 = unsqueeze_170 = None
        mul_212: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(where_6, sub_50)
        sum_17: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_212, [0, 2, 3]);  mul_212 = None
        mul_213: "f32[256]" = torch.ops.aten.mul.Tensor(sum_16, 1.52587890625e-05)
        unsqueeze_171: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_213, 0);  mul_213 = None
        unsqueeze_172: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
        unsqueeze_173: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 3);  unsqueeze_172 = None
        mul_214: "f32[256]" = torch.ops.aten.mul.Tensor(sum_17, 1.52587890625e-05)
        mul_215: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_216: "f32[256]" = torch.ops.aten.mul.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_174: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_216, 0);  mul_216 = None
        unsqueeze_175: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 2);  unsqueeze_174 = None
        unsqueeze_176: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 3);  unsqueeze_175 = None
        mul_217: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_177: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_217, 0);  mul_217 = None
        unsqueeze_178: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
        unsqueeze_179: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
        mul_218: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_176);  sub_50 = unsqueeze_176 = None
        sub_52: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(where_6, mul_218);  mul_218 = None
        sub_53: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(sub_52, unsqueeze_173);  sub_52 = unsqueeze_173 = None
        mul_219: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_179);  sub_53 = unsqueeze_179 = None
        mul_220: "f32[256]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_40);  sum_17 = squeeze_40 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_219, relu_8, primals_80, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_219 = primals_80 = None
        getitem_63: "f32[1024, 128, 16, 16]" = convolution_backward_7[0]
        getitem_64: "f32[256, 128, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        sum_18: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_54: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_182);  convolution_12 = unsqueeze_182 = None
        mul_221: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(where_6, sub_54)
        sum_19: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_221, [0, 2, 3]);  mul_221 = None
        mul_222: "f32[256]" = torch.ops.aten.mul.Tensor(sum_18, 1.52587890625e-05)
        unsqueeze_183: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_222, 0);  mul_222 = None
        unsqueeze_184: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
        unsqueeze_185: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 3);  unsqueeze_184 = None
        mul_223: "f32[256]" = torch.ops.aten.mul.Tensor(sum_19, 1.52587890625e-05)
        mul_224: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_225: "f32[256]" = torch.ops.aten.mul.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
        unsqueeze_186: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_225, 0);  mul_225 = None
        unsqueeze_187: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 2);  unsqueeze_186 = None
        unsqueeze_188: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 3);  unsqueeze_187 = None
        mul_226: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_189: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_226, 0);  mul_226 = None
        unsqueeze_190: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
        unsqueeze_191: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 3);  unsqueeze_190 = None
        mul_227: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_188);  sub_54 = unsqueeze_188 = None
        sub_56: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(where_6, mul_227);  where_6 = mul_227 = None
        sub_57: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_185);  sub_56 = unsqueeze_185 = None
        mul_228: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_191);  sub_57 = unsqueeze_191 = None
        mul_229: "f32[256]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_37);  sum_19 = squeeze_37 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_228, relu_9, primals_74, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_228 = primals_74 = None
        getitem_66: "f32[1024, 256, 8, 8]" = convolution_backward_8[0]
        getitem_67: "f32[256, 256, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        le_7: "b8[1024, 256, 8, 8]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_7: "f32[1024, 256, 8, 8]" = torch.ops.aten.where.self(le_7, full_default, getitem_66);  le_7 = getitem_66 = None
        sum_20: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_58: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_194);  convolution_11 = unsqueeze_194 = None
        mul_230: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(where_7, sub_58)
        sum_21: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 2, 3]);  mul_230 = None
        mul_231: "f32[256]" = torch.ops.aten.mul.Tensor(sum_20, 1.52587890625e-05)
        unsqueeze_195: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_231, 0);  mul_231 = None
        unsqueeze_196: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
        unsqueeze_197: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 3);  unsqueeze_196 = None
        mul_232: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, 1.52587890625e-05)
        mul_233: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_234: "f32[256]" = torch.ops.aten.mul.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        unsqueeze_198: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_234, 0);  mul_234 = None
        unsqueeze_199: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 2);  unsqueeze_198 = None
        unsqueeze_200: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 3);  unsqueeze_199 = None
        mul_235: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_201: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_235, 0);  mul_235 = None
        unsqueeze_202: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
        unsqueeze_203: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 3);  unsqueeze_202 = None
        mul_236: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_200);  sub_58 = unsqueeze_200 = None
        sub_60: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(where_7, mul_236);  where_7 = mul_236 = None
        sub_61: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_197);  sub_60 = unsqueeze_197 = None
        mul_237: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_203);  sub_61 = unsqueeze_203 = None
        mul_238: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_34);  sum_21 = squeeze_34 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_237, relu_8, primals_68, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_237 = primals_68 = None
        getitem_69: "f32[1024, 128, 16, 16]" = convolution_backward_9[0]
        getitem_70: "f32[256, 128, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
        add_116: "f32[1024, 128, 16, 16]" = torch.ops.aten.add.Tensor(getitem_63, getitem_69);  getitem_63 = getitem_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        le_8: "b8[1024, 128, 16, 16]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_8: "f32[1024, 128, 16, 16]" = torch.ops.aten.where.self(le_8, full_default, add_116);  le_8 = add_116 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        sum_22: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_62: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_206);  convolution_10 = unsqueeze_206 = None
        mul_239: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(where_8, sub_62)
        sum_23: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 2, 3]);  mul_239 = None
        mul_240: "f32[128]" = torch.ops.aten.mul.Tensor(sum_22, 3.814697265625e-06)
        unsqueeze_207: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_240, 0);  mul_240 = None
        unsqueeze_208: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
        unsqueeze_209: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 3);  unsqueeze_208 = None
        mul_241: "f32[128]" = torch.ops.aten.mul.Tensor(sum_23, 3.814697265625e-06)
        mul_242: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_243: "f32[128]" = torch.ops.aten.mul.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
        unsqueeze_210: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_243, 0);  mul_243 = None
        unsqueeze_211: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 2);  unsqueeze_210 = None
        unsqueeze_212: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 3);  unsqueeze_211 = None
        mul_244: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_213: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_244, 0);  mul_244 = None
        unsqueeze_214: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
        unsqueeze_215: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 3);  unsqueeze_214 = None
        mul_245: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_212);  sub_62 = unsqueeze_212 = None
        sub_64: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(where_8, mul_245);  mul_245 = None
        sub_65: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_64, unsqueeze_209);  sub_64 = unsqueeze_209 = None
        mul_246: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_215);  sub_65 = unsqueeze_215 = None
        mul_247: "f32[128]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_31);  sum_23 = squeeze_31 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_246, relu_7, primals_62, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_246 = primals_62 = None
        getitem_72: "f32[1024, 128, 16, 16]" = convolution_backward_10[0]
        getitem_73: "f32[128, 128, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        le_9: "b8[1024, 128, 16, 16]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_9: "f32[1024, 128, 16, 16]" = torch.ops.aten.where.self(le_9, full_default, getitem_72);  le_9 = getitem_72 = None
        sum_24: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_66: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_218);  convolution_9 = unsqueeze_218 = None
        mul_248: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(where_9, sub_66)
        sum_25: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 2, 3]);  mul_248 = None
        mul_249: "f32[128]" = torch.ops.aten.mul.Tensor(sum_24, 3.814697265625e-06)
        unsqueeze_219: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_249, 0);  mul_249 = None
        unsqueeze_220: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
        unsqueeze_221: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 3);  unsqueeze_220 = None
        mul_250: "f32[128]" = torch.ops.aten.mul.Tensor(sum_25, 3.814697265625e-06)
        mul_251: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_252: "f32[128]" = torch.ops.aten.mul.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
        unsqueeze_222: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_252, 0);  mul_252 = None
        unsqueeze_223: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
        unsqueeze_224: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 3);  unsqueeze_223 = None
        mul_253: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_225: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_253, 0);  mul_253 = None
        unsqueeze_226: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
        unsqueeze_227: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 3);  unsqueeze_226 = None
        mul_254: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_224);  sub_66 = unsqueeze_224 = None
        sub_68: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(where_9, mul_254);  where_9 = mul_254 = None
        sub_69: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_221);  sub_68 = unsqueeze_221 = None
        mul_255: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_227);  sub_69 = unsqueeze_227 = None
        mul_256: "f32[128]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_28);  sum_25 = squeeze_28 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_255, relu_6, primals_56, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_255 = primals_56 = None
        getitem_75: "f32[1024, 128, 16, 16]" = convolution_backward_11[0]
        getitem_76: "f32[128, 128, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
        add_117: "f32[1024, 128, 16, 16]" = torch.ops.aten.add.Tensor(where_8, getitem_75);  where_8 = getitem_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        le_10: "b8[1024, 128, 16, 16]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_10: "f32[1024, 128, 16, 16]" = torch.ops.aten.where.self(le_10, full_default, add_117);  le_10 = add_117 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        sum_26: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_70: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_230);  convolution_8 = unsqueeze_230 = None
        mul_257: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(where_10, sub_70)
        sum_27: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 2, 3]);  mul_257 = None
        mul_258: "f32[128]" = torch.ops.aten.mul.Tensor(sum_26, 3.814697265625e-06)
        unsqueeze_231: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_258, 0);  mul_258 = None
        unsqueeze_232: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
        unsqueeze_233: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
        mul_259: "f32[128]" = torch.ops.aten.mul.Tensor(sum_27, 3.814697265625e-06)
        mul_260: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_261: "f32[128]" = torch.ops.aten.mul.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
        unsqueeze_234: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_261, 0);  mul_261 = None
        unsqueeze_235: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
        unsqueeze_236: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
        mul_262: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_237: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_262, 0);  mul_262 = None
        unsqueeze_238: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
        unsqueeze_239: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
        mul_263: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_236);  sub_70 = unsqueeze_236 = None
        sub_72: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(where_10, mul_263);  mul_263 = None
        sub_73: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_233);  sub_72 = unsqueeze_233 = None
        mul_264: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_239);  sub_73 = unsqueeze_239 = None
        mul_265: "f32[128]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_25);  sum_27 = squeeze_25 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_264, relu_4, primals_50, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_264 = primals_50 = None
        getitem_78: "f32[1024, 64, 32, 32]" = convolution_backward_12[0]
        getitem_79: "f32[128, 64, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        sum_28: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_74: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_242);  convolution_7 = unsqueeze_242 = None
        mul_266: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(where_10, sub_74)
        sum_29: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 2, 3]);  mul_266 = None
        mul_267: "f32[128]" = torch.ops.aten.mul.Tensor(sum_28, 3.814697265625e-06)
        unsqueeze_243: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_267, 0);  mul_267 = None
        unsqueeze_244: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
        unsqueeze_245: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
        mul_268: "f32[128]" = torch.ops.aten.mul.Tensor(sum_29, 3.814697265625e-06)
        mul_269: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_270: "f32[128]" = torch.ops.aten.mul.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
        unsqueeze_246: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_270, 0);  mul_270 = None
        unsqueeze_247: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
        unsqueeze_248: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
        mul_271: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_249: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
        unsqueeze_250: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
        unsqueeze_251: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
        mul_272: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_248);  sub_74 = unsqueeze_248 = None
        sub_76: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(where_10, mul_272);  where_10 = mul_272 = None
        sub_77: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_76, unsqueeze_245);  sub_76 = unsqueeze_245 = None
        mul_273: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_251);  sub_77 = unsqueeze_251 = None
        mul_274: "f32[128]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_22);  sum_29 = squeeze_22 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_273, relu_5, primals_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_273 = primals_44 = None
        getitem_81: "f32[1024, 128, 16, 16]" = convolution_backward_13[0]
        getitem_82: "f32[128, 128, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        le_11: "b8[1024, 128, 16, 16]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_11: "f32[1024, 128, 16, 16]" = torch.ops.aten.where.self(le_11, full_default, getitem_81);  le_11 = getitem_81 = None
        sum_30: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_78: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_254);  convolution_6 = unsqueeze_254 = None
        mul_275: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(where_11, sub_78)
        sum_31: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 2, 3]);  mul_275 = None
        mul_276: "f32[128]" = torch.ops.aten.mul.Tensor(sum_30, 3.814697265625e-06)
        unsqueeze_255: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_276, 0);  mul_276 = None
        unsqueeze_256: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
        unsqueeze_257: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
        mul_277: "f32[128]" = torch.ops.aten.mul.Tensor(sum_31, 3.814697265625e-06)
        mul_278: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_279: "f32[128]" = torch.ops.aten.mul.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_258: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_279, 0);  mul_279 = None
        unsqueeze_259: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 2);  unsqueeze_258 = None
        unsqueeze_260: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 3);  unsqueeze_259 = None
        mul_280: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_261: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_280, 0);  mul_280 = None
        unsqueeze_262: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
        unsqueeze_263: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
        mul_281: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_260);  sub_78 = unsqueeze_260 = None
        sub_80: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(where_11, mul_281);  where_11 = mul_281 = None
        sub_81: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_257);  sub_80 = unsqueeze_257 = None
        mul_282: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_263);  sub_81 = unsqueeze_263 = None
        mul_283: "f32[128]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_19);  sum_31 = squeeze_19 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_282, relu_4, primals_38, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_282 = primals_38 = None
        getitem_84: "f32[1024, 64, 32, 32]" = convolution_backward_14[0]
        getitem_85: "f32[128, 64, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
        add_118: "f32[1024, 64, 32, 32]" = torch.ops.aten.add.Tensor(getitem_78, getitem_84);  getitem_78 = getitem_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        le_12: "b8[1024, 64, 32, 32]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_12: "f32[1024, 64, 32, 32]" = torch.ops.aten.where.self(le_12, full_default, add_118);  le_12 = add_118 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        sum_32: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_82: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_266);  convolution_5 = unsqueeze_266 = None
        mul_284: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(where_12, sub_82)
        sum_33: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_284, [0, 2, 3]);  mul_284 = None
        mul_285: "f32[64]" = torch.ops.aten.mul.Tensor(sum_32, 9.5367431640625e-07)
        unsqueeze_267: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_285, 0);  mul_285 = None
        unsqueeze_268: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
        unsqueeze_269: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
        mul_286: "f32[64]" = torch.ops.aten.mul.Tensor(sum_33, 9.5367431640625e-07)
        mul_287: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_288: "f32[64]" = torch.ops.aten.mul.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None
        unsqueeze_270: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_288, 0);  mul_288 = None
        unsqueeze_271: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
        unsqueeze_272: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
        mul_289: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_273: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_289, 0);  mul_289 = None
        unsqueeze_274: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
        unsqueeze_275: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
        mul_290: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_272);  sub_82 = unsqueeze_272 = None
        sub_84: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(where_12, mul_290);  mul_290 = None
        sub_85: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_269);  sub_84 = unsqueeze_269 = None
        mul_291: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_275);  sub_85 = unsqueeze_275 = None
        mul_292: "f32[64]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_16);  sum_33 = squeeze_16 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_291, relu_3, primals_32, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_291 = primals_32 = None
        getitem_87: "f32[1024, 64, 32, 32]" = convolution_backward_15[0]
        getitem_88: "f32[64, 64, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        le_13: "b8[1024, 64, 32, 32]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_13: "f32[1024, 64, 32, 32]" = torch.ops.aten.where.self(le_13, full_default, getitem_87);  le_13 = getitem_87 = None
        sum_34: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_86: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_278);  convolution_4 = unsqueeze_278 = None
        mul_293: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(where_13, sub_86)
        sum_35: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3]);  mul_293 = None
        mul_294: "f32[64]" = torch.ops.aten.mul.Tensor(sum_34, 9.5367431640625e-07)
        unsqueeze_279: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
        unsqueeze_280: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
        unsqueeze_281: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
        mul_295: "f32[64]" = torch.ops.aten.mul.Tensor(sum_35, 9.5367431640625e-07)
        mul_296: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_297: "f32[64]" = torch.ops.aten.mul.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        unsqueeze_282: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_297, 0);  mul_297 = None
        unsqueeze_283: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
        unsqueeze_284: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
        mul_298: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_285: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
        unsqueeze_286: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
        unsqueeze_287: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
        mul_299: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_284);  sub_86 = unsqueeze_284 = None
        sub_88: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(where_13, mul_299);  where_13 = mul_299 = None
        sub_89: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(sub_88, unsqueeze_281);  sub_88 = unsqueeze_281 = None
        mul_300: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_287);  sub_89 = unsqueeze_287 = None
        mul_301: "f32[64]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_13);  sum_35 = squeeze_13 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_300, relu_2, primals_26, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_300 = primals_26 = None
        getitem_90: "f32[1024, 64, 32, 32]" = convolution_backward_16[0]
        getitem_91: "f32[64, 64, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
        add_119: "f32[1024, 64, 32, 32]" = torch.ops.aten.add.Tensor(where_12, getitem_90);  where_12 = getitem_90 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        le_14: "b8[1024, 64, 32, 32]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_14: "f32[1024, 64, 32, 32]" = torch.ops.aten.where.self(le_14, full_default, add_119);  le_14 = add_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        sum_36: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_90: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_290);  convolution_3 = unsqueeze_290 = None
        mul_302: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(where_14, sub_90)
        sum_37: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 2, 3]);  mul_302 = None
        mul_303: "f32[64]" = torch.ops.aten.mul.Tensor(sum_36, 9.5367431640625e-07)
        unsqueeze_291: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_303, 0);  mul_303 = None
        unsqueeze_292: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
        unsqueeze_293: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
        mul_304: "f32[64]" = torch.ops.aten.mul.Tensor(sum_37, 9.5367431640625e-07)
        mul_305: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_306: "f32[64]" = torch.ops.aten.mul.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
        unsqueeze_294: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_306, 0);  mul_306 = None
        unsqueeze_295: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
        unsqueeze_296: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
        mul_307: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_297: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
        unsqueeze_298: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
        unsqueeze_299: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
        mul_308: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_296);  sub_90 = unsqueeze_296 = None
        sub_92: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(where_14, mul_308);  mul_308 = None
        sub_93: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(sub_92, unsqueeze_293);  sub_92 = unsqueeze_293 = None
        mul_309: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_299);  sub_93 = unsqueeze_299 = None
        mul_310: "f32[64]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_10);  sum_37 = squeeze_10 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_309, relu, primals_20, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_309 = primals_20 = None
        getitem_93: "f32[1024, 64, 64, 64]" = convolution_backward_17[0]
        getitem_94: "f32[64, 64, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        sum_38: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_94: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_302);  convolution_2 = unsqueeze_302 = None
        mul_311: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(where_14, sub_94)
        sum_39: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 2, 3]);  mul_311 = None
        mul_312: "f32[64]" = torch.ops.aten.mul.Tensor(sum_38, 9.5367431640625e-07)
        unsqueeze_303: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_312, 0);  mul_312 = None
        unsqueeze_304: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
        unsqueeze_305: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
        mul_313: "f32[64]" = torch.ops.aten.mul.Tensor(sum_39, 9.5367431640625e-07)
        mul_314: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_315: "f32[64]" = torch.ops.aten.mul.Tensor(mul_313, mul_314);  mul_313 = mul_314 = None
        unsqueeze_306: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_315, 0);  mul_315 = None
        unsqueeze_307: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 2);  unsqueeze_306 = None
        unsqueeze_308: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 3);  unsqueeze_307 = None
        mul_316: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_309: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
        unsqueeze_310: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
        unsqueeze_311: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
        mul_317: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_308);  sub_94 = unsqueeze_308 = None
        sub_96: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(where_14, mul_317);  where_14 = mul_317 = None
        sub_97: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_305);  sub_96 = unsqueeze_305 = None
        mul_318: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_311);  sub_97 = unsqueeze_311 = None
        mul_319: "f32[64]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_7);  sum_39 = squeeze_7 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_318, relu_1, primals_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_318 = primals_14 = None
        getitem_96: "f32[1024, 64, 32, 32]" = convolution_backward_18[0]
        getitem_97: "f32[64, 64, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        le_15: "b8[1024, 64, 32, 32]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_15: "f32[1024, 64, 32, 32]" = torch.ops.aten.where.self(le_15, full_default, getitem_96);  le_15 = getitem_96 = None
        sum_40: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_98: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_314);  convolution_1 = unsqueeze_314 = None
        mul_320: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(where_15, sub_98)
        sum_41: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 2, 3]);  mul_320 = None
        mul_321: "f32[64]" = torch.ops.aten.mul.Tensor(sum_40, 9.5367431640625e-07)
        unsqueeze_315: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_321, 0);  mul_321 = None
        unsqueeze_316: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
        unsqueeze_317: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 3);  unsqueeze_316 = None
        mul_322: "f32[64]" = torch.ops.aten.mul.Tensor(sum_41, 9.5367431640625e-07)
        mul_323: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_324: "f32[64]" = torch.ops.aten.mul.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
        unsqueeze_318: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_324, 0);  mul_324 = None
        unsqueeze_319: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 2);  unsqueeze_318 = None
        unsqueeze_320: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 3);  unsqueeze_319 = None
        mul_325: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_321: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_325, 0);  mul_325 = None
        unsqueeze_322: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
        unsqueeze_323: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
        mul_326: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_320);  sub_98 = unsqueeze_320 = None
        sub_100: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(where_15, mul_326);  where_15 = mul_326 = None
        sub_101: "f32[1024, 64, 32, 32]" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_317);  sub_100 = unsqueeze_317 = None
        mul_327: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_323);  sub_101 = unsqueeze_323 = None
        mul_328: "f32[64]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_4);  sum_41 = squeeze_4 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_327, relu, primals_8, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_327 = primals_8 = None
        getitem_99: "f32[1024, 64, 64, 64]" = convolution_backward_19[0]
        getitem_100: "f32[64, 64, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
        add_120: "f32[1024, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_93, getitem_99);  getitem_93 = getitem_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:128 in forward, code: x = F.relu(self.bn1(self.conv1(x)))
        le_16: "b8[1024, 64, 64, 64]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_16: "f32[1024, 64, 64, 64]" = torch.ops.aten.where.self(le_16, full_default, add_120);  le_16 = full_default = add_120 = None
        sum_42: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_102: "f32[1024, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_326);  convolution = unsqueeze_326 = None
        mul_329: "f32[1024, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_16, sub_102)
        sum_43: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 2, 3]);  mul_329 = None
        mul_330: "f32[64]" = torch.ops.aten.mul.Tensor(sum_42, 2.384185791015625e-07)
        unsqueeze_327: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_330, 0);  mul_330 = None
        unsqueeze_328: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
        unsqueeze_329: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 3);  unsqueeze_328 = None
        mul_331: "f32[64]" = torch.ops.aten.mul.Tensor(sum_43, 2.384185791015625e-07)
        mul_332: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_333: "f32[64]" = torch.ops.aten.mul.Tensor(mul_331, mul_332);  mul_331 = mul_332 = None
        unsqueeze_330: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_333, 0);  mul_333 = None
        unsqueeze_331: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 2);  unsqueeze_330 = None
        unsqueeze_332: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 3);  unsqueeze_331 = None
        mul_334: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_333: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
        unsqueeze_334: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
        unsqueeze_335: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
        mul_335: "f32[1024, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_332);  sub_102 = unsqueeze_332 = None
        sub_104: "f32[1024, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_16, mul_335);  where_16 = mul_335 = None
        sub_105: "f32[1024, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_329);  sub_104 = unsqueeze_329 = None
        mul_336: "f32[1024, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_335);  sub_105 = unsqueeze_335 = None
        mul_337: "f32[64]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_1);  sum_43 = squeeze_1 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_336, primals_2, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_336 = primals_2 = primals_1 = None
        getitem_103: "f32[64, 9, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
        return (getitem_103, None, None, None, None, mul_337, sum_42, getitem_100, None, None, None, mul_328, sum_40, getitem_97, None, None, None, mul_319, sum_38, getitem_94, None, None, None, mul_310, sum_36, getitem_91, None, None, None, mul_301, sum_34, getitem_88, None, None, None, mul_292, sum_32, getitem_85, None, None, None, mul_283, sum_30, getitem_82, None, None, None, mul_274, sum_28, getitem_79, None, None, None, mul_265, sum_26, getitem_76, None, None, None, mul_256, sum_24, getitem_73, None, None, None, mul_247, sum_22, getitem_70, None, None, None, mul_238, sum_20, getitem_67, None, None, None, mul_229, sum_18, getitem_64, None, None, None, mul_220, sum_16, getitem_61, None, None, None, mul_211, sum_14, getitem_58, None, None, None, mul_202, sum_12, getitem_55, None, None, None, mul_193, sum_10, getitem_52, None, None, None, mul_184, sum_8, getitem_49, None, None, None, mul_175, sum_6, getitem_46, None, None, None, mul_166, sum_4, getitem_43, None, None, None, mul_157, sum_2, mm_1, view_1)
