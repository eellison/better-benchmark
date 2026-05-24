import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[32, 3, 224, 224]", primals_3: "i64[]", primals_4: "f32[32]", primals_5: "f32[32]", primals_6: "f32[32]", primals_7: "f32[32]", primals_8: "f32[32, 32, 3, 3]", primals_9: "i64[]", primals_10: "f32[32]", primals_11: "f32[32]", primals_12: "f32[32]", primals_13: "f32[32]", primals_14: "f32[64, 32, 3, 3]", primals_15: "i64[]", primals_16: "f32[64]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[64]", primals_20: "f32[64, 64, 1, 1]", primals_21: "i64[]", primals_22: "f32[64]", primals_23: "f32[64]", primals_24: "f32[64]", primals_25: "f32[64]", primals_26: "f32[128, 32, 3, 3]", primals_27: "i64[]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[128]", primals_32: "f32[32, 64, 1, 1]", primals_33: "f32[32]", primals_34: "i64[]", primals_35: "f32[32]", primals_36: "f32[32]", primals_37: "f32[32]", primals_38: "f32[32]", primals_39: "f32[128, 32, 1, 1]", primals_40: "f32[128]", primals_41: "f32[256, 64, 1, 1]", primals_42: "i64[]", primals_43: "f32[256]", primals_44: "f32[256]", primals_45: "f32[256]", primals_46: "f32[256]", primals_47: "f32[256, 64, 1, 1]", primals_48: "i64[]", primals_49: "f32[256]", primals_50: "f32[256]", primals_51: "f32[256]", primals_52: "f32[256]", primals_53: "f32[128, 256, 1, 1]", primals_54: "i64[]", primals_55: "f32[128]", primals_56: "f32[128]", primals_57: "f32[128]", primals_58: "f32[128]", primals_59: "f32[256, 64, 3, 3]", primals_60: "i64[]", primals_61: "f32[256]", primals_62: "f32[256]", primals_63: "f32[256]", primals_64: "f32[256]", primals_65: "f32[64, 128, 1, 1]", primals_66: "f32[64]", primals_67: "i64[]", primals_68: "f32[64]", primals_69: "f32[64]", primals_70: "f32[64]", primals_71: "f32[64]", primals_72: "f32[256, 64, 1, 1]", primals_73: "f32[256]", primals_74: "f32[512, 128, 1, 1]", primals_75: "i64[]", primals_76: "f32[512]", primals_77: "f32[512]", primals_78: "f32[512]", primals_79: "f32[512]", primals_80: "f32[512, 256, 1, 1]", primals_81: "i64[]", primals_82: "f32[512]", primals_83: "f32[512]", primals_84: "f32[512]", primals_85: "f32[512]", primals_86: "f32[256, 512, 1, 1]", primals_87: "i64[]", primals_88: "f32[256]", primals_89: "f32[256]", primals_90: "f32[256]", primals_91: "f32[256]", primals_92: "f32[512, 128, 3, 3]", primals_93: "i64[]", primals_94: "f32[512]", primals_95: "f32[512]", primals_96: "f32[512]", primals_97: "f32[512]", primals_98: "f32[128, 256, 1, 1]", primals_99: "f32[128]", primals_100: "i64[]", primals_101: "f32[128]", primals_102: "f32[128]", primals_103: "f32[128]", primals_104: "f32[128]", primals_105: "f32[512, 128, 1, 1]", primals_106: "f32[512]", primals_107: "f32[1024, 256, 1, 1]", primals_108: "i64[]", primals_109: "f32[1024]", primals_110: "f32[1024]", primals_111: "f32[1024]", primals_112: "f32[1024]", primals_113: "f32[1024, 512, 1, 1]", primals_114: "i64[]", primals_115: "f32[1024]", primals_116: "f32[1024]", primals_117: "f32[1024]", primals_118: "f32[1024]", primals_119: "f32[512, 1024, 1, 1]", primals_120: "i64[]", primals_121: "f32[512]", primals_122: "f32[512]", primals_123: "f32[512]", primals_124: "f32[512]", primals_125: "f32[1024, 256, 3, 3]", primals_126: "i64[]", primals_127: "f32[1024]", primals_128: "f32[1024]", primals_129: "f32[1024]", primals_130: "f32[1024]", primals_131: "f32[256, 512, 1, 1]", primals_132: "f32[256]", primals_133: "i64[]", primals_134: "f32[256]", primals_135: "f32[256]", primals_136: "f32[256]", primals_137: "f32[256]", primals_138: "f32[1024, 256, 1, 1]", primals_139: "f32[1024]", primals_140: "f32[2048, 512, 1, 1]", primals_141: "i64[]", primals_142: "f32[2048]", primals_143: "f32[2048]", primals_144: "f32[2048]", primals_145: "f32[2048]", primals_146: "f32[2048, 1024, 1, 1]", primals_147: "i64[]", primals_148: "f32[2048]", primals_149: "f32[2048]", primals_150: "f32[2048]", primals_151: "f32[2048]", primals_152: "f32[1000, 2048]", primals_153: "f32[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:745 in forward_features, code: x = self.conv1(x)
        convolution: "f32[32, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_2, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000024912370735);  squeeze_2 = None
        mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[32, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        relu: "f32[32, 32, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
        convolution_1: "f32[32, 32, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_8, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000024912370735);  squeeze_5 = None
        mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[32, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        relu_1: "f32[32, 32, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
        convolution_2: "f32[32, 64, 112, 112]" = torch.ops.aten.convolution.default(relu_1, primals_14, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:746 in forward_features, code: x = self.bn1(x)
        add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3])
        mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000024912370735);  squeeze_8 = None
        mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
        unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[32, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:747 in forward_features, code: x = self.act1(x)
        relu_2: "f32[32, 64, 112, 112]" = torch.ops.aten.relu.default(add_14);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:748 in forward_features, code: x = self.maxpool(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_2, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu_2 = None
        getitem_6: "f32[32, 64, 56, 56]" = _low_memory_max_pool_with_offsets[0]
        getitem_7: "i8[32, 64, 56, 56]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:117 in forward, code: out = self.conv1(x)
        convolution_3: "f32[32, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_6, primals_20, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_21, 1)
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 64, 1, 1]" = var_mean_3[0]
        getitem_9: "f32[1, 64, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_21: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[64]" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_17: "f32[64]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
        mul_25: "f32[64]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[64]" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_18: "f32[64]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:119 in forward, code: out = self.act1(out)
        relu_3: "f32[32, 64, 56, 56]" = torch.ops.aten.relu.default(add_19);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:90 in forward, code: x = self.conv(x)
        convolution_4: "f32[32, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_26, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_27, 1)
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 128, 1, 1]" = var_mean_4[0]
        getitem_11: "f32[1, 128, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_21: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_4: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_28: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3])
        mul_29: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1);  squeeze_12 = None
        mul_30: "f32[128]" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_22: "f32[128]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_31: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.00000996502277);  squeeze_14 = None
        mul_32: "f32[128]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[128]" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_23: "f32[128]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
        unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        relu_4: "f32[32, 128, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_1: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.reshape.default(relu_4, [32, 2, 64, 56, 56]);  relu_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:98 in forward, code: x_gap = x.sum(dim=1)
        sum_1: "f32[32, 64, 56, 56]" = torch.ops.aten.sum.dim_IntList(view_1, [1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:101 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean: "f32[32, 64, 1, 1]" = torch.ops.aten.mean.dim(sum_1, [2, 3], True);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:102 in forward, code: x_gap = self.fc1(x_gap)
        convolution_5: "f32[32, 32, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_32, primals_33, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_34, 1)
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[1, 32, 1, 1]" = var_mean_5[0]
        getitem_13: "f32[1, 32, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_26: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_5: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_5: "f32[32, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
        mul_35: "f32[32, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_16: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37: "f32[32]" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_27: "f32[32]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_38: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.032258064516129);  squeeze_17 = None
        mul_39: "f32[32]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[32]" = torch.ops.aten.mul.Tensor(primals_36, 0.9)
        add_28: "f32[32]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
        unsqueeze_21: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[32, 32, 1, 1]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
        unsqueeze_23: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_29: "f32[32, 32, 1, 1]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:104 in forward, code: x_gap = self.act1(x_gap)
        relu_5: "f32[32, 32, 1, 1]" = torch.ops.aten.relu.default(add_29);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:105 in forward, code: x_attn = self.fc2(x_gap)
        convolution_6: "f32[32, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_5, primals_39, primals_40, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_2: "f32[32, 1, 2, 64]" = torch.ops.aten.reshape.default(convolution_6, [32, 1, 2, -1])
        permute: "f32[32, 2, 1, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        amax: "f32[32, 1, 1, 64]" = torch.ops.aten.amax.default(permute, [1], True)
        sub_6: "f32[32, 2, 1, 64]" = torch.ops.aten.sub.Tensor(permute, amax);  permute = amax = None
        exp: "f32[32, 2, 1, 64]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_2: "f32[32, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[32, 2, 1, 64]" = torch.ops.aten.div.Tensor(exp, sum_2);  exp = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_3: "f32[32, 128]" = torch.ops.aten.reshape.default(div, [32, -1]);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_4: "f32[32, 128, 1, 1]" = torch.ops.aten.reshape.default(view_3, [32, -1, 1, 1]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_5: "f32[32, 2, 64, 1, 1]" = torch.ops.aten.reshape.default(view_4, [32, 2, 64, 1, 1]);  view_4 = None
        mul_42: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.mul.Tensor(view_1, view_5);  view_1 = view_5 = None
        sum_3: "f32[32, 64, 56, 56]" = torch.ops.aten.sum.dim_IntList(mul_42, [1]);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:132 in forward, code: out = self.conv3(out)
        convolution_7: "f32[32, 256, 56, 56]" = torch.ops.aten.convolution.default(sum_3, primals_41, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_42, 1)
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[1, 256, 1, 1]" = var_mean_6[0]
        getitem_15: "f32[1, 256, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_31: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_6: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_7: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_43: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = None
        squeeze_18: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_19: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_44: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_45: "f32[256]" = torch.ops.aten.mul.Tensor(primals_43, 0.9)
        add_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
        squeeze_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_46: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.00000996502277);  squeeze_20 = None
        mul_47: "f32[256]" = torch.ops.aten.mul.Tensor(mul_46, 0.1);  mul_46 = None
        mul_48: "f32[256]" = torch.ops.aten.mul.Tensor(primals_44, 0.9)
        add_33: "f32[256]" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
        unsqueeze_24: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
        unsqueeze_25: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_49: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_25);  mul_43 = unsqueeze_25 = None
        unsqueeze_26: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
        unsqueeze_27: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_34: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_27);  mul_49 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        convolution_8: "f32[32, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_6, primals_47, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_48, 1)
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[1, 256, 1, 1]" = var_mean_7[0]
        getitem_17: "f32[1, 256, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_36: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_7: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_8: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_50: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = None
        squeeze_21: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_22: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(primals_49, 0.9)
        add_37: "f32[256]" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
        squeeze_23: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.00000996502277);  squeeze_23 = None
        mul_54: "f32[256]" = torch.ops.aten.mul.Tensor(mul_53, 0.1);  mul_53 = None
        mul_55: "f32[256]" = torch.ops.aten.mul.Tensor(primals_50, 0.9)
        add_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
        unsqueeze_28: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
        unsqueeze_29: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_56: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_29);  mul_50 = unsqueeze_29 = None
        unsqueeze_30: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
        unsqueeze_31: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_39: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_31);  mul_56 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:141 in forward, code: out += shortcut
        add_40: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_34, add_39);  add_34 = add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:142 in forward, code: out = self.act3(out)
        relu_6: "f32[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_40);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:117 in forward, code: out = self.conv1(x)
        convolution_9: "f32[32, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_54, 1)
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[1, 128, 1, 1]" = var_mean_8[0]
        getitem_19: "f32[1, 128, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_42: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_8: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_9: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
        mul_57: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = None
        squeeze_24: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_25: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_58: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_59: "f32[128]" = torch.ops.aten.mul.Tensor(primals_55, 0.9)
        add_43: "f32[128]" = torch.ops.aten.add.Tensor(mul_58, mul_59);  mul_58 = mul_59 = None
        squeeze_26: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_60: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.00000996502277);  squeeze_26 = None
        mul_61: "f32[128]" = torch.ops.aten.mul.Tensor(mul_60, 0.1);  mul_60 = None
        mul_62: "f32[128]" = torch.ops.aten.mul.Tensor(primals_56, 0.9)
        add_44: "f32[128]" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None
        unsqueeze_32: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
        unsqueeze_33: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_63: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_33);  mul_57 = unsqueeze_33 = None
        unsqueeze_34: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
        unsqueeze_35: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_45: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_63, unsqueeze_35);  mul_63 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:119 in forward, code: out = self.act1(out)
        relu_7: "f32[32, 128, 56, 56]" = torch.ops.aten.relu.default(add_45);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:90 in forward, code: x = self.conv(x)
        convolution_10: "f32[32, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_59, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_60, 1)
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[1, 256, 1, 1]" = var_mean_9[0]
        getitem_21: "f32[1, 256, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_47: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_9: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_10: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_64: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = None
        squeeze_27: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3])
        mul_65: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1);  squeeze_27 = None
        mul_66: "f32[256]" = torch.ops.aten.mul.Tensor(primals_61, 0.9)
        add_48: "f32[256]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
        squeeze_29: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_67: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.00000996502277);  squeeze_29 = None
        mul_68: "f32[256]" = torch.ops.aten.mul.Tensor(mul_67, 0.1);  mul_67 = None
        mul_69: "f32[256]" = torch.ops.aten.mul.Tensor(primals_62, 0.9)
        add_49: "f32[256]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
        unsqueeze_36: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
        unsqueeze_37: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_70: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_37);  mul_64 = unsqueeze_37 = None
        unsqueeze_38: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1)
        unsqueeze_39: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_50: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_39);  mul_70 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        relu_8: "f32[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_50);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_7: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.reshape.default(relu_8, [32, 2, 128, 56, 56]);  relu_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:98 in forward, code: x_gap = x.sum(dim=1)
        sum_4: "f32[32, 128, 56, 56]" = torch.ops.aten.sum.dim_IntList(view_7, [1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:101 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_1: "f32[32, 128, 1, 1]" = torch.ops.aten.mean.dim(sum_4, [2, 3], True);  sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:102 in forward, code: x_gap = self.fc1(x_gap)
        convolution_11: "f32[32, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_65, primals_66, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_67, 1)
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[1, 64, 1, 1]" = var_mean_10[0]
        getitem_23: "f32[1, 64, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_52: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_10: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_11: "f32[32, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
        mul_71: "f32[32, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = None
        squeeze_30: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_31: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_72: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_73: "f32[64]" = torch.ops.aten.mul.Tensor(primals_68, 0.9)
        add_53: "f32[64]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
        squeeze_32: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_74: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.032258064516129);  squeeze_32 = None
        mul_75: "f32[64]" = torch.ops.aten.mul.Tensor(mul_74, 0.1);  mul_74 = None
        mul_76: "f32[64]" = torch.ops.aten.mul.Tensor(primals_69, 0.9)
        add_54: "f32[64]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
        unsqueeze_40: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1)
        unsqueeze_41: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_77: "f32[32, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_41);  mul_71 = unsqueeze_41 = None
        unsqueeze_42: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1);  primals_71 = None
        unsqueeze_43: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_55: "f32[32, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_43);  mul_77 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:104 in forward, code: x_gap = self.act1(x_gap)
        relu_9: "f32[32, 64, 1, 1]" = torch.ops.aten.relu.default(add_55);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:105 in forward, code: x_attn = self.fc2(x_gap)
        convolution_12: "f32[32, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_9, primals_72, primals_73, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_8: "f32[32, 1, 2, 128]" = torch.ops.aten.reshape.default(convolution_12, [32, 1, 2, -1])
        permute_1: "f32[32, 2, 1, 128]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        amax_1: "f32[32, 1, 1, 128]" = torch.ops.aten.amax.default(permute_1, [1], True)
        sub_12: "f32[32, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_1, amax_1);  permute_1 = amax_1 = None
        exp_1: "f32[32, 2, 1, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_5: "f32[32, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True)
        div_1: "f32[32, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_5);  exp_1 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_9: "f32[32, 256]" = torch.ops.aten.reshape.default(div_1, [32, -1]);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_10: "f32[32, 256, 1, 1]" = torch.ops.aten.reshape.default(view_9, [32, -1, 1, 1]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_11: "f32[32, 2, 128, 1, 1]" = torch.ops.aten.reshape.default(view_10, [32, 2, 128, 1, 1]);  view_10 = None
        mul_78: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.mul.Tensor(view_7, view_11);  view_7 = view_11 = None
        sum_6: "f32[32, 128, 56, 56]" = torch.ops.aten.sum.dim_IntList(mul_78, [1]);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:130 in forward, code: out = self.avd_last(out)
        avg_pool2d: "f32[32, 128, 28, 28]" = torch.ops.aten.avg_pool2d.default(sum_6, [3, 3], [2, 2], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:132 in forward, code: out = self.conv3(out)
        convolution_13: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(avg_pool2d, primals_74, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        add_56: "i64[]" = torch.ops.aten.add.Tensor(primals_75, 1)
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[1, 512, 1, 1]" = var_mean_11[0]
        getitem_25: "f32[1, 512, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_57: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_11: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_13: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_25)
        mul_79: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = None
        squeeze_33: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_34: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_80: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_81: "f32[512]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_58: "f32[512]" = torch.ops.aten.add.Tensor(mul_80, mul_81);  mul_80 = mul_81 = None
        squeeze_35: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_82: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
        mul_83: "f32[512]" = torch.ops.aten.mul.Tensor(mul_82, 0.1);  mul_82 = None
        mul_84: "f32[512]" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_59: "f32[512]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
        unsqueeze_44: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_45: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_85: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_45);  mul_79 = unsqueeze_45 = None
        unsqueeze_46: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_47: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_60: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_47);  mul_85 = unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        avg_pool2d_1: "f32[32, 256, 28, 28]" = torch.ops.aten.avg_pool2d.default(relu_6, [2, 2], [2, 2], [0, 0], True, False)
        convolution_14: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(avg_pool2d_1, primals_80, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_61: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[1, 512, 1, 1]" = var_mean_12[0]
        getitem_27: "f32[1, 512, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_62: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_12: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_14: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_27)
        mul_86: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = None
        squeeze_36: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_37: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_87: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_88: "f32[512]" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_63: "f32[512]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
        squeeze_38: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_89: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
        mul_90: "f32[512]" = torch.ops.aten.mul.Tensor(mul_89, 0.1);  mul_89 = None
        mul_91: "f32[512]" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_64: "f32[512]" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
        unsqueeze_48: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_49: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_92: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_86, unsqueeze_49);  mul_86 = unsqueeze_49 = None
        unsqueeze_50: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_51: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_65: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_51);  mul_92 = unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:141 in forward, code: out += shortcut
        add_66: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_60, add_65);  add_60 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:142 in forward, code: out = self.act3(out)
        relu_10: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_66);  add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:117 in forward, code: out = self.conv1(x)
        convolution_15: "f32[32, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_86, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_87, 1)
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[1, 256, 1, 1]" = var_mean_13[0]
        getitem_29: "f32[1, 256, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_68: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_13: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_15: "f32[32, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_29)
        mul_93: "f32[32, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = None
        squeeze_39: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_40: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_94: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_95: "f32[256]" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_69: "f32[256]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
        squeeze_41: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_96: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000398612827361);  squeeze_41 = None
        mul_97: "f32[256]" = torch.ops.aten.mul.Tensor(mul_96, 0.1);  mul_96 = None
        mul_98: "f32[256]" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_70: "f32[256]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
        unsqueeze_52: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_53: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_99: "f32[32, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_53);  mul_93 = unsqueeze_53 = None
        unsqueeze_54: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_55: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_71: "f32[32, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_55);  mul_99 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:119 in forward, code: out = self.act1(out)
        relu_11: "f32[32, 256, 28, 28]" = torch.ops.aten.relu.default(add_71);  add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:90 in forward, code: x = self.conv(x)
        convolution_16: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_92, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_93, 1)
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[1, 512, 1, 1]" = var_mean_14[0]
        getitem_31: "f32[1, 512, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_73: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_14: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_16: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_31)
        mul_100: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = None
        squeeze_42: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3])
        mul_101: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1);  squeeze_42 = None
        mul_102: "f32[512]" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_74: "f32[512]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
        squeeze_44: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_103: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000398612827361);  squeeze_44 = None
        mul_104: "f32[512]" = torch.ops.aten.mul.Tensor(mul_103, 0.1);  mul_103 = None
        mul_105: "f32[512]" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_75: "f32[512]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
        unsqueeze_56: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_57: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_106: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_57);  mul_100 = unsqueeze_57 = None
        unsqueeze_58: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
        unsqueeze_59: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_76: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_59);  mul_106 = unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        relu_12: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_76);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_13: "f32[32, 2, 256, 28, 28]" = torch.ops.aten.reshape.default(relu_12, [32, 2, 256, 28, 28]);  relu_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:98 in forward, code: x_gap = x.sum(dim=1)
        sum_7: "f32[32, 256, 28, 28]" = torch.ops.aten.sum.dim_IntList(view_13, [1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:101 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_2: "f32[32, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_7, [2, 3], True);  sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:102 in forward, code: x_gap = self.fc1(x_gap)
        convolution_17: "f32[32, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_98, primals_99, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_100, 1)
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[1, 128, 1, 1]" = var_mean_15[0]
        getitem_33: "f32[1, 128, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_78: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_15: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_17: "f32[32, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_33)
        mul_107: "f32[32, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_15);  sub_17 = None
        squeeze_45: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_46: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_108: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_109: "f32[128]" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_79: "f32[128]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
        squeeze_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_110: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.032258064516129);  squeeze_47 = None
        mul_111: "f32[128]" = torch.ops.aten.mul.Tensor(mul_110, 0.1);  mul_110 = None
        mul_112: "f32[128]" = torch.ops.aten.mul.Tensor(primals_102, 0.9)
        add_80: "f32[128]" = torch.ops.aten.add.Tensor(mul_111, mul_112);  mul_111 = mul_112 = None
        unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
        unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_113: "f32[32, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_61);  mul_107 = unsqueeze_61 = None
        unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
        unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_81: "f32[32, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_63);  mul_113 = unsqueeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:104 in forward, code: x_gap = self.act1(x_gap)
        relu_13: "f32[32, 128, 1, 1]" = torch.ops.aten.relu.default(add_81);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:105 in forward, code: x_attn = self.fc2(x_gap)
        convolution_18: "f32[32, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_13, primals_105, primals_106, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_14: "f32[32, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_18, [32, 1, 2, -1])
        permute_2: "f32[32, 2, 1, 256]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        amax_2: "f32[32, 1, 1, 256]" = torch.ops.aten.amax.default(permute_2, [1], True)
        sub_18: "f32[32, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_2, amax_2);  permute_2 = amax_2 = None
        exp_2: "f32[32, 2, 1, 256]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_8: "f32[32, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True)
        div_2: "f32[32, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_2, sum_8);  exp_2 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_15: "f32[32, 512]" = torch.ops.aten.reshape.default(div_2, [32, -1]);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_16: "f32[32, 512, 1, 1]" = torch.ops.aten.reshape.default(view_15, [32, -1, 1, 1]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_17: "f32[32, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_16, [32, 2, 256, 1, 1]);  view_16 = None
        mul_114: "f32[32, 2, 256, 28, 28]" = torch.ops.aten.mul.Tensor(view_13, view_17);  view_13 = view_17 = None
        sum_9: "f32[32, 256, 28, 28]" = torch.ops.aten.sum.dim_IntList(mul_114, [1]);  mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:130 in forward, code: out = self.avd_last(out)
        avg_pool2d_2: "f32[32, 256, 14, 14]" = torch.ops.aten.avg_pool2d.default(sum_9, [3, 3], [2, 2], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:132 in forward, code: out = self.conv3(out)
        convolution_19: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(avg_pool2d_2, primals_107, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        add_82: "i64[]" = torch.ops.aten.add.Tensor(primals_108, 1)
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[1, 1024, 1, 1]" = var_mean_16[0]
        getitem_35: "f32[1, 1024, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_83: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_16: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_19: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_35)
        mul_115: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_16);  sub_19 = None
        squeeze_48: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_49: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_116: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_117: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_109, 0.9)
        add_84: "f32[1024]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        squeeze_50: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_118: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
        mul_119: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_118, 0.1);  mul_118 = None
        mul_120: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_110, 0.9)
        add_85: "f32[1024]" = torch.ops.aten.add.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
        unsqueeze_64: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
        unsqueeze_65: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_121: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_65);  mul_115 = unsqueeze_65 = None
        unsqueeze_66: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
        unsqueeze_67: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_86: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_67);  mul_121 = unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        avg_pool2d_3: "f32[32, 512, 14, 14]" = torch.ops.aten.avg_pool2d.default(relu_10, [2, 2], [2, 2], [0, 0], True, False)
        convolution_20: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(avg_pool2d_3, primals_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_87: "i64[]" = torch.ops.aten.add.Tensor(primals_114, 1)
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[1, 1024, 1, 1]" = var_mean_17[0]
        getitem_37: "f32[1, 1024, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_88: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_17: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_20: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_37)
        mul_122: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_17);  sub_20 = None
        squeeze_51: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_52: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_123: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_124: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_115, 0.9)
        add_89: "f32[1024]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        squeeze_53: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_125: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
        mul_126: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_125, 0.1);  mul_125 = None
        mul_127: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_116, 0.9)
        add_90: "f32[1024]" = torch.ops.aten.add.Tensor(mul_126, mul_127);  mul_126 = mul_127 = None
        unsqueeze_68: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1)
        unsqueeze_69: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_128: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_122, unsqueeze_69);  mul_122 = unsqueeze_69 = None
        unsqueeze_70: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1);  primals_118 = None
        unsqueeze_71: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_91: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_71);  mul_128 = unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:141 in forward, code: out += shortcut
        add_92: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_86, add_91);  add_86 = add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:142 in forward, code: out = self.act3(out)
        relu_14: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_92);  add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:117 in forward, code: out = self.conv1(x)
        convolution_21: "f32[32, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_14, primals_119, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_120, 1)
        var_mean_18 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem_38: "f32[1, 512, 1, 1]" = var_mean_18[0]
        getitem_39: "f32[1, 512, 1, 1]" = var_mean_18[1];  var_mean_18 = None
        add_94: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_18: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_21: "f32[32, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_39)
        mul_129: "f32[32, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_18);  sub_21 = None
        squeeze_54: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_55: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_130: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_131: "f32[512]" = torch.ops.aten.mul.Tensor(primals_121, 0.9)
        add_95: "f32[512]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        squeeze_56: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_132: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
        mul_133: "f32[512]" = torch.ops.aten.mul.Tensor(mul_132, 0.1);  mul_132 = None
        mul_134: "f32[512]" = torch.ops.aten.mul.Tensor(primals_122, 0.9)
        add_96: "f32[512]" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None
        unsqueeze_72: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1)
        unsqueeze_73: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_135: "f32[32, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_73);  mul_129 = unsqueeze_73 = None
        unsqueeze_74: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
        unsqueeze_75: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_97: "f32[32, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_135, unsqueeze_75);  mul_135 = unsqueeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:119 in forward, code: out = self.act1(out)
        relu_15: "f32[32, 512, 14, 14]" = torch.ops.aten.relu.default(add_97);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:90 in forward, code: x = self.conv(x)
        convolution_22: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_15, primals_125, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_126, 1)
        var_mean_19 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem_40: "f32[1, 1024, 1, 1]" = var_mean_19[0]
        getitem_41: "f32[1, 1024, 1, 1]" = var_mean_19[1];  var_mean_19 = None
        add_99: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_19: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_22: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_41)
        mul_136: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_19);  sub_22 = None
        squeeze_57: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3])
        mul_137: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1);  squeeze_57 = None
        mul_138: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_127, 0.9)
        add_100: "f32[1024]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        squeeze_59: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_139: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
        mul_140: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_139, 0.1);  mul_139 = None
        mul_141: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_128, 0.9)
        add_101: "f32[1024]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
        unsqueeze_76: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1)
        unsqueeze_77: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_142: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_77);  mul_136 = unsqueeze_77 = None
        unsqueeze_78: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1)
        unsqueeze_79: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_102: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_79);  mul_142 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        relu_16: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_102);  add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        view_19: "f32[32, 2, 512, 14, 14]" = torch.ops.aten.reshape.default(relu_16, [32, 2, 512, 14, 14]);  relu_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:98 in forward, code: x_gap = x.sum(dim=1)
        sum_10: "f32[32, 512, 14, 14]" = torch.ops.aten.sum.dim_IntList(view_19, [1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:101 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_3: "f32[32, 512, 1, 1]" = torch.ops.aten.mean.dim(sum_10, [2, 3], True);  sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:102 in forward, code: x_gap = self.fc1(x_gap)
        convolution_23: "f32[32, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_131, primals_132, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        add_103: "i64[]" = torch.ops.aten.add.Tensor(primals_133, 1)
        var_mean_20 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42: "f32[1, 256, 1, 1]" = var_mean_20[0]
        getitem_43: "f32[1, 256, 1, 1]" = var_mean_20[1];  var_mean_20 = None
        add_104: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_20: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_23: "f32[32, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_43)
        mul_143: "f32[32, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_20);  sub_23 = None
        squeeze_60: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_61: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_144: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_145: "f32[256]" = torch.ops.aten.mul.Tensor(primals_134, 0.9)
        add_105: "f32[256]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        squeeze_62: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_146: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.032258064516129);  squeeze_62 = None
        mul_147: "f32[256]" = torch.ops.aten.mul.Tensor(mul_146, 0.1);  mul_146 = None
        mul_148: "f32[256]" = torch.ops.aten.mul.Tensor(primals_135, 0.9)
        add_106: "f32[256]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
        unsqueeze_80: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_136, -1)
        unsqueeze_81: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_149: "f32[32, 256, 1, 1]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_81);  mul_143 = unsqueeze_81 = None
        unsqueeze_82: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1);  primals_137 = None
        unsqueeze_83: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_107: "f32[32, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_83);  mul_149 = unsqueeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:104 in forward, code: x_gap = self.act1(x_gap)
        relu_17: "f32[32, 256, 1, 1]" = torch.ops.aten.relu.default(add_107);  add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:105 in forward, code: x_attn = self.fc2(x_gap)
        convolution_24: "f32[32, 1024, 1, 1]" = torch.ops.aten.convolution.default(relu_17, primals_138, primals_139, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        view_20: "f32[32, 1, 2, 512]" = torch.ops.aten.reshape.default(convolution_24, [32, 1, 2, -1])
        permute_3: "f32[32, 2, 1, 512]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        amax_3: "f32[32, 1, 1, 512]" = torch.ops.aten.amax.default(permute_3, [1], True)
        sub_24: "f32[32, 2, 1, 512]" = torch.ops.aten.sub.Tensor(permute_3, amax_3);  permute_3 = amax_3 = None
        exp_3: "f32[32, 2, 1, 512]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_11: "f32[32, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(exp_3, [1], True)
        div_3: "f32[32, 2, 1, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_11);  exp_3 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        view_21: "f32[32, 1024]" = torch.ops.aten.reshape.default(div_3, [32, -1]);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        view_22: "f32[32, 1024, 1, 1]" = torch.ops.aten.reshape.default(view_21, [32, -1, 1, 1]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        view_23: "f32[32, 2, 512, 1, 1]" = torch.ops.aten.reshape.default(view_22, [32, 2, 512, 1, 1]);  view_22 = None
        mul_150: "f32[32, 2, 512, 14, 14]" = torch.ops.aten.mul.Tensor(view_19, view_23);  view_19 = view_23 = None
        sum_12: "f32[32, 512, 14, 14]" = torch.ops.aten.sum.dim_IntList(mul_150, [1]);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:130 in forward, code: out = self.avd_last(out)
        avg_pool2d_4: "f32[32, 512, 7, 7]" = torch.ops.aten.avg_pool2d.default(sum_12, [3, 3], [2, 2], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:132 in forward, code: out = self.conv3(out)
        convolution_25: "f32[32, 2048, 7, 7]" = torch.ops.aten.convolution.default(avg_pool2d_4, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        add_108: "i64[]" = torch.ops.aten.add.Tensor(primals_141, 1)
        var_mean_21 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
        getitem_44: "f32[1, 2048, 1, 1]" = var_mean_21[0]
        getitem_45: "f32[1, 2048, 1, 1]" = var_mean_21[1];  var_mean_21 = None
        add_109: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_21: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_25: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_45)
        mul_151: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_21);  sub_25 = None
        squeeze_63: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3])
        mul_152: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1);  squeeze_63 = None
        mul_153: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_110: "f32[2048]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
        squeeze_65: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_154: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0006381620931717);  squeeze_65 = None
        mul_155: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_154, 0.1);  mul_154 = None
        mul_156: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
        add_111: "f32[2048]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        unsqueeze_84: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_85: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_157: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_85);  mul_151 = unsqueeze_85 = None
        unsqueeze_86: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_145, -1)
        unsqueeze_87: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_112: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_87);  mul_157 = unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        avg_pool2d_5: "f32[32, 1024, 7, 7]" = torch.ops.aten.avg_pool2d.default(relu_14, [2, 2], [2, 2], [0, 0], True, False)
        convolution_26: "f32[32, 2048, 7, 7]" = torch.ops.aten.convolution.default(avg_pool2d_5, primals_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_113: "i64[]" = torch.ops.aten.add.Tensor(primals_147, 1)
        var_mean_22 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem_46: "f32[1, 2048, 1, 1]" = var_mean_22[0]
        getitem_47: "f32[1, 2048, 1, 1]" = var_mean_22[1];  var_mean_22 = None
        add_114: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_22: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_26: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_47)
        mul_158: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_22);  sub_26 = None
        squeeze_66: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3])
        mul_159: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1);  squeeze_66 = None
        mul_160: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
        add_115: "f32[2048]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
        squeeze_68: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_161: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0006381620931717);  squeeze_68 = None
        mul_162: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_161, 0.1);  mul_161 = None
        mul_163: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
        add_116: "f32[2048]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        unsqueeze_88: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_89: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_164: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_158, unsqueeze_89);  mul_158 = unsqueeze_89 = None
        unsqueeze_90: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_151, -1)
        unsqueeze_91: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_117: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_91);  mul_164 = unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:141 in forward, code: out += shortcut
        add_118: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_112, add_117);  add_112 = add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:142 in forward, code: out = self.act3(out)
        relu_18: "f32[32, 2048, 7, 7]" = torch.ops.aten.relu.default(add_118);  add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_4: "f32[32, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_18, [-1, -2], True);  relu_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_24: "f32[32, 2048]" = torch.ops.aten.reshape.default(mean_4, [32, 2048]);  mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:772 in forward_head, code: return x if pre_logits else self.fc(x)
        permute_4: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_152, [1, 0])
        addmm: "f32[32, 1000]" = torch.ops.aten.addmm.default(primals_153, view_24, permute_4);  primals_153 = permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        unsqueeze_117: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_118: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
        unsqueeze_119: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        unsqueeze_142: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_143: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
        unsqueeze_144: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        unsqueeze_154: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_155: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
        unsqueeze_156: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        unsqueeze_166: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_167: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
        unsqueeze_168: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        unsqueeze_179: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_180: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
        unsqueeze_181: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        unsqueeze_204: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_205: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None
        unsqueeze_206: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        unsqueeze_216: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_217: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 2);  unsqueeze_216 = None
        unsqueeze_218: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 3);  unsqueeze_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        unsqueeze_228: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_229: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
        unsqueeze_230: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        unsqueeze_241: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_242: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        unsqueeze_266: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_267: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
        unsqueeze_268: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:139 in forward, code: shortcut = self.downsample(x)
        unsqueeze_278: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_279: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
        unsqueeze_280: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:133 in forward, code: out = self.bn3(out)
        unsqueeze_290: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_291: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
        unsqueeze_292: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:103 in forward, code: x_gap = self.bn1(x_gap)
        unsqueeze_303: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_304: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
        unsqueeze_305: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:118 in forward, code: out = self.bn1(out)
        unsqueeze_328: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_329: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
        unsqueeze_330: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:745 in forward_features, code: x = self.conv1(x)
        unsqueeze_352: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_353: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
        unsqueeze_354: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
        unsqueeze_364: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_365: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
        unsqueeze_366: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None

        # No stacktrace found for following nodes
        copy_: "i64[]" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_5);  primals_9 = add_5 = copy__3 = None
        copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_10, add_7);  primals_10 = add_7 = copy__4 = None
        copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_11, add_8);  primals_11 = add_8 = copy__5 = None
        copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_15, add_10);  primals_15 = add_10 = copy__6 = None
        copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_16, add_12);  primals_16 = add_12 = copy__7 = None
        copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_17, add_13);  primals_17 = add_13 = copy__8 = None
        copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_21, add_15);  primals_21 = add_15 = copy__9 = None
        copy__10: "f32[64]" = torch.ops.aten.copy_.default(primals_22, add_17);  primals_22 = add_17 = copy__10 = None
        copy__11: "f32[64]" = torch.ops.aten.copy_.default(primals_23, add_18);  primals_23 = add_18 = copy__11 = None
        copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_27, add_20);  primals_27 = add_20 = copy__12 = None
        copy__13: "f32[128]" = torch.ops.aten.copy_.default(primals_28, add_22);  primals_28 = add_22 = copy__13 = None
        copy__14: "f32[128]" = torch.ops.aten.copy_.default(primals_29, add_23);  primals_29 = add_23 = copy__14 = None
        copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_34, add_25);  primals_34 = add_25 = copy__15 = None
        copy__16: "f32[32]" = torch.ops.aten.copy_.default(primals_35, add_27);  primals_35 = add_27 = copy__16 = None
        copy__17: "f32[32]" = torch.ops.aten.copy_.default(primals_36, add_28);  primals_36 = add_28 = copy__17 = None
        copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_42, add_30);  primals_42 = add_30 = copy__18 = None
        copy__19: "f32[256]" = torch.ops.aten.copy_.default(primals_43, add_32);  primals_43 = add_32 = copy__19 = None
        copy__20: "f32[256]" = torch.ops.aten.copy_.default(primals_44, add_33);  primals_44 = add_33 = copy__20 = None
        copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_48, add_35);  primals_48 = add_35 = copy__21 = None
        copy__22: "f32[256]" = torch.ops.aten.copy_.default(primals_49, add_37);  primals_49 = add_37 = copy__22 = None
        copy__23: "f32[256]" = torch.ops.aten.copy_.default(primals_50, add_38);  primals_50 = add_38 = copy__23 = None
        copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_54, add_41);  primals_54 = add_41 = copy__24 = None
        copy__25: "f32[128]" = torch.ops.aten.copy_.default(primals_55, add_43);  primals_55 = add_43 = copy__25 = None
        copy__26: "f32[128]" = torch.ops.aten.copy_.default(primals_56, add_44);  primals_56 = add_44 = copy__26 = None
        copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_60, add_46);  primals_60 = add_46 = copy__27 = None
        copy__28: "f32[256]" = torch.ops.aten.copy_.default(primals_61, add_48);  primals_61 = add_48 = copy__28 = None
        copy__29: "f32[256]" = torch.ops.aten.copy_.default(primals_62, add_49);  primals_62 = add_49 = copy__29 = None
        copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_67, add_51);  primals_67 = add_51 = copy__30 = None
        copy__31: "f32[64]" = torch.ops.aten.copy_.default(primals_68, add_53);  primals_68 = add_53 = copy__31 = None
        copy__32: "f32[64]" = torch.ops.aten.copy_.default(primals_69, add_54);  primals_69 = add_54 = copy__32 = None
        copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_75, add_56);  primals_75 = add_56 = copy__33 = None
        copy__34: "f32[512]" = torch.ops.aten.copy_.default(primals_76, add_58);  primals_76 = add_58 = copy__34 = None
        copy__35: "f32[512]" = torch.ops.aten.copy_.default(primals_77, add_59);  primals_77 = add_59 = copy__35 = None
        copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_81, add_61);  primals_81 = add_61 = copy__36 = None
        copy__37: "f32[512]" = torch.ops.aten.copy_.default(primals_82, add_63);  primals_82 = add_63 = copy__37 = None
        copy__38: "f32[512]" = torch.ops.aten.copy_.default(primals_83, add_64);  primals_83 = add_64 = copy__38 = None
        copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_87, add_67);  primals_87 = add_67 = copy__39 = None
        copy__40: "f32[256]" = torch.ops.aten.copy_.default(primals_88, add_69);  primals_88 = add_69 = copy__40 = None
        copy__41: "f32[256]" = torch.ops.aten.copy_.default(primals_89, add_70);  primals_89 = add_70 = copy__41 = None
        copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_93, add_72);  primals_93 = add_72 = copy__42 = None
        copy__43: "f32[512]" = torch.ops.aten.copy_.default(primals_94, add_74);  primals_94 = add_74 = copy__43 = None
        copy__44: "f32[512]" = torch.ops.aten.copy_.default(primals_95, add_75);  primals_95 = add_75 = copy__44 = None
        copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_100, add_77);  primals_100 = add_77 = copy__45 = None
        copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_101, add_79);  primals_101 = add_79 = copy__46 = None
        copy__47: "f32[128]" = torch.ops.aten.copy_.default(primals_102, add_80);  primals_102 = add_80 = copy__47 = None
        copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_108, add_82);  primals_108 = add_82 = copy__48 = None
        copy__49: "f32[1024]" = torch.ops.aten.copy_.default(primals_109, add_84);  primals_109 = add_84 = copy__49 = None
        copy__50: "f32[1024]" = torch.ops.aten.copy_.default(primals_110, add_85);  primals_110 = add_85 = copy__50 = None
        copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_114, add_87);  primals_114 = add_87 = copy__51 = None
        copy__52: "f32[1024]" = torch.ops.aten.copy_.default(primals_115, add_89);  primals_115 = add_89 = copy__52 = None
        copy__53: "f32[1024]" = torch.ops.aten.copy_.default(primals_116, add_90);  primals_116 = add_90 = copy__53 = None
        copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_120, add_93);  primals_120 = add_93 = copy__54 = None
        copy__55: "f32[512]" = torch.ops.aten.copy_.default(primals_121, add_95);  primals_121 = add_95 = copy__55 = None
        copy__56: "f32[512]" = torch.ops.aten.copy_.default(primals_122, add_96);  primals_122 = add_96 = copy__56 = None
        copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_126, add_98);  primals_126 = add_98 = copy__57 = None
        copy__58: "f32[1024]" = torch.ops.aten.copy_.default(primals_127, add_100);  primals_127 = add_100 = copy__58 = None
        copy__59: "f32[1024]" = torch.ops.aten.copy_.default(primals_128, add_101);  primals_128 = add_101 = copy__59 = None
        copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_133, add_103);  primals_133 = add_103 = copy__60 = None
        copy__61: "f32[256]" = torch.ops.aten.copy_.default(primals_134, add_105);  primals_134 = add_105 = copy__61 = None
        copy__62: "f32[256]" = torch.ops.aten.copy_.default(primals_135, add_106);  primals_135 = add_106 = copy__62 = None
        copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_141, add_108);  primals_141 = add_108 = copy__63 = None
        copy__64: "f32[2048]" = torch.ops.aten.copy_.default(primals_142, add_110);  primals_142 = add_110 = copy__64 = None
        copy__65: "f32[2048]" = torch.ops.aten.copy_.default(primals_143, add_111);  primals_143 = add_111 = copy__65 = None
        copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_147, add_113);  primals_147 = add_113 = copy__66 = None
        copy__67: "f32[2048]" = torch.ops.aten.copy_.default(primals_148, add_115);  primals_148 = add_115 = copy__67 = None
        copy__68: "f32[2048]" = torch.ops.aten.copy_.default(primals_149, add_116);  primals_149 = add_116 = copy__68 = None
        return (addmm, primals_1, primals_2, primals_6, primals_8, primals_12, primals_14, primals_18, primals_19, primals_20, primals_24, primals_26, primals_30, primals_31, primals_32, primals_37, primals_39, primals_41, primals_45, primals_47, primals_51, primals_53, primals_57, primals_59, primals_63, primals_64, primals_65, primals_70, primals_72, primals_74, primals_78, primals_80, primals_84, primals_86, primals_90, primals_92, primals_96, primals_97, primals_98, primals_103, primals_105, primals_107, primals_111, primals_113, primals_117, primals_119, primals_123, primals_125, primals_129, primals_130, primals_131, primals_136, primals_138, primals_140, primals_144, primals_145, primals_146, primals_150, primals_151, primals_152, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, getitem_5, rsqrt_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, getitem_11, rsqrt_4, mean, convolution_5, squeeze_16, relu_5, convolution_6, sum_3, convolution_7, squeeze_19, convolution_8, squeeze_22, relu_6, convolution_9, squeeze_25, relu_7, convolution_10, getitem_21, rsqrt_9, mean_1, convolution_11, squeeze_31, relu_9, convolution_12, sum_6, avg_pool2d, convolution_13, squeeze_34, avg_pool2d_1, convolution_14, squeeze_37, relu_10, convolution_15, squeeze_40, relu_11, convolution_16, getitem_31, rsqrt_14, mean_2, convolution_17, squeeze_46, relu_13, convolution_18, sum_9, avg_pool2d_2, convolution_19, squeeze_49, avg_pool2d_3, convolution_20, squeeze_52, relu_14, convolution_21, squeeze_55, relu_15, convolution_22, getitem_41, rsqrt_19, mean_3, convolution_23, squeeze_61, relu_17, convolution_24, sum_12, avg_pool2d_4, convolution_25, getitem_45, rsqrt_21, avg_pool2d_5, convolution_26, getitem_47, rsqrt_22, view_24, unsqueeze_119, unsqueeze_144, unsqueeze_156, unsqueeze_168, unsqueeze_181, unsqueeze_206, unsqueeze_218, unsqueeze_230, unsqueeze_243, unsqueeze_268, unsqueeze_280, unsqueeze_292, unsqueeze_305, unsqueeze_330, unsqueeze_354, unsqueeze_366)
