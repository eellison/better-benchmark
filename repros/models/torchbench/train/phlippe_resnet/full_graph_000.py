import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3]", primals_2: "f32[128, 3, 32, 32]", primals_3: "i64[]", primals_4: "f32[16]", primals_5: "f32[16]", primals_6: "f32[16]", primals_7: "f32[16]", primals_8: "f32[16, 16, 3, 3]", primals_9: "i64[]", primals_10: "f32[16]", primals_11: "f32[16]", primals_12: "f32[16]", primals_13: "f32[16]", primals_14: "f32[16, 16, 3, 3]", primals_15: "i64[]", primals_16: "f32[16]", primals_17: "f32[16]", primals_18: "f32[16]", primals_19: "f32[16]", primals_20: "f32[16, 16, 3, 3]", primals_21: "i64[]", primals_22: "f32[16]", primals_23: "f32[16]", primals_24: "f32[16]", primals_25: "f32[16]", primals_26: "f32[16, 16, 3, 3]", primals_27: "i64[]", primals_28: "f32[16]", primals_29: "f32[16]", primals_30: "f32[16]", primals_31: "f32[16]", primals_32: "f32[16, 16, 3, 3]", primals_33: "i64[]", primals_34: "f32[16]", primals_35: "f32[16]", primals_36: "f32[16]", primals_37: "f32[16]", primals_38: "f32[16, 16, 3, 3]", primals_39: "i64[]", primals_40: "f32[16]", primals_41: "f32[16]", primals_42: "f32[16]", primals_43: "f32[16]", primals_44: "f32[32, 16, 3, 3]", primals_45: "i64[]", primals_46: "f32[32]", primals_47: "f32[32]", primals_48: "f32[32]", primals_49: "f32[32]", primals_50: "f32[32, 32, 3, 3]", primals_51: "i64[]", primals_52: "f32[32]", primals_53: "f32[32]", primals_54: "f32[32]", primals_55: "f32[32]", primals_56: "f32[32, 16, 1, 1]", primals_57: "f32[32]", primals_58: "f32[32, 32, 3, 3]", primals_59: "i64[]", primals_60: "f32[32]", primals_61: "f32[32]", primals_62: "f32[32]", primals_63: "f32[32]", primals_64: "f32[32, 32, 3, 3]", primals_65: "i64[]", primals_66: "f32[32]", primals_67: "f32[32]", primals_68: "f32[32]", primals_69: "f32[32]", primals_70: "f32[32, 32, 3, 3]", primals_71: "i64[]", primals_72: "f32[32]", primals_73: "f32[32]", primals_74: "f32[32]", primals_75: "f32[32]", primals_76: "f32[32, 32, 3, 3]", primals_77: "i64[]", primals_78: "f32[32]", primals_79: "f32[32]", primals_80: "f32[32]", primals_81: "f32[32]", primals_82: "f32[64, 32, 3, 3]", primals_83: "i64[]", primals_84: "f32[64]", primals_85: "f32[64]", primals_86: "f32[64]", primals_87: "f32[64]", primals_88: "f32[64, 64, 3, 3]", primals_89: "i64[]", primals_90: "f32[64]", primals_91: "f32[64]", primals_92: "f32[64]", primals_93: "f32[64]", primals_94: "f32[64, 32, 1, 1]", primals_95: "f32[64]", primals_96: "f32[64, 64, 3, 3]", primals_97: "i64[]", primals_98: "f32[64]", primals_99: "f32[64]", primals_100: "f32[64]", primals_101: "f32[64]", primals_102: "f32[64, 64, 3, 3]", primals_103: "i64[]", primals_104: "f32[64]", primals_105: "f32[64]", primals_106: "f32[64]", primals_107: "f32[64]", primals_108: "f32[64, 64, 3, 3]", primals_109: "i64[]", primals_110: "f32[64]", primals_111: "f32[64]", primals_112: "f32[64]", primals_113: "f32[64]", primals_114: "f32[64, 64, 3, 3]", primals_115: "i64[]", primals_116: "f32[64]", primals_117: "f32[64]", primals_118: "f32[64]", primals_119: "f32[64]", primals_120: "f32[10, 64]", primals_121: "f32[10]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:137 in forward, code: x = self.input_net(x)
        convolution: "f32[128, 16, 32, 32]" = torch.ops.aten.convolution.default(primals_2, primals_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 16, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 16, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000076294527394);  squeeze_2 = None
        mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        relu: "f32[128, 16, 32, 32]" = torch.ops.aten.relu.default(add_4);  add_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        convolution_1: "f32[128, 16, 32, 32]" = torch.ops.aten.convolution.default(relu, primals_8, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 16, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 16, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[16]" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_7: "f32[16]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000076294527394);  squeeze_5 = None
        mul_11: "f32[16]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[16]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_8: "f32[16]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        relu_1: "f32[128, 16, 32, 32]" = torch.ops.aten.relu.default(add_9);  add_9 = None
        convolution_2: "f32[128, 16, 32, 32]" = torch.ops.aten.convolution.default(relu_1, primals_14, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 16, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 16, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_7: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16: "f32[16]" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_12: "f32[16]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000076294527394);  squeeze_8 = None
        mul_18: "f32[16]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[16]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_13: "f32[16]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:54 in forward, code: out = z + x
        add_15: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(add_14, relu);  add_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        relu_2: "f32[128, 16, 32, 32]" = torch.ops.aten.relu.default(add_15);  add_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        convolution_3: "f32[128, 16, 32, 32]" = torch.ops.aten.convolution.default(relu_2, primals_20, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_16: "i64[]" = torch.ops.aten.add.Tensor(primals_21, 1)
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 16, 1, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 16, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_17: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_3: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
        mul_21: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_10: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[16]" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_18: "f32[16]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_24: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000076294527394);  squeeze_11 = None
        mul_25: "f32[16]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[16]" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_19: "f32[16]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_20: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        relu_3: "f32[128, 16, 32, 32]" = torch.ops.aten.relu.default(add_20);  add_20 = None
        convolution_4: "f32[128, 16, 32, 32]" = torch.ops.aten.convolution.default(relu_3, primals_26, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_21: "i64[]" = torch.ops.aten.add.Tensor(primals_27, 1)
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 16, 1, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 16, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_22: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_4: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
        mul_28: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_13: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30: "f32[16]" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_23: "f32[16]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_31: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000076294527394);  squeeze_14 = None
        mul_32: "f32[16]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[16]" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_24: "f32[16]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_25: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:54 in forward, code: out = z + x
        add_26: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(add_25, relu_2);  add_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        relu_4: "f32[128, 16, 32, 32]" = torch.ops.aten.relu.default(add_26);  add_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        convolution_5: "f32[128, 16, 32, 32]" = torch.ops.aten.convolution.default(relu_4, primals_32, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_27: "i64[]" = torch.ops.aten.add.Tensor(primals_33, 1)
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 16, 1, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 16, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_28: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_5: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_5: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
        mul_35: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_16: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37: "f32[16]" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_29: "f32[16]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_38: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000076294527394);  squeeze_17 = None
        mul_39: "f32[16]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[16]" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_30: "f32[16]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_31: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        relu_5: "f32[128, 16, 32, 32]" = torch.ops.aten.relu.default(add_31);  add_31 = None
        convolution_6: "f32[128, 16, 32, 32]" = torch.ops.aten.convolution.default(relu_5, primals_38, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_32: "i64[]" = torch.ops.aten.add.Tensor(primals_39, 1)
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[1, 16, 1, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 16, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_33: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_6: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_6: "f32[128, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_42: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_19: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[16]" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_34: "f32[16]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_45: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000076294527394);  squeeze_20 = None
        mul_46: "f32[16]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[16]" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_35: "f32[16]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[128, 16, 32, 32]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_36: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:54 in forward, code: out = z + x
        add_37: "f32[128, 16, 32, 32]" = torch.ops.aten.add.Tensor(add_36, relu_4);  add_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        relu_6: "f32[128, 16, 32, 32]" = torch.ops.aten.relu.default(add_37);  add_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        convolution_7: "f32[128, 32, 16, 16]" = torch.ops.aten.convolution.default(relu_6, primals_44, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_38: "i64[]" = torch.ops.aten.add.Tensor(primals_45, 1)
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[1, 32, 1, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 32, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_39: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_7: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_7: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_49: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_22: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_51: "f32[32]" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_40: "f32[32]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_52: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.000030518509476);  squeeze_23 = None
        mul_53: "f32[32]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[32]" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_41: "f32[32]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_42: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        relu_7: "f32[128, 32, 16, 16]" = torch.ops.aten.relu.default(add_42);  add_42 = None
        convolution_8: "f32[128, 32, 16, 16]" = torch.ops.aten.convolution.default(relu_7, primals_50, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_43: "i64[]" = torch.ops.aten.add.Tensor(primals_51, 1)
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[1, 32, 1, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 32, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_44: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_8: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_8: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_56: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_25: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58: "f32[32]" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_45: "f32[32]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_59: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.000030518509476);  squeeze_26 = None
        mul_60: "f32[32]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[32]" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_46: "f32[32]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_47: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:53 in forward, code: x = self.downsample(x)
        convolution_9: "f32[128, 32, 16, 16]" = torch.ops.aten.convolution.default(relu_6, primals_56, primals_57, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:54 in forward, code: out = z + x
        add_48: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(add_47, convolution_9);  add_47 = convolution_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        relu_8: "f32[128, 32, 16, 16]" = torch.ops.aten.relu.default(add_48);  add_48 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        convolution_10: "f32[128, 32, 16, 16]" = torch.ops.aten.convolution.default(relu_8, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_49: "i64[]" = torch.ops.aten.add.Tensor(primals_59, 1)
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[1, 32, 1, 1]" = var_mean_9[0]
        getitem_19: "f32[1, 32, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_50: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_9: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_9: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_19)
        mul_63: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_28: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[32]" = torch.ops.aten.mul.Tensor(primals_60, 0.9)
        add_51: "f32[32]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_66: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.000030518509476);  squeeze_29 = None
        mul_67: "f32[32]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[32]" = torch.ops.aten.mul.Tensor(primals_61, 0.9)
        add_52: "f32[32]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
        unsqueeze_37: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
        unsqueeze_39: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_53: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        relu_9: "f32[128, 32, 16, 16]" = torch.ops.aten.relu.default(add_53);  add_53 = None
        convolution_11: "f32[128, 32, 16, 16]" = torch.ops.aten.convolution.default(relu_9, primals_64, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_54: "i64[]" = torch.ops.aten.add.Tensor(primals_65, 1)
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[1, 32, 1, 1]" = var_mean_10[0]
        getitem_21: "f32[1, 32, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_55: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_10: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_10: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_21)
        mul_70: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_31: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72: "f32[32]" = torch.ops.aten.mul.Tensor(primals_66, 0.9)
        add_56: "f32[32]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_73: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.000030518509476);  squeeze_32 = None
        mul_74: "f32[32]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[32]" = torch.ops.aten.mul.Tensor(primals_67, 0.9)
        add_57: "f32[32]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
        unsqueeze_41: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
        unsqueeze_43: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_58: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:54 in forward, code: out = z + x
        add_59: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(add_58, relu_8);  add_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        relu_10: "f32[128, 32, 16, 16]" = torch.ops.aten.relu.default(add_59);  add_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        convolution_12: "f32[128, 32, 16, 16]" = torch.ops.aten.convolution.default(relu_10, primals_70, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_60: "i64[]" = torch.ops.aten.add.Tensor(primals_71, 1)
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[1, 32, 1, 1]" = var_mean_11[0]
        getitem_23: "f32[1, 32, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_61: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_11: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_11: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_23)
        mul_77: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_34: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_79: "f32[32]" = torch.ops.aten.mul.Tensor(primals_72, 0.9)
        add_62: "f32[32]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_80: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.000030518509476);  squeeze_35 = None
        mul_81: "f32[32]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[32]" = torch.ops.aten.mul.Tensor(primals_73, 0.9)
        add_63: "f32[32]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
        unsqueeze_45: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
        unsqueeze_47: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_64: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        relu_11: "f32[128, 32, 16, 16]" = torch.ops.aten.relu.default(add_64);  add_64 = None
        convolution_13: "f32[128, 32, 16, 16]" = torch.ops.aten.convolution.default(relu_11, primals_76, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_65: "i64[]" = torch.ops.aten.add.Tensor(primals_77, 1)
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[1, 32, 1, 1]" = var_mean_12[0]
        getitem_25: "f32[1, 32, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_66: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_12: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_12: "f32[128, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_25)
        mul_84: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_37: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86: "f32[32]" = torch.ops.aten.mul.Tensor(primals_78, 0.9)
        add_67: "f32[32]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_87: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.000030518509476);  squeeze_38 = None
        mul_88: "f32[32]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[32]" = torch.ops.aten.mul.Tensor(primals_79, 0.9)
        add_68: "f32[32]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
        unsqueeze_49: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[128, 32, 16, 16]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
        unsqueeze_51: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_69: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:54 in forward, code: out = z + x
        add_70: "f32[128, 32, 16, 16]" = torch.ops.aten.add.Tensor(add_69, relu_10);  add_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        relu_12: "f32[128, 32, 16, 16]" = torch.ops.aten.relu.default(add_70);  add_70 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        convolution_14: "f32[128, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_12, primals_82, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_71: "i64[]" = torch.ops.aten.add.Tensor(primals_83, 1)
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[1, 64, 1, 1]" = var_mean_13[0]
        getitem_27: "f32[1, 64, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_72: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_13: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_13: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_27)
        mul_91: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_40: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93: "f32[64]" = torch.ops.aten.mul.Tensor(primals_84, 0.9)
        add_73: "f32[64]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_94: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001220852154804);  squeeze_41 = None
        mul_95: "f32[64]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[64]" = torch.ops.aten.mul.Tensor(primals_85, 0.9)
        add_74: "f32[64]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
        unsqueeze_53: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
        unsqueeze_55: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_75: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        relu_13: "f32[128, 64, 8, 8]" = torch.ops.aten.relu.default(add_75);  add_75 = None
        convolution_15: "f32[128, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_13, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_76: "i64[]" = torch.ops.aten.add.Tensor(primals_89, 1)
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[1, 64, 1, 1]" = var_mean_14[0]
        getitem_29: "f32[1, 64, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_77: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_14: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_14: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_29)
        mul_98: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_43: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_99: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_100: "f32[64]" = torch.ops.aten.mul.Tensor(primals_90, 0.9)
        add_78: "f32[64]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_101: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001220852154804);  squeeze_44 = None
        mul_102: "f32[64]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[64]" = torch.ops.aten.mul.Tensor(primals_91, 0.9)
        add_79: "f32[64]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
        unsqueeze_57: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
        unsqueeze_59: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_80: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:53 in forward, code: x = self.downsample(x)
        convolution_16: "f32[128, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_12, primals_94, primals_95, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:54 in forward, code: out = z + x
        add_81: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(add_80, convolution_16);  add_80 = convolution_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        relu_14: "f32[128, 64, 8, 8]" = torch.ops.aten.relu.default(add_81);  add_81 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        convolution_17: "f32[128, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_14, primals_96, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_82: "i64[]" = torch.ops.aten.add.Tensor(primals_97, 1)
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[1, 64, 1, 1]" = var_mean_15[0]
        getitem_31: "f32[1, 64, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_83: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_15: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_15: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_31)
        mul_105: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_46: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[64]" = torch.ops.aten.mul.Tensor(primals_98, 0.9)
        add_84: "f32[64]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_108: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001220852154804);  squeeze_47 = None
        mul_109: "f32[64]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[64]" = torch.ops.aten.mul.Tensor(primals_99, 0.9)
        add_85: "f32[64]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1)
        unsqueeze_61: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1);  primals_101 = None
        unsqueeze_63: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_86: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        relu_15: "f32[128, 64, 8, 8]" = torch.ops.aten.relu.default(add_86);  add_86 = None
        convolution_18: "f32[128, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_15, primals_102, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_87: "i64[]" = torch.ops.aten.add.Tensor(primals_103, 1)
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[1, 64, 1, 1]" = var_mean_16[0]
        getitem_33: "f32[1, 64, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_88: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_16: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_16: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_33)
        mul_112: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_49: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114: "f32[64]" = torch.ops.aten.mul.Tensor(primals_104, 0.9)
        add_89: "f32[64]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_115: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001220852154804);  squeeze_50 = None
        mul_116: "f32[64]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[64]" = torch.ops.aten.mul.Tensor(primals_105, 0.9)
        add_90: "f32[64]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1)
        unsqueeze_65: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1);  primals_107 = None
        unsqueeze_67: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_91: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:54 in forward, code: out = z + x
        add_92: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(add_91, relu_14);  add_91 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        relu_16: "f32[128, 64, 8, 8]" = torch.ops.aten.relu.default(add_92);  add_92 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        convolution_19: "f32[128, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_16, primals_108, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_109, 1)
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[1, 64, 1, 1]" = var_mean_17[0]
        getitem_35: "f32[1, 64, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_94: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_17: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_17: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_35)
        mul_119: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_52: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_121: "f32[64]" = torch.ops.aten.mul.Tensor(primals_110, 0.9)
        add_95: "f32[64]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_122: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001220852154804);  squeeze_53 = None
        mul_123: "f32[64]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124: "f32[64]" = torch.ops.aten.mul.Tensor(primals_111, 0.9)
        add_96: "f32[64]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1)
        unsqueeze_69: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1);  primals_113 = None
        unsqueeze_71: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_97: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        relu_17: "f32[128, 64, 8, 8]" = torch.ops.aten.relu.default(add_97);  add_97 = None
        convolution_20: "f32[128, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_17, primals_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_115, 1)
        var_mean_18 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[1, 64, 1, 1]" = var_mean_18[0]
        getitem_37: "f32[1, 64, 1, 1]" = var_mean_18[1];  var_mean_18 = None
        add_99: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_18: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_18: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_37)
        mul_126: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_55: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_128: "f32[64]" = torch.ops.aten.mul.Tensor(primals_116, 0.9)
        add_100: "f32[64]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_129: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001220852154804);  squeeze_56 = None
        mul_130: "f32[64]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131: "f32[64]" = torch.ops.aten.mul.Tensor(primals_117, 0.9)
        add_101: "f32[64]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1)
        unsqueeze_73: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1);  primals_119 = None
        unsqueeze_75: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_102: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:54 in forward, code: out = z + x
        add_103: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(add_102, relu_16);  add_102 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        relu_18: "f32[128, 64, 8, 8]" = torch.ops.aten.relu.default(add_103);  add_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:139 in forward, code: x = self.output_net(x)
        mean: "f32[128, 64, 1, 1]" = torch.ops.aten.mean.dim(relu_18, [-1, -2], True)
        view: "f32[128, 64]" = torch.ops.aten.reshape.default(mean, [128, 64]);  mean = None
        permute: "f32[64, 10]" = torch.ops.aten.permute.default(primals_120, [1, 0])
        addmm: "f32[128, 10]" = torch.ops.aten.addmm.default(primals_121, view, permute);  primals_121 = permute = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        le: "b8[128, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        unsqueeze_76: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_77: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 2);  unsqueeze_76 = None
        unsqueeze_78: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, 3);  unsqueeze_77 = None
        unsqueeze_88: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_89: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 2);  unsqueeze_88 = None
        unsqueeze_90: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_89, 3);  unsqueeze_89 = None
        unsqueeze_100: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_101: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, 2);  unsqueeze_100 = None
        unsqueeze_102: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 3);  unsqueeze_101 = None
        unsqueeze_112: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_113: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 2);  unsqueeze_112 = None
        unsqueeze_114: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 3);  unsqueeze_113 = None
        unsqueeze_124: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_125: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 2);  unsqueeze_124 = None
        unsqueeze_126: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 3);  unsqueeze_125 = None
        unsqueeze_136: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_137: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 2);  unsqueeze_136 = None
        unsqueeze_138: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 3);  unsqueeze_137 = None
        unsqueeze_148: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_149: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
        unsqueeze_150: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
        unsqueeze_160: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_161: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
        unsqueeze_162: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
        unsqueeze_172: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_173: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
        unsqueeze_174: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
        unsqueeze_184: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_185: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
        unsqueeze_186: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
        unsqueeze_196: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_197: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
        unsqueeze_198: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
        unsqueeze_208: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_209: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
        unsqueeze_210: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
        unsqueeze_220: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_221: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
        unsqueeze_222: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
        unsqueeze_232: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_233: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
        unsqueeze_234: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
        unsqueeze_244: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_245: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
        unsqueeze_246: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
        unsqueeze_256: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_257: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
        unsqueeze_258: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
        unsqueeze_268: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_269: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
        unsqueeze_270: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
        unsqueeze_280: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_281: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
        unsqueeze_282: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:137 in forward, code: x = self.input_net(x)
        unsqueeze_292: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_293: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
        unsqueeze_294: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None

        # No stacktrace found for following nodes
        copy_: "i64[]" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[16]" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[16]" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_5);  primals_9 = add_5 = copy__3 = None
        copy__4: "f32[16]" = torch.ops.aten.copy_.default(primals_10, add_7);  primals_10 = add_7 = copy__4 = None
        copy__5: "f32[16]" = torch.ops.aten.copy_.default(primals_11, add_8);  primals_11 = add_8 = copy__5 = None
        copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_15, add_10);  primals_15 = add_10 = copy__6 = None
        copy__7: "f32[16]" = torch.ops.aten.copy_.default(primals_16, add_12);  primals_16 = add_12 = copy__7 = None
        copy__8: "f32[16]" = torch.ops.aten.copy_.default(primals_17, add_13);  primals_17 = add_13 = copy__8 = None
        copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_21, add_16);  primals_21 = add_16 = copy__9 = None
        copy__10: "f32[16]" = torch.ops.aten.copy_.default(primals_22, add_18);  primals_22 = add_18 = copy__10 = None
        copy__11: "f32[16]" = torch.ops.aten.copy_.default(primals_23, add_19);  primals_23 = add_19 = copy__11 = None
        copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_27, add_21);  primals_27 = add_21 = copy__12 = None
        copy__13: "f32[16]" = torch.ops.aten.copy_.default(primals_28, add_23);  primals_28 = add_23 = copy__13 = None
        copy__14: "f32[16]" = torch.ops.aten.copy_.default(primals_29, add_24);  primals_29 = add_24 = copy__14 = None
        copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_33, add_27);  primals_33 = add_27 = copy__15 = None
        copy__16: "f32[16]" = torch.ops.aten.copy_.default(primals_34, add_29);  primals_34 = add_29 = copy__16 = None
        copy__17: "f32[16]" = torch.ops.aten.copy_.default(primals_35, add_30);  primals_35 = add_30 = copy__17 = None
        copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_39, add_32);  primals_39 = add_32 = copy__18 = None
        copy__19: "f32[16]" = torch.ops.aten.copy_.default(primals_40, add_34);  primals_40 = add_34 = copy__19 = None
        copy__20: "f32[16]" = torch.ops.aten.copy_.default(primals_41, add_35);  primals_41 = add_35 = copy__20 = None
        copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_45, add_38);  primals_45 = add_38 = copy__21 = None
        copy__22: "f32[32]" = torch.ops.aten.copy_.default(primals_46, add_40);  primals_46 = add_40 = copy__22 = None
        copy__23: "f32[32]" = torch.ops.aten.copy_.default(primals_47, add_41);  primals_47 = add_41 = copy__23 = None
        copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_51, add_43);  primals_51 = add_43 = copy__24 = None
        copy__25: "f32[32]" = torch.ops.aten.copy_.default(primals_52, add_45);  primals_52 = add_45 = copy__25 = None
        copy__26: "f32[32]" = torch.ops.aten.copy_.default(primals_53, add_46);  primals_53 = add_46 = copy__26 = None
        copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_59, add_49);  primals_59 = add_49 = copy__27 = None
        copy__28: "f32[32]" = torch.ops.aten.copy_.default(primals_60, add_51);  primals_60 = add_51 = copy__28 = None
        copy__29: "f32[32]" = torch.ops.aten.copy_.default(primals_61, add_52);  primals_61 = add_52 = copy__29 = None
        copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_65, add_54);  primals_65 = add_54 = copy__30 = None
        copy__31: "f32[32]" = torch.ops.aten.copy_.default(primals_66, add_56);  primals_66 = add_56 = copy__31 = None
        copy__32: "f32[32]" = torch.ops.aten.copy_.default(primals_67, add_57);  primals_67 = add_57 = copy__32 = None
        copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_71, add_60);  primals_71 = add_60 = copy__33 = None
        copy__34: "f32[32]" = torch.ops.aten.copy_.default(primals_72, add_62);  primals_72 = add_62 = copy__34 = None
        copy__35: "f32[32]" = torch.ops.aten.copy_.default(primals_73, add_63);  primals_73 = add_63 = copy__35 = None
        copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_77, add_65);  primals_77 = add_65 = copy__36 = None
        copy__37: "f32[32]" = torch.ops.aten.copy_.default(primals_78, add_67);  primals_78 = add_67 = copy__37 = None
        copy__38: "f32[32]" = torch.ops.aten.copy_.default(primals_79, add_68);  primals_79 = add_68 = copy__38 = None
        copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_83, add_71);  primals_83 = add_71 = copy__39 = None
        copy__40: "f32[64]" = torch.ops.aten.copy_.default(primals_84, add_73);  primals_84 = add_73 = copy__40 = None
        copy__41: "f32[64]" = torch.ops.aten.copy_.default(primals_85, add_74);  primals_85 = add_74 = copy__41 = None
        copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_89, add_76);  primals_89 = add_76 = copy__42 = None
        copy__43: "f32[64]" = torch.ops.aten.copy_.default(primals_90, add_78);  primals_90 = add_78 = copy__43 = None
        copy__44: "f32[64]" = torch.ops.aten.copy_.default(primals_91, add_79);  primals_91 = add_79 = copy__44 = None
        copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_97, add_82);  primals_97 = add_82 = copy__45 = None
        copy__46: "f32[64]" = torch.ops.aten.copy_.default(primals_98, add_84);  primals_98 = add_84 = copy__46 = None
        copy__47: "f32[64]" = torch.ops.aten.copy_.default(primals_99, add_85);  primals_99 = add_85 = copy__47 = None
        copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_103, add_87);  primals_103 = add_87 = copy__48 = None
        copy__49: "f32[64]" = torch.ops.aten.copy_.default(primals_104, add_89);  primals_104 = add_89 = copy__49 = None
        copy__50: "f32[64]" = torch.ops.aten.copy_.default(primals_105, add_90);  primals_105 = add_90 = copy__50 = None
        copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_109, add_93);  primals_109 = add_93 = copy__51 = None
        copy__52: "f32[64]" = torch.ops.aten.copy_.default(primals_110, add_95);  primals_110 = add_95 = copy__52 = None
        copy__53: "f32[64]" = torch.ops.aten.copy_.default(primals_111, add_96);  primals_111 = add_96 = copy__53 = None
        copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_115, add_98);  primals_115 = add_98 = copy__54 = None
        copy__55: "f32[64]" = torch.ops.aten.copy_.default(primals_116, add_100);  primals_116 = add_100 = copy__55 = None
        copy__56: "f32[64]" = torch.ops.aten.copy_.default(primals_117, add_101);  primals_117 = add_101 = copy__56 = None
        return (addmm, primals_1, primals_2, primals_6, primals_8, primals_12, primals_14, primals_18, primals_20, primals_24, primals_26, primals_30, primals_32, primals_36, primals_38, primals_42, primals_44, primals_48, primals_50, primals_54, primals_56, primals_58, primals_62, primals_64, primals_68, primals_70, primals_74, primals_76, primals_80, primals_82, primals_86, primals_88, primals_92, primals_94, primals_96, primals_100, primals_102, primals_106, primals_108, primals_112, primals_114, primals_118, primals_120, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, relu_5, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, relu_7, convolution_8, squeeze_25, relu_8, convolution_10, squeeze_28, relu_9, convolution_11, squeeze_31, relu_10, convolution_12, squeeze_34, relu_11, convolution_13, squeeze_37, relu_12, convolution_14, squeeze_40, relu_13, convolution_15, squeeze_43, relu_14, convolution_17, squeeze_46, relu_15, convolution_18, squeeze_49, relu_16, convolution_19, squeeze_52, relu_17, convolution_20, squeeze_55, view, le, unsqueeze_78, unsqueeze_90, unsqueeze_102, unsqueeze_114, unsqueeze_126, unsqueeze_138, unsqueeze_150, unsqueeze_162, unsqueeze_174, unsqueeze_186, unsqueeze_198, unsqueeze_210, unsqueeze_222, unsqueeze_234, unsqueeze_246, unsqueeze_258, unsqueeze_270, unsqueeze_282, unsqueeze_294)
